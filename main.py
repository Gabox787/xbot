"""
Автономный Telegram-бот для публикации крипто-новостей.

Пайплайн:
1. Парсинг сырых твитов (fetch_new_posts)
2. Фильтрация уже опубликованных твитов по локальной SQLite-базе
3. Генерация текста поста + промпта для картинки через GPT-4o-mini (JSON-ответ)
4. Генерация картинки через DALL-E 3
5. Публикация в Telegram-канал (sendPhoto)
6. Сохранение ID твита в базу, чтобы не постить дубликаты

Предназначен для запуска на Render как Web Service: поднимает лёгкий HTTP-сервер
(для health-check запросов от Render/UptimeRobot, чтобы сервис не засыпал), а сам
пайплайн бота крутится в фоновом потоке.
"""

import os
import sys
import json
import time
import sqlite3
import logging
import threading
from datetime import datetime, timezone

import requests
import schedule
from flask import Flask
from openai import OpenAI
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Настройка окружения и логирования
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("crypto_bot")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Необязательные параметры со значениями по умолчанию
RSS_SOURCE_URL = os.getenv("RSS_SOURCE_URL", "")  # URL источника твитов (RSS/JSON API)
DB_PATH = os.getenv("DB_PATH", "posts.db")
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "15"))
PORT = int(os.getenv("PORT", "10000"))  # Render передаёт актуальный порт через PORT

REQUIRED_ENV_VARS = ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "OPENAI_API_KEY"]

# Мини веб-сервер только для health-check запросов (Render / UptimeRobot).
# Никакой бизнес-логики здесь нет — вся работа бота идёт в фоновом потоке.
app = Flask(__name__)


@app.route("/")
def health_check():
    return "Bot is alive", 200

SYSTEM_PROMPT = (
    "Ты — профессиональный криптовалютный аналитик, редактор Telegram-канала "
    "и ИИ-художник. Твоя задача: проанализировать сырой твит о криптовалюте и "
    "создать из него два элемента.\n"
    "ПРАВИЛА ДЛЯ ТЕКСТА: Извлекай только факты. Тон: объективный. Язык: "
    "Русский (с крипто-сленгом). Структура строго: [Эмодзи] [ЗАГОЛОВОК КАПСОМ] "
    "\n\n 🔹 Суть: [1-2 предложения] \n 💡 Импакт: [1 предложение о влиянии на "
    "рынок].\n"
    "ПРАВИЛА ДЛЯ КАРТИНКИ: Промпт СТРОГО НА АНГЛИЙСКОМ. Описывай атмосферную, "
    "стильную иллюстрацию (cyberpunk, 3D render, neon). НИКАКОГО ТЕКСТА на "
    "картинке. Длина: 30-50 слов.\n"
    'ВЫВОД: ТОЛЬКО JSON: { "telegram_text": "текст", "image_prompt": "промпт" }'
)

client: OpenAI | None = None  # инициализируется в main() после проверки ключей


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

def validate_env() -> None:
    """Проверяет наличие обязательных переменных окружения."""
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        logger.error(
            "Отсутствуют обязательные переменные окружения: %s",
            ", ".join(missing),
        )
        sys.exit(1)


def init_db() -> None:
    """Создаёт таблицу опубликованных постов, если её ещё нет."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS published_posts (
                tweet_id TEXT PRIMARY KEY,
                published_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Работа с базой данных
# ---------------------------------------------------------------------------

def is_already_published(tweet_id: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM published_posts WHERE tweet_id = ?", (tweet_id,)
        )
        return cur.fetchone() is not None
    finally:
        conn.close()


def mark_as_published(tweet_id: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT OR IGNORE INTO published_posts (tweet_id, published_at) "
            "VALUES (?, ?)",
            (tweet_id, datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Шаг 1: Парсинг источника твитов
# ---------------------------------------------------------------------------

def fetch_new_posts() -> list[dict]:
    """
    Получает сырые посты из источника (RSS/JSON API).

    Ожидаемый формат каждого элемента: {"id": "уникальный_id", "text": "текст твита"}.

    Если RSS_SOURCE_URL не задан, функция работает как заглушка и возвращает
    пустой список — подставь сюда свой парсер (Nitter RSS, сторонний API,
    Twitter/X API и т.д.).
    """
    if not RSS_SOURCE_URL:
        logger.warning(
            "RSS_SOURCE_URL не задан — fetch_new_posts() работает как заглушка "
            "и не возвращает постов. Подключи свой источник данных."
        )
        return []

    try:
        response = requests.get(RSS_SOURCE_URL, timeout=15)
        response.raise_for_status()
        data = response.json()

        posts = []
        for item in data.get("items", []):
            tweet_id = item.get("id")
            text = item.get("text", "")
            if tweet_id and text:
                posts.append({"id": str(tweet_id), "text": text})
        return posts
    except Exception as exc:
        logger.error("Ошибка при получении твитов из источника: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Шаг 2: Генерация текста и промпта для картинки (GPT-4o-mini)
# ---------------------------------------------------------------------------

def generate_content(raw_tweet_text: str) -> dict | None:
    """Отправляет сырой твит в GPT-4o-mini и получает JSON с текстом и image_prompt."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            response_format={"type": "json_object"},
            temperature=0.7,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_tweet_text},
            ],
        )
        raw_content = response.choices[0].message.content
        data = json.loads(raw_content)

        if "telegram_text" not in data or "image_prompt" not in data:
            logger.error("Ответ OpenAI не содержит нужных полей: %s", data)
            return None

        return data
    except Exception as exc:
        logger.error("Ошибка генерации контента через GPT-4o-mini: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Шаг 3: Генерация картинки (DALL-E 3)
# ---------------------------------------------------------------------------

def generate_image(image_prompt: str) -> str | None:
    """Генерирует изображение через DALL-E 3 и возвращает его URL."""
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url
    except Exception as exc:
        logger.error("Ошибка генерации изображения через DALL-E 3: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Шаг 4: Публикация в Telegram
# ---------------------------------------------------------------------------

def publish_to_telegram(image_url: str, caption: str) -> bool:
    """Публикует картинку с подписью в Telegram-канал через метод sendPhoto."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "photo": image_url,
        "caption": caption[:1024],  # Telegram ограничивает caption 1024 символами
        "parse_mode": "HTML",
    }
    try:
        response = requests.post(url, data=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        if not result.get("ok"):
            logger.error("Telegram API вернул ошибку: %s", result)
            return False
        return True
    except Exception as exc:
        logger.error("Ошибка при публикации в Telegram: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Основной пайплайн
# ---------------------------------------------------------------------------

def process_pipeline() -> None:
    """Полный цикл: парсинг -> фильтрация -> генерация -> публикация -> сохранение."""
    logger.info("Запуск нового цикла проверки твитов...")

    posts = fetch_new_posts()
    if not posts:
        logger.info("Новых постов не найдено.")
        return

    for post in posts:
        tweet_id = post.get("id")
        raw_text = post.get("text", "")

        if not tweet_id or not raw_text:
            continue

        if is_already_published(tweet_id):
            logger.info("Твит %s уже был опубликован ранее — пропуск.", tweet_id)
            continue

        logger.info("Обработка нового твита: %s", tweet_id)

        content = generate_content(raw_text)
        if content is None:
            logger.warning(
                "Не удалось сгенерировать текст для твита %s, пропуск.", tweet_id
            )
            continue

        time.sleep(2)  # пауза перед генерацией картинки (rate limit)

        image_url = generate_image(content["image_prompt"])
        if image_url is None:
            logger.warning(
                "Не удалось сгенерировать картинку для твита %s, пропуск.", tweet_id
            )
            continue

        time.sleep(2)  # пауза перед публикацией (rate limit)

        success = publish_to_telegram(image_url, content["telegram_text"])
        if success:
            mark_as_published(tweet_id)
            logger.info("Твит %s успешно опубликован.", tweet_id)
        else:
            logger.warning(
                "Публикация твита %s не удалась, будет повторена на следующей "
                "итерации.",
                tweet_id,
            )

        time.sleep(5)  # пауза между постами


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------

def run_bot_loop() -> None:
    """Бесконечный цикл пайплайна бота. Выполняется в фоновом потоке."""
    logger.info("Бот запущен. Выполняется первый запуск пайплайна...")
    try:
        process_pipeline()
    except Exception as exc:
        logger.error("Ошибка в первом запуске пайплайна: %s", exc)

    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(process_pipeline)
    logger.info(
        "Бот будет проверять новые посты каждые %s минут.", CHECK_INTERVAL_MINUTES
    )

    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except Exception as exc:
            # Бот никогда не должен падать целиком — логируем и продолжаем работу
            logger.error("Непредвиденная ошибка в основном цикле: %s", exc)
            time.sleep(10)


def main() -> None:
    global client

    validate_env()
    client = OpenAI(api_key=OPENAI_API_KEY)
    init_db()

    # Пайплайн бота работает в отдельном потоке, чтобы не блокировать Flask —
    # Render считает сервис "живым", пока отвечает HTTP-порт.
    bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
    bot_thread.start()

    logger.info("Запускаю HTTP-сервер на порту %s для health-check.", PORT)
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
