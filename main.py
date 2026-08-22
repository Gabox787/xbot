"""
Автономный Telegram-бот для публикации крипто-новостей.

Пайплайн:
1. Парсинг сырых твитов (fetch_new_posts)
2. Фильтрация уже опубликованных твитов по локальной SQLite-базе
3. Генерация текста поста + промпта для картинки через Gemini 3.6 Flash (JSON-ответ)
4. Генерация картинки через Pollinations.ai (бесплатно, без ключа)
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
import html
import sqlite3
import logging
import threading
import urllib.parse
from datetime import datetime, timezone

import requests
import schedule
import feedparser
from flask import Flask
from google import genai
from google.genai import types
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Необязательные параметры со значениями по умолчанию
RSS_SOURCE_URL = os.getenv("RSS_SOURCE_URL", "")  # URL источника твитов (RSS/JSON API)
DB_PATH = os.getenv("DB_PATH", "posts.db")
CHECK_INTERVAL_MINUTES = int(os.getenv("CHECK_INTERVAL_MINUTES", "15"))
PORT = int(os.getenv("PORT", "10000"))  # Render передаёт актуальный порт через PORT

REQUIRED_ENV_VARS = ["TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "GEMINI_API_KEY"]
TELEGRAM_CAPTION_LIMIT = 1024  # ограничение Telegram на длину caption у sendPhoto

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

client: genai.Client | None = None  # инициализируется в main() после проверки ключей


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
    Получает свежие записи из RSS-ленты (например, крипто-новостного издания).

    Ожидаемый формат каждого элемента после парсинга: {"id": ..., "text": ..., "link": "ссылка на статью"}.

    Если RSS_SOURCE_URL не задан, функция работает как заглушка и возвращает
    пустой список.
    """
    if not RSS_SOURCE_URL:
        logger.warning(
            "RSS_SOURCE_URL не задан — fetch_new_posts() работает как заглушка "
            "и не возвращает постов. Укажи ссылку на RSS-ленту в переменных окружения."
        )
        return []

    try:
        feed = feedparser.parse(RSS_SOURCE_URL)

        if feed.bozo and not feed.entries:
            logger.error("Не удалось разобрать RSS-ленту: %s", feed.bozo_exception)
            return []

        posts = []
        for entry in feed.entries:
            link = entry.get("link", "")
            # id записи: сначала пробуем guid/id, иначе берём ссылку на статью
            post_id = entry.get("id") or link
            # текст: краткое описание, если его нет — заголовок
            text = entry.get("summary") or entry.get("title", "")

            if post_id and text:
                posts.append({"id": str(post_id), "text": text, "link": link})

        return posts
    except Exception as exc:
        logger.error("Ошибка при получении твитов из источника: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Шаг 2: Генерация текста и промпта для картинки (Gemini 2.5 Flash)
# ---------------------------------------------------------------------------

def generate_content(raw_tweet_text: str) -> dict | None:
    """Отправляет сырой твит в Gemini 3.6 Flash и получает JSON с текстом и image_prompt."""
    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=raw_tweet_text,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                temperature=0.7,
            ),
        )
        data = json.loads(response.text)

        if "telegram_text" not in data or "image_prompt" not in data:
            logger.error("Ответ Gemini не содержит нужных полей: %s", data)
            return None

        return data
    except Exception as exc:
        logger.error("Ошибка генерации контента через Gemini 3.6 Flash: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Шаг 3: Генерация картинки (Pollinations.ai — бесплатно, без ключа)
# ---------------------------------------------------------------------------

def generate_image(image_prompt: str) -> bytes | None:
    """Генерирует изображение через Pollinations.ai и возвращает его байты.

    Pollinations.ai не требует API-ключа и не имеет платного порога, в отличие
    от текущих моделей Gemini image (Nano Banana), которые с середины 2026 года
    стали платными без бесплатного тарифа.
    """
    try:
        encoded_prompt = urllib.parse.quote(image_prompt)
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}"
        response = requests.get(
            url,
            params={"width": 1024, "height": 1024, "nologo": "true"},
            timeout=60,
        )
        response.raise_for_status()
        return response.content
    except Exception as exc:
        logger.error("Ошибка генерации изображения через Pollinations.ai: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Шаг 4: Публикация в Telegram
# ---------------------------------------------------------------------------

def publish_to_telegram(image_bytes: bytes, caption: str) -> bool:
    """Публикует картинку (байты) с подписью в Telegram-канал через метод sendPhoto."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "caption": caption,
        "parse_mode": "HTML",
    }
    files = {"photo": ("image.png", image_bytes, "image/png")}
    try:
        response = requests.post(url, data=data, files=files, timeout=60)
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
# Вспомогательное: красивое имя источника по ссылке
# ---------------------------------------------------------------------------

def get_source_name(url: str) -> str:
    """Извлекает и форматирует название источника из домена ссылки.

    Например: https://cointelegraph.com/news/... -> "Cointelegraph".
    """
    try:
        netloc = urllib.parse.urlparse(url).netloc
        netloc = netloc.replace("www.", "")
        name = netloc.split(".")[0]
        return name.capitalize() if name else "Источник"
    except Exception:
        return "Источник"


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

        image_bytes = generate_image(content["image_prompt"])
        if image_bytes is None:
            logger.warning(
                "Не удалось сгенерировать картинку для твита %s, пропуск.", tweet_id
            )
            continue

        time.sleep(2)  # пауза перед публикацией (rate limit)

        source_link = post.get("link", "")
        main_text = content["telegram_text"]
        footer = ""
        if source_link:
            source_name = get_source_name(source_link)
            safe_link = html.escape(source_link, quote=True)
            footer = f'\n\n🔗 Источник: <a href="{safe_link}">{source_name}</a>'

        # Обрезаем именно основной текст, а не готовую строку с футером —
        # иначе можно случайно разорвать HTML-тег ссылки при обрезке по лимиту.
        max_text_length = TELEGRAM_CAPTION_LIMIT - len(footer)
        if len(main_text) > max_text_length:
            main_text = main_text[: max_text_length - 1].rstrip() + "…"

        caption = main_text + footer

        success = publish_to_telegram(image_bytes, caption)
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
    client = genai.Client(api_key=GEMINI_API_KEY)
    init_db()

    # Пайплайн бота работает в отдельном потоке, чтобы не блокировать Flask —
    # Render считает сервис "живым", пока отвечает HTTP-порт.
    bot_thread = threading.Thread(target=run_bot_loop, daemon=True)
    bot_thread.start()

    logger.info("Запускаю HTTP-сервер на порту %s для health-check.", PORT)
    app.run(host="0.0.0.0", port=PORT)


if __name__ == "__main__":
    main()
