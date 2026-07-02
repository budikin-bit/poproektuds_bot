import os
from dotenv import load_dotenv
import pytz

load_dotenv()

# ---------- Токен бота ----------
MAX_BOT_TOKEN = os.getenv("MAX_BOT_TOKEN")

# ---------- Контроль доступа ----------
# Список user_id, которым разрешено пользоваться ботом (через запятую).
# Если пусто — бот закрыт для всех (безопасный дефолт, чтобы не забыть включить).
_allowed_raw = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = {
    item.strip() for item in _allowed_raw.split(",") if item.strip()
}

# ---------- OpenModel (Anthropic-совместимый шлюз) ----------
OPENMODEL_API_KEY = os.getenv("OPENMODEL_API_KEY")
OPENMODEL_BASE_URL = os.getenv("OPENMODEL_BASE_URL", "https://api.openmodel.ai")
MODEL = os.getenv("MODEL", "deepseek-v4-flash")

# Параметры генерации
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "24000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# ---------- Настройки бота ----------
ANALYSIS_BLOCKS = int(os.getenv("ANALYSIS_BLOCKS", "5"))      # сколько блоков в экспресс-анализе
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "300"))            # таймаут на вызов LLM (сек)
MASK_PII = os.getenv("MASK_PII", "true").lower() == "true"    # маскировать ли персональные данные

# ---------- База данных ----------
DB_PATH = os.getenv("DB_PATH", "data/bot.db")                 # путь к файлу SQLite
ALLOW_MEMORY_STORAGE = os.getenv("ALLOW_MEMORY_STORAGE", "0").lower() in ("1", "true", "yes")

# ---------- Дневной лимит ----------
DAILY_LIMIT = int(os.getenv("DAILY_LIMIT", "10"))             # бесплатных разборов в день
TIMEZONE = pytz.timezone(os.getenv("TIMEZONE", "Europe/Moscow"))

# ---------- Системный промпт (из файла) ----------
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.md")
try:
    with open(_PROMPT_PATH, encoding="utf-8") as _f:
        SYSTEM_PROMPT = _f.read().strip()
except OSError as _e:
    import logging as _logging
    _logging.getLogger(__name__).critical(
        "Не удалось прочитать системный промпт: %s (%s). "
        "Создайте файл prompts/system_prompt.md рядом с config.py — "
        "без него бот не может формировать анализ. Останавливаюсь.",
        _PROMPT_PATH, _e,
    )
    raise SystemExit(1)

if not SYSTEM_PROMPT:
    import logging as _logging
    _logging.getLogger(__name__).critical(
        "Файл системного промпта пуст: %s. Заполните его — "
        "без промпта бот не может формировать анализ. Останавливаюсь.",
        _PROMPT_PATH,
    )
    raise SystemExit(1)
