import os
import logging
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Хелперы ──
def _bool(name, default=False):
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

def _int(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise RuntimeError(f"Переменная {name}={raw!r} должна быть целым числом")

def _float(name, default):
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise RuntimeError(f"Переменная {name}={raw!r} должна быть числом")

# ── Бот / хранилище ──
MAX_BOT_TOKEN = os.getenv("MAX_BOT_TOKEN")
DB_PATH = os.getenv("DB_PATH")
ALLOW_MEMORY_STORAGE = _bool("ALLOW_MEMORY_STORAGE", False)
TIMEZONE = ZoneInfo(os.getenv("TIMEZONE", "Asia/Yakutsk"))

# ── LLM (универсальный) ──
# Стиль API: "openai", "yandex", "anthropic" – задаётся в .env
LLM_API_STYLE = os.getenv("LLM_API_STYLE", "anthropic").lower()

# Ключ (приоритет: LLM_API_KEY, затем специфичные)
LLM_API_KEY = (
    os.getenv("LLM_API_KEY")
    or os.getenv("OPENMODEL_API_KEY")
    or os.getenv("DEEPSEEK_API_KEY")
    or os.getenv("YANDEX_API_KEY")
)

# Базовый URL – подставляется умолчание в зависимости от стиля, если не задан явно
_DEFAULT_BASE_URL = {
    "openai": "https://api.deepseek.com/v1",           # можно переопределить
    "yandex": "https://llm.api.cloud.yandex.net/v1",
    "anthropic": "https://api.openmodel.ai",
}.get(LLM_API_STYLE, "https://api.openmodel.ai")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", _DEFAULT_BASE_URL)

# Модель – обязательна, проверьте в документации вашего провайдера
MODEL = os.getenv("MODEL", "deepseek-v4-flash")

MAX_TOKENS = _int("MAX_TOKENS", 4096)
TEMPERATURE = _float("TEMPERATURE", 0.3)
LLM_TIMEOUT = _int("LLM_TIMEOUT", 600)           # общий таймаут всей операции
LLM_REQUEST_TIMEOUT = _int("LLM_REQUEST_TIMEOUT", 120)  # таймаут на HTTP-запрос
LLM_RETRIES = _int("LLM_RETRIES", 3)

# ── Приватность (152-ФЗ) ──
MASK_PII = _bool("MASK_PII", True)
PRIVACY_POLICY_VERSION = os.getenv("PRIVACY_POLICY_VERSION", "v1")

# ── Анализ ──
ANALYSIS_BLOCKS = _int("ANALYSIS_BLOCKS", 5)

# ── Дневной лимит ──
DAILY_LIMIT = _int("DAILY_LIMIT", 10)   # 0 = без лимита

# ── Системный промпт ──
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.md")
try:
    with open(_PROMPT_PATH, encoding="utf-8") as _f:
        SYSTEM_PROMPT = _f.read().strip()
except FileNotFoundError:
    raise RuntimeError(
        f"Не найден системный промпт: {_PROMPT_PATH}. "
        "Положите prompts/system_prompt.md рядом с config.py."
    )

# При необходимости можно добавить функцию обрезки промпта для бесплатной версии,
# но сейчас оставим как есть.