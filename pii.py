"""Маскировка персональных данных (152-ФЗ) перед записью в БД и отправкой в LLM.
Слой 1: regex — email, телефоны РФ, СНИЛС, ИНН, паспорт, карты.
Слой 2: cloakllm Shield (spaCy NER) — ФИО, организации, геолокации.
Если cloakllm недоступен — работает только regex, предупреждение выводится 1 раз.
"""
import re
import logging

logger = logging.getLogger(__name__)

# ── Regex-правила ──
_EMAIL_RE    = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SNILS_RE    = re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}[\s-]?\d{2}(?!\d)")
_PHONE_RE    = re.compile(
    r"(?<!\d)(?:\+7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-()]?\d{2}[\s\-()]?\d{2}(?!\d)"
)
_PASSPORT_RE = re.compile(r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)")
_CARD_RE     = re.compile(r"(?<!\d)(?:\d[ -]?){16}(?!\d)")
_INN_RE      = re.compile(r"(?<!\d)\d{12}(?!\d)")

_RULES = [
    (_EMAIL_RE,    "[EMAIL]"),
    (_CARD_RE,     "[CARD]"),
    (_SNILS_RE,    "[SNILS]"),
    (_PASSPORT_RE, "[PASSPORT]"),
    (_PHONE_RE,    "[PHONE]"),
    (_INN_RE,      "[INN]"),
]

# ── cloakllm Shield (ленивая инициализация) ──
_shield = None
_shield_tried = False


def _get_shield():
    global _shield, _shield_tried
    if _shield_tried:
        return _shield
    _shield_tried = True
    try:
        from cloakllm import Shield, ShieldConfig
        _shield = Shield(ShieldConfig())
        logger.info("PII: cloakllm Shield инициализирован (NER-маскировка ФИО активна).")
    except ImportError:
        logger.warning(
            "PII: пакет cloakllm не найден — ФИО маскироваться НЕ будут "
            "(только regex). Установите cloakllm + python -m spacy download ru_core_news_sm."
        )
        _shield = None
    except Exception as e:
        logger.exception("PII: ошибка инициализации cloakllm Shield: %s", e)
        _shield = None
    return _shield


def mask_pii(text: str) -> str:
    """Замаскировать ПД. Идемпотентно: уже подставленные [TOKEN] не ломаются."""
    if not text:
        return text

    # Слой 1: regex (быстро, детерминированно)
    for rx, repl in _RULES:
        text = rx.sub(repl, text)

    # Слой 2: cloakllm NER (ФИО, организации, геолокации)
    shield = _get_shield()
    if shield is not None:
        try:
            cloaked, _ = shield.sanitize(text)
            text = cloaked
        except Exception:
            logger.exception("PII: ошибка при вызове Shield.sanitize, используется regex-результат")

    return text


def contains_pii(text: str) -> bool:
    return bool(text) and mask_pii(text) != text


def warmup() -> None:
    """Предзагрузка Shield на старте процесса (избегает задержки при первом вызове)."""
    _get_shield()
