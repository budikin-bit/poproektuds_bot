"""Маскировка персональных данных (152-ФЗ) перед записью в БД и отправкой в LLM.
Базовый слой — регулярки (e-mail, телефоны РФ, СНИЛС, ИНН, паспорт, карты).
Если установлен пакет `cloakllm`, используется его функция маскировки (включая ФИО).
Иначе работает только regex (об этом предупреждаем 1 раз).
"""
import re
import logging

logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_SNILS_RE = re.compile(r"(?<!\d)\d{3}-\d{3}-\d{3}[\s-]?\d{2}(?!\d)")
_PHONE_RE = re.compile(
    r"(?<!\d)(?:\+7|8)[\s\-()]?\d{3}[\s\-()]?\d{3}[\s\-()]?\d{2}[\s\-()]?\d{2}(?!\d)"
)
_PASSPORT_RE = re.compile(r"(?<!\d)\d{2}\s?\d{2}\s?\d{6}(?!\d)")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){16}(?!\d)")
_INN_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")  # ИНН физлица; 10-значные не трогаем

_RULES = [
    (_EMAIL_RE, "[EMAIL]"),
    (_CARD_RE, "[CARD]"),
    (_SNILS_RE, "[SNILS]"),
    (_PASSPORT_RE, "[PASSPORT]"),
    (_PHONE_RE, "[PHONE]"),
    (_INN_RE, "[INN]"),
]

# ── Опциональный cloakllm (заменяет natasha) ──
_cloakllm = None
_cloakllm_tried = False


def _get_cloakllm():
    global _cloakllm, _cloakllm_tried
    if _cloakllm_tried:
        return _cloakllm
    _cloakllm_tried = True
    try:
        import cloakllm
        # Предполагаем, что cloakllm предоставляет функцию mask_pii(text) -> str
        if hasattr(cloakllm, "mask_pii"):
            _cloakllm = cloakllm.mask_pii
            logger.info("PII: cloakllm.mask_pii подключён (маскировка ФИО активна).")
        else:
            logger.warning(
                "PII: cloakllm найден, но не имеет функции mask_pii. "
                "Будет использован только regex."
            )
            _cloakllm = None
    except ImportError:
        logger.warning(
            "PII: пакет cloakllm не найден — ФИО маскироваться НЕ будут "
            "(только e-mail/телефон/СНИЛС/паспорт/ИНН/карты). "
            "Установите cloakllm для полноценной защиты."
        )
        _cloakllm = None
    except Exception as e:
        logger.exception("PII: ошибка инициализации cloakllm: %s", e)
        _cloakllm = None
    return _cloakllm


def _regex_mask(text: str) -> str:
    for rx, repl in _RULES:
        text = rx.sub(repl, text)
    return text


def mask_pii(text: str) -> str:
    """Замаскировать ПД. Идемпотентно: уже подставленные [TOKEN] не ломаются."""
    if not text:
        return text

    # Сначала применяем регулярки (быстро)
    text = _regex_mask(text)

    # Затем, если cloakllm доступен, применяем его (может маскировать имена)
    cloak_func = _get_cloakllm()
    if cloak_func is not None:
        try:
            text = cloak_func(text)
        except Exception:
            logger.exception("PII: ошибка при вызове cloakllm.mask_pii, используется regex-результат")
    return text


def contains_pii(text: str) -> bool:
    return bool(text) and mask_pii(text) != text


def warmup() -> None:
    """Предзагрузка cloakllm на старте процесса."""
    _get_cloakllm()