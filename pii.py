"""Маскировка персональных данных (152-ФЗ) перед записью в БД и отправкой в LLM.

Два режима:
  mask_pii(text)          — только regex (email, телефон, СНИЛС, ИНН, паспорт, карта).
                            Используется для полей с географией/названиями, чтобы NER
                            не подменял топонимы и аббревиатуры токенами.
  mask_pii_full(text)     — regex + cloakllm NER (PERSON, ORG, GPE).
                            Используется только для полей, где ожидаются реальные ФИО.

В wizard.py поля разделены на два типа (см. PII_FULL_FIELDS).
"""
import re
import logging
import threading

logger = logging.getLogger(__name__)

# ── Поля, для которых включается NER (могут содержать реальные ФИО) ──
PII_FULL_FIELDS = {"audience", "age_features", "existing_data"}

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
_shield_init_lock = threading.Lock()


def _get_shield():
    """Потокобезопасно: параллельные to_thread из bot.py могли одновременно
    пройти проверку _shield_tried и дважды инициализировать Shield
    (двойная загрузка spacy-модели). Теперь — double-checked locking.
    """
    global _shield, _shield_tried
    if _shield_tried:
        return _shield
    with _shield_init_lock:
        if _shield_tried:  # другой поток успел первым
            return _shield
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
        finally:
            # Флаг выставляем последним: пока он False, другие потоки ждут
            # на замке и не увидят наполовину созданный Shield.
            _shield_tried = True
    return _shield


def _regex_mask(text: str) -> str:
    for rx, repl in _RULES:
        text = rx.sub(repl, text)
    return text


def mask_pii(text: str) -> str:
    """Только regex — для полей с географией и названиями.
    Не трогает топонимы, аббревиатуры, названия организаций.
    """
    if not text:
        return text
    return _regex_mask(text)


def mask_pii_full(text: str) -> str:
    """Regex + NER — для полей, где ожидаются реальные ФИО (audience, existing_data и т.п.)."""
    if not text:
        return text
    text = _regex_mask(text)
    shield = _get_shield()
    if shield is not None:
        try:
            cloaked, _ = shield.sanitize(text)
            text = cloaked
        except Exception:
            logger.exception("PII: ошибка Shield.sanitize, используется regex-результат")
    return text


def mask_pii_for_field(text: str, field_id: str) -> str:
    """Выбирает режим маскировки в зависимости от поля wizard."""
    if field_id in PII_FULL_FIELDS:
        return mask_pii_full(text)
    return mask_pii(text)


def contains_pii(text: str) -> bool:
    return bool(text) and mask_pii_full(text) != text


def warmup() -> None:
    """Предзагрузка на старте процесса: конструируем Shield И выполняем
    холостой sanitize. Конструктор Shield не загружает NER-пайплайн —
    spacy-модель подтягивается лениво при первом вызове sanitize(),
    и без холостого вызова первый пользователь ждал ~5 секунд
    (подтверждено логами) даже при инициализированном Shield.
    """
    shield = _get_shield()
    if shield is not None:
        try:
            cloaked, _ = shield.sanitize(
                "Прогрев пайплайна: Иван Иванович, Москва."
            )
            if "Иван" in cloaked:
                logger.warning(
                    "PII: САМОПРОВЕРКА ПРОВАЛЕНА — тестовое ФИО не "
                    "замаскировано NER-ом. Вероятно, не установлена русская "
                    "модель (ru_core_news_sm) и cloakllm работает на "
                    "английской. Маскировка русских ФИО фактически НЕ "
                    "работает, полагаться можно только на regex-правила."
                )
            else:
                logger.info(
                    "PII: NER-пайплайн прогрет, тестовое ФИО замаскировано — "
                    "самопроверка OK."
                )
        except Exception:
            logger.exception("PII: ошибка холостого sanitize при прогреве")
