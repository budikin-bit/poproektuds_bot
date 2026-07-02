"""Учёт согласий пользователей с условиями использования и политикой
конфиденциальности (Правила MAX п. 4.2, 4.4; Требования п. 2.1).

Использует таблицу consents (chat_id, version, ts), заложенную в db.py.
Согласие привязано к версии документов (config.CONSENT_VERSION):
при обновлении документов версия меняется — has_consent вернёт False,
и бот попросит принять условия заново.

Fallback без БД — в памяти процесса (теряется при рестарте: пользователей
попросят принять условия повторно, что безопасно).
"""
import time
import logging
import db
from config import CONSENT_VERSION

logger = logging.getLogger(__name__)

_mem: dict = {}  # chat_id -> version (fallback без БД)


def has_consent(chat_id) -> bool:
    """Принял ли пользователь ТЕКУЩУЮ версию документов."""
    key = str(chat_id)
    c = db.conn()
    with db.lock:
        if c is None:
            return _mem.get(key) == CONSENT_VERSION
        row = c.execute(
            "SELECT version FROM consents WHERE chat_id = ?", (key,)
        ).fetchone()
        return bool(row) and row[0] == CONSENT_VERSION


def record_consent(chat_id) -> None:
    """Зафиксировать принятие текущей версии документов (версия + время)."""
    key = str(chat_id)
    c = db.conn()
    with db.lock:
        if c is None:
            _mem[key] = CONSENT_VERSION
            logger.info("Согласие (память) chat_id=%s version=%s", key, CONSENT_VERSION)
            return
        try:
            c.execute(
                "INSERT OR REPLACE INTO consents(chat_id, version, ts) "
                "VALUES(?, ?, ?)",
                (key, CONSENT_VERSION, time.time()),
            )
            c.commit()
            logger.info("Согласие зафиксировано chat_id=%s version=%s", key, CONSENT_VERSION)
        except Exception:
            c.rollback()
            raise


def revoke_consent(chat_id) -> None:
    """Отозвать согласие (используется командой /delete)."""
    key = str(chat_id)
    c = db.conn()
    with db.lock:
        if c is None:
            _mem.pop(key, None)
            return
        try:
            c.execute("DELETE FROM consents WHERE chat_id = ?", (key,))
            c.commit()
            logger.info("Согласие отозвано chat_id=%s", key)
        except Exception:
            c.rollback()
            raise
