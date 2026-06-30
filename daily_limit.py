"""Глобальный дневной лимит бесплатных разборов.
Исправлено:
• атомарный инкремент через SQL (UPDATE ... WHERE count < limit), а не
read-modify-write в Python — корректно при нескольких воркерах;
• граница суток по фиксированной таймзоне (config.TIMEZONE), не по TZ сервера;
• отдельная таблица daily_limits вместо JSON в kv.
"""
import datetime
import logging
from config import DAILY_LIMIT, TIMEZONE
import db

logger = logging.getLogger(__name__)

_mem = {"day": None, "count": 0}  # fallback без БД


def _today() -> str:
    return datetime.datetime.now(TIMEZONE).date().isoformat()


def free_uses_left():
    """Сколько бесплатных разборов осталось сегодня. None — без лимита."""
    if DAILY_LIMIT <= 0:
        return None
    day = _today()
    c = db.conn()
    with db.lock:
        if c is None:
            if _mem["day"] != day:
                _mem.update(day=day, count=0)
            return max(0, DAILY_LIMIT - _mem["count"])
        row = c.execute(
            "SELECT count FROM daily_limits WHERE day = ?", (day,)
        ).fetchone()
        return max(0, DAILY_LIMIT - (row[0] if row else 0))


def try_consume_free() -> bool:
    """Атомарно занять один слот. True — можно, False — лимит исчерпан."""
    if DAILY_LIMIT <= 0:
        return True
    day = _today()
    c = db.conn()
    with db.lock:
        if c is None:
            if _mem["day"] != day:
                _mem.update(day=day, count=0)
            if _mem["count"] >= DAILY_LIMIT:
                return False
            _mem["count"] += 1
            return True
        try:
            c.execute(
                "INSERT INTO daily_limits(day, count) VALUES(?, 0) "
                "ON CONFLICT(day) DO NOTHING",
                (day,),
            )
            cur = c.execute(
                "UPDATE daily_limits SET count = count + 1 "
                "WHERE day = ? AND count < ?",
                (day, DAILY_LIMIT),
            )
            c.commit()
            ok = cur.rowcount > 0
            if ok:
                logger.info("Бесплатный разбор занят за %s", day)
            return ok
        except Exception:
            c.rollback()
            raise


def refund_free() -> None:
    """Вернуть занятый слот (если разбор не состоялся). Не уходит ниже 0."""
    if DAILY_LIMIT <= 0:
        return
    day = _today()
    c = db.conn()
    with db.lock:
        if c is None:
            if _mem["day"] == day and _mem["count"] > 0:
                _mem["count"] -= 1
            return
        try:
            c.execute(
                "UPDATE daily_limits SET count = count - 1 "
                "WHERE day = ? AND count > 0",
                (day,),
            )
            c.commit()
            logger.info("Возврат слота за %s", day)
        except Exception:
            c.rollback()
            raise