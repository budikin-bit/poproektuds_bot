"""Хранилище сессий с корректным TTL, потокобезопасностью и атомарными
изменениями состояния.
Исправлено относительно прошлой версии:
• TTL проверяется и при попадании в кэш (раньше — только при чтении из БД);
• повреждённый JSON удаляется (раньше «ядовитая запись» зацикливала ошибки);
• read-modify-write сериализуется через общий db.lock;
• mutate_session — атомарная операция load→mutate→persist;
• кэш ограничен по размеру (LRU), чтобы не течь по памяти.
"""
import json
import time
import logging
from collections import OrderedDict
from typing import Dict, Any, Callable
import db

logger = logging.getLogger(__name__)

SESSION_TTL_SECONDS = 24 * 60 * 60
EVENTS_TTL_SECONDS = 24 * 60 * 60  # хранение event_id для дедупликации
_MAX_CACHE = 5000  # потолок LRU-кэша в памяти

_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_lock = db.lock  # единый замок со всеми модулями БД
_last_purge = 0.0


def _new_state() -> Dict[str, Any]:
    return {"step": 0, "data": {}, "finished": False, "ts": time.time()}


def _key(user_id) -> str:
    return str(user_id)


def _is_expired(session: Dict[str, Any]) -> bool:
    return (time.time() - session.get("ts", 0)) > SESSION_TTL_SECONDS


def _cache_put(key, session):
    _cache[key] = session
    _cache.move_to_end(key)
    while len(_cache) > _MAX_CACHE:
        _cache.popitem(last=False)


def _persist_locked(key: str, session: Dict[str, Any]) -> None:
    session["ts"] = time.time()
    c = db.conn()
    if c is None:
        return
    c.execute(
        "INSERT OR REPLACE INTO sessions (key, data, ts) VALUES (?, ?, ?)",
        (key, json.dumps(session, ensure_ascii=False), session["ts"]),
    )
    c.commit()


def _delete_locked(key: str) -> None:
    _cache.pop(key, None)
    c = db.conn()
    if c is not None:
        c.execute("DELETE FROM sessions WHERE key = ?", (key,))
        c.commit()


def _load_locked(key: str):
    """Сессия из кэша/БД с учётом TTL или None. Повреждённые записи удаляются."""
    session = _cache.get(key)
    if session is not None:
        if _is_expired(session):
            _delete_locked(key)
            return None
        _cache.move_to_end(key)
        return session
    c = db.conn()
    if c is not None:
        row = c.execute("SELECT data, ts FROM sessions WHERE key = ?", (key,)).fetchone()
        if row:
            try:
                data = json.loads(row[0])
                if not isinstance(data, dict):
                    raise ValueError("session payload is not a dict")
            except Exception:
                logger.warning("Повреждённая сессия key=%s — удаляю", key)
                _delete_locked(key)
                return None
            if (time.time() - (row[1] or 0)) > SESSION_TTL_SECONDS:
                _delete_locked(key)
                return None
            _cache_put(key, data)
            return data
    return None


def _purge_locked() -> None:
    cutoff = time.time() - SESSION_TTL_SECONDS
    c = db.conn()
    if c is not None:
        c.execute("DELETE FROM sessions WHERE ts < ?", (cutoff,))
        # Заодно чистим таблицу дедупликации событий (bot._seen_event):
        # без этого она растёт бесконечно. Суток хранения более чем
        # достаточно — повторные доставки от MAX приходят в пределах минут.
        c.execute(
            "DELETE FROM processed_events WHERE ts < ?",
            (time.time() - EVENTS_TTL_SECONDS,),
        )
        c.commit()
    for k in [k for k, s in _cache.items() if s.get("ts", 0) < cutoff]:
        _cache.pop(k, None)


def _maybe_purge_locked() -> None:
    global _last_purge
    now = time.time()
    if now - _last_purge > 3600:
        _last_purge = now
        try:
            _purge_locked()
        except Exception:
            logger.exception("Ошибка очистки протухших сессий")


def get_session(user_id) -> Dict[str, Any]:
    """Вернуть сессию (создаёт новую при отсутствии/протухании).
    Возвращает живой объект из кэша — для обратной совместимости с кодом,
    меняющим его «на месте». Для конкурентных изменений используйте
    mutate_session (атомарно).
    """
    key = _key(user_id)
    with _lock:
        _maybe_purge_locked()
        session = _load_locked(key)
        if session is None:
            session = _new_state()
            _cache_put(key, session)
            _persist_locked(key, session)
        return session


def save_session(user_id) -> None:
    key = _key(user_id)
    with _lock:
        if key in _cache:
            _persist_locked(key, _cache[key])


def mutate_session(user_id, mutator: Callable[[Dict[str, Any]], Any]):
    """Атомарно: load → mutator(session) → persist. Возвращает результат mutator."""
    key = _key(user_id)
    with _lock:
        _maybe_purge_locked()
        session = _load_locked(key)
        if session is None:
            session = _new_state()
            _cache_put(key, session)
        result = mutator(session)
        _persist_locked(key, session)
        return result


def update_session(user_id, **kwargs) -> None:
    mutate_session(user_id, lambda s: s.update(kwargs))


def delete_session(user_id) -> None:
    with _lock:
        _delete_locked(_key(user_id))


def reset_session(user_id) -> Dict[str, Any]:
    key = _key(user_id)
    with _lock:
        session = _new_state()
        _cache_put(key, session)
        _persist_locked(key, session)
        return session


# Чистим протухшие сессии при старте.
with _lock:
    try:
        _purge_locked()
    except Exception:
        logger.exception("Ошибка стартовой очистки сессий")