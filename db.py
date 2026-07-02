"""Единое SQLite-соединение + процесс-уровневая сериализация записи.
Все модули состояния (sessions, daily_limits, consents, processed_events) пишут через
одно соединение и ОБЯЗАНЫ держать db.lock на время read-modify-write, иначе
параллельные транзакции дают 'database is locked' и интерливинг.
"""
import os
import sys
import logging
import sqlite3
import threading
from config import DB_PATH, ALLOW_MEMORY_STORAGE

logger = logging.getLogger(__name__)

_conn = None
lock = threading.RLock()


def _init_schema(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions "
        "(key TEXT PRIMARY KEY, data TEXT, ts REAL)"
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_ts ON sessions(ts)")
    # consents — задел под фиксацию согласий на обработку ПД (152-ФЗ);
    # кодом пока не используется, но схему держим готовой.
    conn.execute(
        "CREATE TABLE IF NOT EXISTS consents "
        "(chat_id TEXT PRIMARY KEY, version TEXT, ts REAL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS daily_limits "
        "(day TEXT PRIMARY KEY, count INTEGER NOT NULL DEFAULT 0)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS processed_events "
        "(event_id TEXT PRIMARY KEY, ts REAL)"
    )
    # Миграция (этап 3.2): колонка status ('processing' | 'done') для схемы
    # дедупликации «пометить после обработки». Записи без статуса (старые)
    # считаем 'done' — они уже были обработаны при старой схеме.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(processed_events)")}
    if "status" not in cols:
        conn.execute(
            "ALTER TABLE processed_events "
            "ADD COLUMN status TEXT NOT NULL DEFAULT 'done'"
        )
    conn.commit()


if DB_PATH:
    try:
        _dir = os.path.dirname(os.path.abspath(DB_PATH))
        os.makedirs(_dir, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute("PRAGMA busy_timeout=5000")
        _conn.execute("PRAGMA synchronous=NORMAL")
        _conn.execute("PRAGMA foreign_keys=ON")
        _init_schema(_conn)
        logger.info("SQLite-хранилище подключено: %s", DB_PATH)
    except Exception:
        logger.exception("Не удалось открыть SQLite (%s)", DB_PATH)
        if not ALLOW_MEMORY_STORAGE:
            logger.critical(
                "DB_PATH задан, но БД недоступна. Останавливаюсь "
                "(ALLOW_MEMORY_STORAGE=0)."
            )
            sys.exit(1)
        _conn = None
else:
    if not ALLOW_MEMORY_STORAGE:
        logger.critical(
            "DB_PATH не задан и ALLOW_MEMORY_STORAGE=0 — останавливаюсь, "
            "чтобы не терять данные. Для локальной отладки задайте "
            "ALLOW_MEMORY_STORAGE=1."
        )
        sys.exit(1)
    logger.warning(
        "DB_PATH не задан — состояние в памяти (теряется при перезапуске)."
    )


def conn():
    return _conn


def close():
    global _conn
    if _conn is not None:
        with lock:
            try:
                _conn.commit()
                _conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except Exception:
                logger.exception("Ошибка при закрытии БД")
            finally:
                _conn.close()
                _conn = None