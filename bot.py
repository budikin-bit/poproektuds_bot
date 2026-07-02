import asyncio
import time
import logging
import logging.handlers

# pii не имеет project-зависимостей и ничего не логирует при импорте —
# его можно (и нужно) импортировать до настройки логирования: он
# используется фильтром ниже.
import pii


class _PiiLogFilter(logging.Filter):
    """Страховочный фильтр: regex-маскировка ПД (email, телефоны, паспорта,
    СНИЛС, карты, ИНН) во ВСЕХ записях лога, включая логи библиотек.

    Это защита от будущих регрессий, а не замена правилу «не логировать
    пользовательский текст»: NER здесь не запускается (слишком дорого на
    каждую строку), поэтому ФИО фильтр НЕ ловит.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = pii.mask_pii(record.msg)
            if record.args:
                record.args = tuple(
                    pii.mask_pii(a) if isinstance(a, str) else a
                    for a in record.args
                )
        except Exception:
            # Логирование не должно падать из-за фильтра.
            pass
        return True


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    fh = logging.handlers.RotatingFileHandler(
        "bot.log", maxBytes=10_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logging.getLogger().addHandler(fh)
    # fontTools при каждой генерации PDF пишет ~100 INFO-строк про глифы —
    # оставляем от него только предупреждения и ошибки.
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    # Фильтры логгера не наследуются обработчиками — вешаем на каждый handler.
    pii_filter = _PiiLogFilter()
    for handler in logging.getLogger().handlers:
        handler.addFilter(pii_filter)


# Настраиваем логирование ДО импорта остальных модулей проекта: db.py
# пишет важные строки при импорте («SQLite-хранилище подключено», ошибки
# открытия БД) — раньше они терялись, т.к. обработчики ещё не существовали.
_setup_logging()
logger = logging.getLogger(__name__)

from maxapi import Bot, Dispatcher  # noqa: E402
from maxapi.types import (  # noqa: E402
    MessageCreated, BotStarted, MessageCallback, CallbackButton, InputMediaBuffer,
)
from maxapi.enums.parse_mode import ParseMode  # noqa: E402
from maxapi.enums.upload_type import UploadType  # noqa: E402
from maxapi.utils.inline_keyboard import InlineKeyboardBuilder  # noqa: E402

import db  # noqa: E402
from config import (  # noqa: E402
    MAX_BOT_TOKEN, SYSTEM_PROMPT, ANALYSIS_BLOCKS, LLM_TIMEOUT, MASK_PII,
    ALLOWED_USER_IDS,
)
from session_manager import get_session, update_session, reset_session  # noqa: E402
from wizard import (  # noqa: E402
    get_question_text, get_keyboard_for_step, process_answer,
    get_collected_data, handle_callback, TOTAL_STEPS,
)
from daily_limit import try_consume_free, refund_free  # noqa: E402
from formatter import build_user_prompt, format_for_max  # noqa: E402
from llm_client import call_llm_with_budget, split_message  # noqa: E402

try:
    from pdf_generator import generate_pdf_from_text  # noqa: E402
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PDF-генерация отключена: модуль не найден")


bot = Bot(token=MAX_BOT_TOKEN)
dp = Dispatcher()

_running_analysis: set = set()
_chat_locks: dict = {}
_MAX_CHAT_LOCKS = 1000  # потолок, после которого выбрасываем свободные локи


def _chat_lock(chat_id) -> asyncio.Lock:
    lock = _chat_locks.get(chat_id)
    if lock is None:
        if len(_chat_locks) >= _MAX_CHAT_LOCKS:
            # Удаляем только незанятые локи. Безопасно, т.к. выполняется в
            # event loop без await, а `async with _chat_lock(x)` захватывает
            # лок сразу после получения, не уступая управление циклу, —
            # окна, в котором чужая ссылка указывает на удалённый лок, нет.
            for cid in [c for c, l in _chat_locks.items() if not l.locked()]:
                del _chat_locks[cid]
            logger.info(
                "Очистка _chat_locks: осталось %d занятых", len(_chat_locks)
            )
        lock = asyncio.Lock()
        _chat_locks[chat_id] = lock
    return lock


_PROCESSING_TTL = 600  # сек: старше — «processing» считается зависшим (процесс упал)


def _event_id(event):
    eid = (
        getattr(getattr(event, "message", None), "body", None)
        and getattr(event.message.body, "mid", None)
    )
    return str(eid) if eid else None


def _claim_event(eid) -> bool:
    """Атомарно занять событие. True — наше, обрабатываем; False — дубль
    (уже обработано или прямо сейчас обрабатывается другим воркером).

    Схема «пометить ПОСЛЕ обработки»: клейм ставит status='processing';
    _finish_event переводит в 'done'; _release_event снимает клейм при
    сбое, чтобы повторная доставка от MAX была обработана, а не потеряна.
    Зависший 'processing' старше _PROCESSING_TTL можно перезанять.
    """
    if not eid:
        logger.warning("Не удалось получить event_id, дубли не будут блокироваться")
        return True
    c = db.conn()
    if c is None:
        logger.warning("БД недоступна, дубли не блокируются")
        return True
    now = time.time()
    try:
        with db.lock:
            cur = c.execute(
                "INSERT INTO processed_events(event_id, ts, status) "
                "VALUES(?, ?, 'processing') "
                "ON CONFLICT(event_id) DO UPDATE SET ts = excluded.ts, "
                "status = 'processing' "
                "WHERE processed_events.status = 'processing' "
                "AND processed_events.ts < ?",
                (eid, now, now - _PROCESSING_TTL),
            )
            c.commit()
            claimed = cur.rowcount > 0
            if not claimed:
                logger.info("Дубль события %s — пропускаем", eid)
            return claimed
    except Exception:
        # Учёт дублей не должен ронять обработчик: в худшем случае
        # событие обработается повторно, это лучше потери сообщения.
        logger.exception("Ошибка клейма события %s — обрабатываем без дедупликации", eid)
        return True


def _finish_event(eid) -> None:
    """Пометить событие успешно обработанным."""
    if not eid:
        return
    c = db.conn()
    if c is None:
        return
    try:
        with db.lock:
            c.execute(
                "UPDATE processed_events SET status = 'done', ts = ? "
                "WHERE event_id = ?",
                (time.time(), eid),
            )
            c.commit()
    except Exception:
        logger.exception("Ошибка фиксации события %s", eid)


def _release_event(eid) -> None:
    """Снять клейм после сбоя: повторная доставка события будет обработана."""
    if not eid:
        return
    c = db.conn()
    if c is None:
        return
    try:
        with db.lock:
            c.execute(
                "DELETE FROM processed_events "
                "WHERE event_id = ? AND status = 'processing'",
                (eid,),
            )
            c.commit()
            logger.info("Клейм события %s снят — повторная доставка будет обработана", eid)
    except Exception:
        logger.exception("Ошибка снятия клейма события %s", eid)


def _restart_keyboard():
    kb = InlineKeyboardBuilder()
    kb.row(CallbackButton(text="🔄 Начать заново", payload="restart"))
    return kb.as_markup()


def _extract_user_id(event) -> str | None:
    """Достаёт user_id из разных типов событий maxapi."""
    for attr_path in ("user_id", "from_user.user_id", "user.user_id", "sender.user_id"):
        obj = event
        try:
            for part in attr_path.split("."):
                obj = getattr(obj, part)
            if obj is not None:
                return str(obj)
        except AttributeError:
            continue
    return None


async def _check_access(event, chat_id, user_id) -> bool:
    """True — доступ разрешён. False — отказано (с отправкой сообщения)."""
    if not ALLOWED_USER_IDS:
        logger.warning(
            "ALLOWED_USER_IDS пуст — доступ закрыт для всех. "
            "Задайте переменную окружения, чтобы разрешить себе пользоваться ботом."
        )
        return False
    if user_id is None:
        logger.warning("Не удалось определить user_id для chat_id=%s — доступ закрыт", chat_id)
        try:
            await bot.send_message(chat_id=chat_id, text="⛔ Доступ ограничен.")
        except Exception:
            pass
        return False
    # Разрешаем доступ, если chat_id ИЛИ user_id есть в списке.
    # В MAX для личных чатов chat_id == идентификатор пользователя,
    # поэтому достаточно указать любое из двух значений в ALLOWED_USER_IDS.
    allowed = ALLOWED_USER_IDS
    if str(chat_id) not in allowed and user_id not in allowed:
        logger.info("Доступ запрещён для user_id=%s (chat_id=%s)", user_id, chat_id)
        try:
            await bot.send_message(
                chat_id=chat_id,
                text="⛔ Этот бот доступен только ограниченному кругу пользователей.",
            )
        except Exception:
            pass
        return False
    return True


async def send_text_parts(chat_id, full_response):
    md_text = format_for_max(full_response, markdown=True)
    parts = split_message(md_text)
    total = len(parts)
    for i, part in enumerate(parts):
        prefix = f"📄 Часть {i + 1}/{total}\n\n" if total > 1 else ""
        try:
            await bot.send_message(
                chat_id=chat_id, text=prefix + part, format=ParseMode.MARKDOWN
            )
        except Exception as md_error:
            logger.warning(
                "Markdown части %s/%s не прошёл (%s) — шлю чистым текстом",
                i + 1, total, md_error,
            )
            plain = prefix + format_for_max(part, markdown=False)
            await bot.send_message(chat_id=chat_id, text=plain)


async def _start_survey(chat_id):
    logger.info("Запуск опроса для chat_id=%s", chat_id)
    reset_session(chat_id)
    await send_step(chat_id, get_question_text(0), get_keyboard_for_step(0))


async def send_step(chat_id, text, keyboard):
    logger.info(
        "Отправка шага для %s: текст=%r",
        chat_id, text[:50] + "..." if len(text) > 50 else text,
    )
    if keyboard is not None:
        await bot.send_message(chat_id=chat_id, text=text, attachments=[keyboard])
    else:
        await bot.send_message(chat_id=chat_id, text=text)


async def run_analysis(chat_id):
    logger.info("run_analysis начат для %s", chat_id)
    if chat_id in _running_analysis:
        logger.info("Анализ для %s уже идёт — уведомление", chat_id)
        await bot.send_message(
            chat_id=chat_id,
            text="⏳ Ваш анализ уже выполняется — дождитесь результата."
        )
        return
    if not try_consume_free():
        logger.info("Дневной лимит исчерпан для %s", chat_id)
        update_session(chat_id, finished=True)
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "😔 На сегодня бесплатные разборы закончились.\n\n"
                "Загляните завтра — пожалуйста, попробуйте снова."
            ),
            attachments=[_restart_keyboard()],
        )
        return

    _running_analysis.add(chat_id)
    refund_state = {"consumed": True, "refunded": False}

    async def _fail(user_text):
        if refund_state["consumed"] and not refund_state["refunded"]:
            refund_free()
            refund_state["refunded"] = True
        update_session(chat_id, finished=True)
        await bot.send_message(
            chat_id=chat_id, text=user_text, attachments=[_restart_keyboard()]
        )

    try:
        data = get_collected_data(chat_id)
        project_name = data.get("project_name", "Экспертный анализ")

        await bot.send_message(
            chat_id=chat_id,
            text=(
                f"⏳ Запущен экспресс-анализ ({ANALYSIS_BLOCKS} блоков).\n"
                "Время обработки — до 5 минут. Не отправляйте новые сообщения — "
                "PDF появится автоматически."
            ),
        )

        user_prompt = build_user_prompt(data)

        try:
            full_response = await call_llm_with_budget(
                user_prompt, system_prompt=SYSTEM_PROMPT
            )
        except asyncio.TimeoutError:
            logger.error("Таймаут генерации (%sс) для %s", LLM_TIMEOUT, chat_id)
            await _fail(
                "⏳ Модель слишком долго не отвечает. Бесплатный разбор "
                "возвращён — попробуйте чуть позже."
            )
            return
        except Exception:
            logger.exception("Ошибка обращения к модели для %s", chat_id)
            await _fail(
                "⚠️ Не удалось обратиться к модели. Бесплатный разбор "
                "возвращён — попробуйте ещё раз."
            )
            return

        if not full_response or not full_response.strip():
            logger.error("Пустой ответ модели для %s", chat_id)
            await _fail(
                "⚠️ Модель вернула пустой ответ (сбой провайдера). "
                "Бесплатный разбор возвращён — попробуйте ещё раз."
            )
            return

        delivered = False
        if PDF_AVAILABLE:
            try:
                pdf_bytes = await asyncio.to_thread(
                    generate_pdf_from_text, full_response,
                    project_name=project_name, free_blocks=ANALYSIS_BLOCKS,
                )
                media = InputMediaBuffer(
                    buffer=pdf_bytes,
                    filename="expert_analysis.pdf",
                    type=UploadType.FILE,
                )
                await bot.send_message(
                    chat_id=chat_id,
                    text="📄 Ваш экспресс-анализ в PDF.",
                    attachments=[media],
                )
                delivered = True
                logger.info("PDF успешно отправлен для %s", chat_id)
            except Exception:
                logger.exception("PDF не отправлен — откат на текст")

        if not delivered:
            try:
                await send_text_parts(chat_id, full_response)
                delivered = True
                logger.info("Текстовая часть отправлена для %s", chat_id)
            except Exception:
                logger.exception("Текстовая доставка не удалась для %s", chat_id)

        if not delivered:
            await _fail(
                "⚠️ Анализ готов, но отправить результат не удалось "
                "(техсбой). Бесплатный разбор возвращён."
            )
            return

        await bot.send_message(
            chat_id=chat_id,
            text="✅ Экспресс-анализ завершён.",
        )
        # update_session(finished=True) здесь НЕ нужен: wizard уже выставил
        # finished при завершении опроса. Повторная запись после долгого
        # анализа могла бы затереть состояние сессии, начатой заново.

        # Дисклеймер о том, что это не готовый текст заявки
        await bot.send_message(
            chat_id=chat_id,
            text=(
                "⚠️ **Дисклеймер**: Это разбор-помощник, а не готовый текст заявки — не вставляйте дословно. "
                "Эксперты конкурсов отклоняют заявки, целиком написанные ИИ: добавьте свои факты, цифры и живой голос."
            ),
            format=ParseMode.MARKDOWN,
        )

        await bot.send_message(
            chat_id=chat_id,
            text="Чтобы запустить новый опрос, нажмите кнопку ниже.",
            attachments=[_restart_keyboard()],
        )

    finally:
        _running_analysis.discard(chat_id)
        logger.info("run_analysis завершён для %s", chat_id)


@dp.bot_started()
async def on_bot_started(event: BotStarted):
    user_id = _extract_user_id(event)
    if not await _check_access(event, event.chat_id, user_id):
        return
    reset_session(event.chat_id)
    await bot.send_message(
        chat_id=event.chat_id,
        text=(
            f"👋 Добро пожаловать в AI-эксперта по социальным проектам!\n\n"
            f"Я задам вам {TOTAL_STEPS} коротких вопросов.\n"
            "⚠️ Пожалуйста, не указывайте персональные данные (ФИО, телефоны).\n\n"
            "Напишите /start."
        ),
    )


@dp.message_created()
async def handle_text(event: MessageCreated):
    chat_id = None
    eid = None
    claimed = False
    try:
        chat_id, user_id = event.get_ids()
        body = event.message.body
        text = body.text if body else None
        # ВАЖНО (152-ФЗ): текст пользователя НЕ логируем — он может содержать
        # ПД до маскировки. Пишем только факт получения и длину.
        logger.info(
            "Получено сообщение от %s: %d символов",
            chat_id, len(text) if text else 0,
        )

        if not await _check_access(event, chat_id, str(user_id) if user_id is not None else None):
            return

        if not text:
            logger.info("Пустое сообщение, игнорируем")
            return

        eid = _event_id(event)
        if not _claim_event(eid):
            return
        claimed = True

        if text.strip().lower() == "/start":
            logger.info("Команда /start от %s", chat_id)
            # Пока идёт анализ, рестарт запрещён: reset_session создал бы
            # новую сессию, которую завершение старого анализа могло бы
            # пометить finished посреди опроса.
            if chat_id in _running_analysis:
                await bot.send_message(
                    chat_id=chat_id,
                    text="⏳ Сейчас выполняется анализ — дождитесь результата, "
                         "после этого можно будет начать заново.",
                )
                return
            async with _chat_lock(chat_id):
                await _start_survey(chat_id)
            return

        async with _chat_lock(chat_id):
            session = get_session(chat_id)
            if session.get("finished", False):
                await event.message.answer(
                    "✅ Опрос завершён. Нажмите «Начать заново» под последним сообщением."
                )
                logger.info("Сессия завершена, отправлено уведомление")
                return

            # Тяжёлую обработку (NER-маскировка ПД + запись в sqlite) уводим в
            # поток, чтобы не блокировать event loop.
            next_text, next_keyboard = await asyncio.to_thread(
                process_answer, chat_id, text
            )
            logger.info(
                "process_answer вернул: next_text=%r, next_keyboard=%r",
                next_text[:50] + "..." if next_text and len(next_text) > 50 else next_text,
                next_keyboard,
            )

        if next_text is None:
            logger.info("Опрос завершён, запуск анализа для %s", chat_id)
            await run_analysis(chat_id)
        else:
            await send_step(chat_id, next_text, next_keyboard)

        logger.info("Обработка сообщения %s завершена", chat_id)

    except Exception:
        logger.exception("КРИТИЧЕСКАЯ ОШИБКА в handle_text для chat_id=%s", chat_id)
        if claimed:
            # Снимаем клейм ДО уведомления: повторная доставка события
            # от MAX будет обработана заново, сообщение не потеряется.
            _release_event(eid)
            claimed = False
        try:
            if chat_id:
                await bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "⚠️ Произошла внутренняя ошибка. Пожалуйста, "
                        "попробуйте позже или нажмите «Начать заново»."
                    ),
                    attachments=[_restart_keyboard()],
                )
        except Exception:
            pass
    finally:
        # Сюда попадают все успешные пути, включая ранние return
        # (/start, «опрос завершён» и т.п.). При исключении claimed уже
        # сброшен в except — двойной записи не будет.
        if claimed:
            _finish_event(eid)


@dp.message_callback()
async def on_callback(event: MessageCallback):
    try:
        chat_id, user_id = event.get_ids()
        payload = event.callback.payload
        # Payload может быть произвольным (handle_callback трактует его как
        # ответ пользователя) — логируем только длину, не содержимое.
        logger.info("Callback from %s: %d символов", chat_id, len(payload or ""))

        if not await _check_access(event, chat_id, str(user_id) if user_id is not None else None):
            return

        if payload == "restart":
            # Пока идёт анализ, рестарт запрещён (см. комментарий в handle_text).
            if chat_id in _running_analysis:
                await event.answer(
                    new_text="⏳ Анализ ещё выполняется — дождитесь результата.",
                    attachments=[],
                    raise_if_not_exists=False,
                )
                return
            await event.answer(
                new_text="🔄 Начинаем новый опрос!", attachments=[],
                raise_if_not_exists=False,
            )
            async with _chat_lock(chat_id):
                await _start_survey(chat_id)
            return

        async with _chat_lock(chat_id):
            session = get_session(chat_id)
            if session.get("finished", False):
                await event.answer(
                    new_text="✅ Опрос уже завершён.", attachments=[],
                    raise_if_not_exists=False,
                )
                return
            # См. handle_text: тяжёлую обработку держим вне event loop.
            next_text, next_keyboard = await asyncio.to_thread(
                handle_callback, chat_id, payload
            )

        await event.answer(
            new_text="✅ Ответ принят", attachments=[],
            raise_if_not_exists=False,
        )
        if next_text is None:
            await run_analysis(chat_id)
        else:
            await send_step(chat_id, next_text, next_keyboard)

    except Exception:
        logger.exception("КРИТИЧЕСКАЯ ОШИБКА в on_callback для chat_id=%s", chat_id)
        try:
            await event.answer(
                new_text="⚠️ Ошибка обработки", attachments=[],
                raise_if_not_exists=False,
            )
        except Exception:
            pass


async def main():
    # Прогрев NER-модели до приёма трафика: иначе первый ответ с полем
    # из PII_FULL_FIELDS словит паузу в несколько секунд на загрузку spacy.
    if MASK_PII:
        logger.info("Прогрев PII-маскировки…")
        await asyncio.to_thread(pii.warmup)
        logger.info("Прогрев PII завершён")
    try:
        await dp.start_polling(bot)
    finally:
        try:
            await bot.session.close()
        except Exception:
            pass
        db.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен")