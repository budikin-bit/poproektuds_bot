import asyncio
import logging
import time
import httpx
from anthropic import (
    AsyncAnthropic,
    APIConnectionError,   # сетевые сбои; APITimeoutError — его подкласс
    APIStatusError,       # ответ с HTTP-ошибкой (4xx/5xx)
)

from config import (
    OPENMODEL_API_KEY,
    OPENMODEL_BASE_URL,
    MODEL,
    SYSTEM_PROMPT,           # используется как значение по умолчанию
    MAX_TOKENS,
    TEMPERATURE,
    LLM_TIMEOUT,             # импортируем для таймаута
)

MAX_MESSAGE_LENGTH = 3800

# HTTP-коды, при которых повтор имеет смысл (перегрузка/временный сбой).
_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}

# Меньше этого остатка бюджета новую попытку/фallback не начинаем:
# генерация 5 блоков занимает минуты, шансов успеть нет.
_MIN_ATTEMPT_SECONDS = 45.0

# Клиент с большим таймаутом (5 минут на чтение, 30 сек на соединение).
# max_retries=0: ретраями управляем сами в call_llm, иначе внутренние
# повторы SDK (по умолчанию 2) умножаются на наши и дают до 9 запросов.
client = AsyncAnthropic(
    api_key=OPENMODEL_API_KEY,
    base_url=OPENMODEL_BASE_URL,
    timeout=httpx.Timeout(300.0, connect=30.0),
    max_retries=0,
)

logger = logging.getLogger(__name__)


def _is_retryable(exc: Exception) -> bool:
    """Стоит ли повторять запрос после этой ошибки."""
    if isinstance(exc, APIConnectionError):
        # Сетевые сбои и таймауты (APITimeoutError — подкласс). Сюда же SDK
        # заворачивает httpx.RemoteProtocolError при обрыве стрима.
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in _RETRYABLE_STATUS
    if isinstance(exc, httpx.TransportError):
        # Подстраховка: обрыв уже открытого стрима может прилететь голым
        # httpx-исключением из итератора, минуя обёртки SDK.
        return True
    return False


async def call_llm(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    retries: int = 3,
    deadline: float | None = None,
) -> str:
    """Запрос к модели через OpenModel с повторными попытками и fallback на не-стриминг.

    Аргументы:
        user_prompt: промпт пользователя
        system_prompt: системный промпт (по умолчанию из config)
        retries: число повторных попыток при временных ошибках
        deadline: абсолютный дедлайн time.monotonic(); None — без ограничения.
            Новая попытка или fallback не стартуют, если до дедлайна
            осталось меньше _MIN_ATTEMPT_SECONDS.
    """
    def _remaining() -> float | None:
        return None if deadline is None else deadline - time.monotonic()

    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            # Пытаемся использовать стриминг (быстрее, но может рваться)
            return await _stream_call(user_prompt, system_prompt, _remaining())
        except Exception as e:
            if not _is_retryable(e):
                # Невосстановимая ошибка (4xx-валидация, неверный ключ и т.п.)
                logger.exception("Неповторяемая ошибка в call_llm")
                raise
            logger.warning(
                "Стриминг, попытка %s/%s упала: %s: %s",
                attempt, retries, type(e).__name__, e,
            )
            last_exception = e

            pause = 2 ** attempt  # 2, 4, 8 сек
            rem = _remaining()
            can_retry = attempt < retries and (
                rem is None or rem > pause + _MIN_ATTEMPT_SECONDS
            )
            if can_retry:
                await asyncio.sleep(pause)
                continue

            # Ретраи кончились (или на них нет бюджета) — fallback без стрима,
            # если время ещё позволяет.
            rem = _remaining()
            if rem is not None and rem < _MIN_ATTEMPT_SECONDS:
                logger.warning(
                    "Бюджет времени исчерпан (осталось %.0fс) — "
                    "fallback не запускаем", rem,
                )
                raise last_exception
            logger.warning("Стриминг не удался, пробуем обычный запрос без стрима")
            try:
                return await _non_stream_call(user_prompt, system_prompt, rem)
            except Exception as e2:
                logger.exception("Обычный запрос тоже упал")
                raise e2 from e

    # Если сюда дошли – все попытки провалились
    raise last_exception


def _request_timeout(remaining: float | None, default_read: float) -> httpx.Timeout:
    """Таймаут запроса: не больше остатка бюджета."""
    read = default_read if remaining is None else max(5.0, min(default_read, remaining))
    return httpx.Timeout(read, connect=min(30.0, read))


async def _stream_call(
    user_prompt: str, system_prompt: str, remaining: float | None = None
) -> str:
    """Стриминговый вызов с переданным system_prompt."""
    chunks = []
    async with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=_request_timeout(remaining, 300.0),
    ) as stream:
        async for text in stream.text_stream:
            chunks.append(text)
    return "".join(chunks).strip()


async def _non_stream_call(
    user_prompt: str, system_prompt: str, remaining: float | None = None
) -> str:
    """Обычный (нестриминговый) вызов; таймаут — остаток бюджета."""
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=_request_timeout(remaining, 300.0),
    )
    # Берём только текстовые блоки: через шлюз первым может прийти
    # thinking-блок, и content[0].text уронил бы обработку.
    parts = [
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts).strip()


def split_message(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Разбивает текст на части строго меньше max_len."""
    if len(text) <= max_len:
        return [text]

    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break

        split_pos = text.rfind('\n', 0, max_len)
        if split_pos == -1:
            split_pos = text.rfind('. ', 0, max_len)
        if split_pos == -1:
            split_pos = text.rfind(' ', 0, max_len)
        if split_pos == -1 or split_pos < max_len // 2:
            split_pos = max_len

        part = text[:split_pos].strip()
        parts.append(part)
        text = text[split_pos:].strip()

    return parts


# ---------- НОВАЯ ФУНКЦИЯ, ожидаемая в bot.py ----------
async def call_llm_with_budget(user_prompt: str, system_prompt: str = SYSTEM_PROMPT, timeout: int = LLM_TIMEOUT) -> str:
    """Вызов LLM с общим бюджетом времени.

    Дедлайн передаётся внутрь call_llm, чтобы ретраи и fallback знали,
    сколько времени осталось, и не начинали заведомо обречённые попытки.
    wait_for остаётся жёсткой внешней страховкой (+5с на завершение).
    """
    deadline = time.monotonic() + timeout
    return await asyncio.wait_for(
        call_llm(user_prompt, system_prompt, deadline=deadline),
        timeout=timeout + 5,
    )