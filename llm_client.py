import asyncio
import logging
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


async def call_llm(user_prompt: str, system_prompt: str = SYSTEM_PROMPT, retries: int = 3) -> str:
    """Запрос к модели через OpenModel с повторными попытками и fallback на не-стриминг.

    Аргументы:
        user_prompt: промпт пользователя
        system_prompt: системный промпт (по умолчанию из config)
        retries: число повторных попыток при временных ошибках
    """
    last_exception = None

    for attempt in range(1, retries + 1):
        try:
            # Пытаемся использовать стриминг (быстрее, но может рваться)
            return await _stream_call(user_prompt, system_prompt)
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
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)  # 2, 4, 8 сек
            else:
                logger.warning("Стриминг не удался, пробуем обычный запрос без стрима")
                try:
                    return await _non_stream_call(user_prompt, system_prompt)
                except Exception as e2:
                    logger.exception("Обычный запрос тоже упал")
                    raise e2 from e

    # Если сюда дошли – все попытки стриминга провалились, а fallback тоже упал
    raise last_exception


async def _stream_call(user_prompt: str, system_prompt: str) -> str:
    """Стриминговый вызов с переданным system_prompt."""
    chunks = []
    async with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    ) as stream:
        async for text in stream.text_stream:
            chunks.append(text)
    return "".join(chunks).strip()


async def _non_stream_call(user_prompt: str, system_prompt: str) -> str:
    """Обычный (нестриминговый) вызов с большим таймаутом."""
    response = await client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        timeout=httpx.Timeout(600.0, connect=30.0),  # 10 минут на чтение
    )
    return response.content[0].text.strip()


def split_message(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    """Разбивает текст на части строго меньше max_len."""
    if len(text) < max_len:
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
    """Вызов LLM с ограничением по времени (таймаут).
    
    Аргументы:
        user_prompt: промпт пользователя
        system_prompt: системный промпт
        timeout: максимальное время ожидания в секундах
    Возвращает:
        ответ модели
    Исключения:
        asyncio.TimeoutError, если время истекло
        любые другие ошибки от call_llm
    """
    return await asyncio.wait_for(
        call_llm(user_prompt, system_prompt),
        timeout=timeout
    )