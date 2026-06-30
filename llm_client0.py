import asyncio
import random
import logging
import httpx
from typing import Optional, Dict, Any
from config import (
    LLM_API_STYLE,
    LLM_API_KEY,
    LLM_BASE_URL,
    MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    LLM_REQUEST_TIMEOUT,
    LLM_TIMEOUT,
    LLM_RETRIES,
    SYSTEM_PROMPT,
    MASK_PII,
)
from pii import mask_pii

logger = logging.getLogger(__name__)

def _build_headers(style: str, api_key: str) -> Dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if style == "openai":
        headers["Authorization"] = f"Bearer {api_key}"
    elif style == "anthropic":
        headers["Authorization"] = f"Bearer {api_key}"
    elif style == "yandex":
        headers["Authorization"] = f"Api-Key {api_key}"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers

def _build_payload(style: str, model: str, system: str, user: str, max_tokens: int, temperature: float) -> Dict[str, Any]:
    if style == "openai":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
    elif style == "yandex":
        # Yandex использует messages с полем "text" и modelUri
        return {
            "modelUri": model,          # например, "gpt://<folder-id>/yandexgpt/latest"
            "messages": [
                {"role": "system", "text": system},
                {"role": "user", "text": user}
            ],
            "temperature": temperature,
            "maxTokens": max_tokens,
        }
    elif style == "anthropic":
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
    else:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

def _parse_response(style: str, data: Dict[str, Any]) -> str:
    if style == "yandex":
        try:
            return data["result"]["alternatives"][0]["message"]["text"]
        except (KeyError, IndexError):
            raise ValueError("Неожиданный формат ответа Yandex")
    else:
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError):
            raise ValueError("Неожиданный формат ответа")

def _is_retryable(status: int, text: str) -> bool:
    if status == 429 or (500 <= status < 600):
        return True
    if 400 <= status < 500:
        return False
    return False

async def call_llm(
    user_prompt: str,
    system_prompt: str = SYSTEM_PROMPT,
    retries: int = LLM_RETRIES,
) -> str:
    if not LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY не задан (проверьте .env)")

    if MASK_PII:
        user_prompt = mask_pii(user_prompt)

    # Формируем URL
    if LLM_API_STYLE == "yandex":
        # Правильный эндпоинт для Yandex Foundation Models
        url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    else:
        url = f"{LLM_BASE_URL.rstrip('/')}"

    headers = _build_headers(LLM_API_STYLE, LLM_API_KEY)
    payload = _build_payload(
        LLM_API_STYLE,
        MODEL,
        system_prompt,
        user_prompt,
        MAX_TOKENS,
        TEMPERATURE
    )

    last_exc = None
    async with httpx.AsyncClient(timeout=LLM_REQUEST_TIMEOUT) as client:
        for attempt in range(1, retries + 1):
            try:
                logger.info("LLM (%s): попытка %s/%s", LLM_API_STYLE, attempt, retries)
                resp = await client.post(url, json=payload, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    content = _parse_response(LLM_API_STYLE, data)
                    if not content.strip():
                        raise ValueError("Пустой ответ модели")
                    logger.info("Ответ получен, длина: %s символов", len(content))
                    return content
                else:
                    error_text = resp.text
                    retryable = _is_retryable(resp.status_code, error_text)
                    logger.warning(
                        "Попытка %s упала (retryable=%s): HTTP %s: %s",
                        attempt, retryable, resp.status_code, error_text[:200]
                    )
                    if not retryable or attempt >= retries:
                        raise Exception(f"HTTP {resp.status_code}: {error_text}")
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
                last_exc = e
                retryable = True
                logger.warning("Попытка %s: сетевая ошибка (%s)", attempt, e)
                if attempt >= retries:
                    raise
            except Exception as e:
                last_exc = e
                retryable = True if not isinstance(e, httpx.HTTPStatusError) else False
                if not retryable or attempt >= retries:
                    raise
                logger.warning("Попытка %s: исключение (%s), повторяем", attempt, e)

            wait = min(2 ** attempt, 30) + random.uniform(0, 1)
            await asyncio.sleep(wait)

    raise last_exc or RuntimeError("Неизвестная ошибка при вызове LLM")

async def call_llm_with_budget(
    user_prompt: str, system_prompt: str = SYSTEM_PROMPT
) -> str:
    return await asyncio.wait_for(
        call_llm(user_prompt, system_prompt),
        timeout=LLM_TIMEOUT
    )

MAX_MESSAGE_LENGTH = 3800

def split_message(text: str, max_len: int = MAX_MESSAGE_LENGTH) -> list[str]:
    if not text:
        return []
    if len(text) <= max_len:
        return [text]
    parts = []
    while text:
        if len(text) <= max_len:
            parts.append(text)
            break
        split_pos = text.rfind("\n", 0, max_len)
        if split_pos <= 0:
            split_pos = text.rfind(". ", 0, max_len)
        if split_pos <= 0:
            split_pos = text.rfind(" ", 0, max_len)
        if split_pos <= 0 or split_pos < max_len // 2:
            split_pos = max_len
        part = text[:split_pos].strip()
        if part:
            parts.append(part)
        text = text[split_pos:].strip()
    return parts