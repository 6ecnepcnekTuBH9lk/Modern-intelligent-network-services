import logging
import time

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from .config import LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_TEMPERATURE, LLM_MODEL, Settings
from .schemas import LLMChatResponse

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings: Settings, client: httpx.AsyncClient):
        self.settings = settings
        self.client = client

    async def chat(
        self, prompt: str, request_id: str, model: str = LLM_MODEL,
        temperature: float = LLM_DEFAULT_TEMPERATURE, max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
    ) -> LLMChatResponse:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": (
                    "Ты помощник сервиса агроскоринга. Отвечай кратко и понятно. "
                    "Не изменяй переданный risk_score и risk_level. "
                    "Не придумывай причины, которых нет во входных данных."
                )},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        started = time.perf_counter()
        try:
            response = await self.client.post(
                self.settings.llm_base_url.rstrip("/") + "/chat/completions",
                headers={"Content-Type": "application/json"}, json=payload,
            )
        except httpx.TimeoutException as exc:
            logger.warning("LLM timeout | request_id=%s", request_id)
            raise HTTPException(504, "LLM request timeout") from exc
        except httpx.RequestError as exc:
            logger.warning("LLM connection error | request_id=%s | error=%s", request_id, exc)
            raise HTTPException(503, "LLM service is unavailable") from exc
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if response.status_code != 200:
            logger.error("LLM upstream error | request_id=%s | status=%s", request_id, response.status_code)
            raise HTTPException(502, f"LLM provider returned an error (HTTP {response.status_code})")
        try:
            data = response.json()
        except ValueError as exc:
            raise HTTPException(502, "LLM provider returned invalid JSON") from exc
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or not data["choices"]:
            raise HTTPException(502, "LLM response does not contain choices")
        choice = data["choices"][0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise HTTPException(502, "LLM response does not contain message content")
        try:
            result = LLMChatResponse(
                request_id=request_id, requested_model=model, returned_model=data.get("model"),
                answer=content, reasoning=message.get("reasoning_content"),
                finish_reason=choice.get("finish_reason"), usage=data.get("usage"), latency_ms=latency_ms,
            )
        except ValidationError as exc:
            raise HTTPException(502, "LLM provider returned invalid response fields") from exc
        logger.info("LLM completed | request_id=%s | latency_ms=%s", request_id, latency_ms)
        return result


def build_scoring_explanation_prompt(prediction: dict) -> str:
    return (
        "Сформируй краткое объяснение для оператора банка "
        "по результату алгоритмического агроскоринга.\n\n"
        f"risk_score: {prediction['risk_score']}\n"
        f"risk_level: {prediction['risk_level']}\n"
        f"базовая рекомендация: {prediction['recommendation']}\n"
        f"версия скоринговой модели: {prediction['model_version']}\n\n"
        "Требования к ответу:\n1. 2-4 предложения.\n2. Не меняй категорию риска.\n"
        "3. Не придумывай дополнительные факторы риска.\n"
        "4. Укажи, что решение требует применения правил банка "
        "и при необходимости проверки оператором."
    )
