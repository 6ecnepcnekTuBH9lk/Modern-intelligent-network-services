import asyncio
import time
import uuid

from fastapi import HTTPException

from .batching import BatchProcessor
from .config import ALLOWED_REGIONS, MODEL_VERSION, WEEKDAY_MAP, Settings
from .database import Database
from .llm import LLMClient, build_scoring_explanation_prompt
from .model import InferenceEngine
from .schemas import FarmRequest


def get_risk_level(score: float) -> str:
    if score < 0.3:
        return "low"
    if score < 0.7:
        return "medium"
    return "high"


def get_recommendation(level: str) -> str:
    if level == "low":
        return "Стандартное рассмотрение"
    if level == "medium":
        return "Требуется дополнительная проверка"
    return "Высокий риск. Требуется ручное рассмотрение"


def validate_business_rules(data: FarmRequest) -> None:
    if data.region not in ALLOWED_REGIONS:
        raise HTTPException(400, f"Unknown region: {data.region}. Allowed regions: {sorted(ALLOWED_REGIONS)}")


def make_prediction_result(data: FarmRequest, score: float, request_id: str) -> dict:
    level = get_risk_level(score)
    return {
        "request_id": request_id, "farm_id": data.farm_id, "risk_score": score,
        "risk_level": level, "recommendation": get_recommendation(level), "model_version": MODEL_VERSION,
    }


class PredictionService:
    def __init__(self, settings: Settings, db: Database, engine: InferenceEngine, batches: BatchProcessor):
        self.settings = settings
        self.db = db
        self.engine = engine
        self.batches = batches

    def validate(self, data: FarmRequest) -> None:
        if not self.settings.model_ready:
            raise HTTPException(503, "Model is unavailable")
        validate_business_rules(data)

    async def predict(self, data: FarmRequest, request_id: str, source: str = "single") -> dict:
        self.validate(data)
        try:
            score = await asyncio.wait_for(
                self.engine.model_predict_single(data), timeout=self.settings.inference_timeout_seconds,
            )
        except TimeoutError as exc:
            raise HTTPException(504, "Inference timeout") from exc
        result = make_prediction_result(data, score, request_id)
        await self.db.save_prediction(result, source)
        return result

    async def predict_batch(self, items: list[FarmRequest], request_id: str) -> dict:
        for data in items:
            self.validate(data)
        started = time.perf_counter()
        try:
            scores = await asyncio.wait_for(
                self.engine.model_predict_batch(items), timeout=self.settings.inference_timeout_seconds,
            )
        except TimeoutError as exc:
            raise HTTPException(504, "Batch inference timeout") from exc
        response_items = []
        for data, score in zip(items, scores, strict=True):
            item = make_prediction_result(data, score, request_id)
            item.pop("request_id")
            response_items.append(item)
        await self.db.save_predictions_batch(request_id, response_items)
        return {
            "request_id": request_id, "batch_size": len(items), "items": response_items,
            "processing_time_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    async def predict_queued(self, data: FarmRequest, request_id: str) -> dict:
        self.validate(data)
        score = await self.batches.submit(data, request_id)
        result = make_prediction_result(data, score, request_id)
        await self.db.save_prediction(result, "queued")
        return result

    async def predict_with_llm(self, data: FarmRequest, request_id: str, llm: LLMClient) -> dict:
        prediction = await self.predict(data, request_id, source="single+llm")
        explanation = await llm.chat(build_scoring_explanation_prompt(prediction), request_id)
        return {"prediction": prediction, "llm": explanation}

    async def list_predictions(self, limit: int, risk_level: str | None) -> list[dict]:
        if risk_level is not None and risk_level not in {"low", "medium", "high"}:
            raise HTTPException(400, "risk_level must be low, medium or high")
        return await self.db.get_predictions(limit, risk_level)

    async def request_predictions(self, request_id: str) -> list[dict]:
        results = await self.db.get_by_request_id(request_id)
        if not results:
            raise HTTPException(404, "Prediction not found")
        return results

    async def get_prediction(self, request_id: str) -> dict:
        results = await self.request_predictions(request_id)
        if len(results) > 1:
            raise HTTPException(
                409, "This request_id contains multiple predictions. "
                f"Use /requests/{request_id}/predictions",
            )
        return results[0]

    async def weekday_page(self, url, weekday: str, page: int, per_page: int) -> dict:
        weekday = weekday.lower()
        if weekday not in WEEKDAY_MAP:
            raise HTTPException(400, "weekday must be one of: monday, tuesday, wednesday, thursday, friday, saturday, sunday")
        data, total_items, total_pages = await self.db.get_weekday_page(WEEKDAY_MAP[weekday], page, per_page)
        if total_pages > 0 and page > total_pages:
            raise HTTPException(404, f"Page {page} does not exist. Total pages: {total_pages}")
        if total_pages == 0 and page > 1:
            raise HTTPException(404, "No data for this weekday")
        return {"data": data, "meta": {
            "weekday": weekday, "current_page": page, "per_page": per_page,
            "total_items": total_items, "total_pages": total_pages,
            "next_page": str(url.include_query_params(weekday=weekday, page=page + 1, per_page=per_page)) if page < total_pages else None,
            "prev_page": str(url.include_query_params(weekday=weekday, page=page - 1, per_page=per_page)) if page > 1 and total_pages else None,
        }}

    async def seed_weekdays(self, count: int) -> dict:
        rows = []
        for i in range(count):
            score = round(min(0.1 + (i % 9) * 0.1, 1.0), 2)
            level = get_risk_level(score)
            rows.append((str(uuid.uuid4()), f"DEMO-{i + 1:03d}", score, level,
                         get_recommendation(level), MODEL_VERSION, "seed", f"-{i % 21} days"))
        await self.db.seed_weekdays(rows)
        return {"status": "ok", "created": count,
                "message": "Demo predictions created. Use /predictions/by-weekday for pagination."}
