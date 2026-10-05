"""Бизнес-валидация, постпроцессинг и работа с прогнозами."""

import logging
import uuid

from fastapi import HTTPException, status

from model import MODEL_VERSION, calculate_risk
from schemas import FarmRequest, PredictionResponse
from storage import predictions

logger = logging.getLogger(__name__)
ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}


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


def create_prediction(request: FarmRequest) -> PredictionResponse:
    if request.region not in ALLOWED_REGIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown region: {request.region}. "
                f"Allowed regions: {sorted(ALLOWED_REGIONS)}"
            ),
        )

    logger.info("Prediction request received | farm_id=%s", request.farm_id)
    score = calculate_risk(request)
    level = get_risk_level(score)
    result = PredictionResponse(
        request_id=str(uuid.uuid4()),
        farm_id=request.farm_id,
        risk_score=score,
        risk_level=level,
        recommendation=get_recommendation(level),
        model_version=MODEL_VERSION,
    )
    predictions[result.request_id] = result
    logger.info(
        "Prediction completed | request_id=%s | farm_id=%s | model_version=%s | risk_score=%s | risk_level=%s",
        result.request_id, result.farm_id, MODEL_VERSION, score, level,
    )
    return result


def find_prediction(request_id: str) -> PredictionResponse:
    if request_id not in predictions:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Prediction not found")
    return predictions[request_id]


def list_predictions(limit: int = 10, risk_level: str | None = None) -> list[PredictionResponse]:
    values = list(predictions.values())
    if risk_level is not None:
        if risk_level not in ALLOWED_RISK_LEVELS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="risk_level must be 'low', 'medium' or 'high'",
            )
        values = [item for item in values if item.risk_level == risk_level]
    return values[:limit]
