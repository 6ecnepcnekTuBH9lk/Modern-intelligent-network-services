"""REST API агроскоринга: лабораторная работа № 1."""

import logging
import time

from fastapi import FastAPI, HTTPException, Query, Request, status

from model import MODEL_NAME, MODEL_TYPE, MODEL_VERSION
from schemas import FarmRequest, HealthResponse, ModelInfoResponse, PredictionResponse
from services import create_prediction, find_prediction, list_predictions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agro Scoring API",
    description="REST API для оценки риска сельскохозяйственных предприятий.",
    version="1.0.0",
)

# Как в примере преподавателя: False отключает инференс с HTTP 503.
MODEL_READY = True


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    """Измеряет длительность запроса и добавляет X-Process-Time в секундах."""
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Request failed | method=%s | path=%s", request.method, request.url.path)
        raise
    process_time = time.perf_counter() - start_time
    response.headers["X-Process-Time"] = str(round(process_time, 6))
    logger.info(
        "HTTP request | method=%s | path=%s | status=%s | duration=%.6f",
        request.method, request.url.path, response.status_code, process_time,
    )
    return response


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
    description="Используется для проверки того, что REST API запущен и отвечает.",
)
def health():
    return {"status": "ok"}


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
    description="Возвращает название, версию, тип и текущее состояние модели.",
)
def model_info():
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "model_type": MODEL_TYPE,
        "status": "ready" if MODEL_READY else "unavailable",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Оценить риск хозяйства",
    description=(
        "Принимает характеристики хозяйства, выполняет структурную и бизнес-валидацию, "
        "рассчитывает риск, сохраняет прогноз и возвращает его вместе с UUID запроса."
    ),
    responses={
        400: {"description": "Неизвестный регион хозяйства"},
        503: {"description": "Модель временно недоступна"},
    },
)
def predict(request: FarmRequest):
    # Проверка готовности предшествует бизнес-валидации, как у преподавателя.
    if not MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is temporarily unavailable",
        )
    return create_prediction(request)


@app.get(
    "/predictions",
    response_model=list[PredictionResponse],
    summary="Получить список прогнозов",
    description=(
        "Возвращает прогнозы в порядке добавления. Поддерживает ограничение "
        "количества результатов и фильтрацию по уровню риска."
    ),
    responses={400: {"description": "Недопустимая категория риска"}},
)
def get_predictions(
    limit: int = Query(default=10, ge=1, le=100, description="Максимальное количество результатов"),
    risk_level: str | None = Query(default=None, description="Фильтр по категории риска: low, medium или high"),
):
    return list_predictions(limit, risk_level)


@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получить прогноз по request_id",
    description="Возвращает сохраненный прогноз по его уникальному идентификатору.",
    responses={404: {"description": "Прогноз с указанным request_id не найден"}},
)
def get_prediction(request_id: str):
    return find_prediction(request_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
