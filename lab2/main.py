import logging
import time
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from .batching import BatchProcessor
from .config import MODEL_NAME, MODEL_TYPE, MODEL_VERSION, Settings
from .database import Database
from .llm import LLMClient
from .model import InferenceEngine
from .schemas import (
    BatchPredictionResponse, BatchRequest, ErrorResponse, FarmRequest, HealthResponse,
    LLMChatRequest, LLMChatResponse, ModelInfoResponse, PaginatedPredictionsResponse,
    PredictionResponse, PredictWithLLMResponse, QueueStatusResponse, SeedResponse,
)
from .services import PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def errors(*codes: int) -> dict:
    descriptions = {
        400: "Нарушение бизнес-правил", 404: "Результат или страница не найдены",
        409: "request_id связан с несколькими прогнозами", 422: "Ошибка валидации входных данных",
        429: "Очередь переполнена", 500: "Внутренняя ошибка приложения",
        502: "Ошибка ответа внешнего LLM API", 503: "Модель, БД или внешний сервис недоступны",
        504: "Превышено время ожидания инференса или LLM",
    }
    return {code: {"model": ErrorResponse, "description": descriptions[code]} for code in codes}


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db = Database(settings)
        await db.initialize()
        engine = InferenceEngine(settings)
        batches = BatchProcessor(settings, engine)
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.llm_timeout_seconds)) as client:
            app.state.database = db
            app.state.engine = engine
            app.state.batches = batches
            app.state.llm = LLMClient(settings, client)
            app.state.service = PredictionService(settings, db, engine, batches)
            batches.start()
            try:
                yield
            finally:
                await batches.close()

    app = FastAPI(
        title="Agro Scoring API",
        description=(
            "Агроскоринг: сквозной request_id, SQLite/aiosqlite, одиночный и пакетный "
            "инференс, очередь и интеграция с Vireonix LLM. Для передачи своего "
            "идентификатора используйте HTTP-заголовок X-Request-ID."
        ),
        version="4.0.0",
        lifespan=lifespan,
        responses=errors(500),
    )
    app.state.settings = settings

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        started = time.perf_counter()
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        logger.info(
            "Request started | request_id=%s | method=%s | path=%s",
            request_id, request.method, request.url.path,
        )
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled error | request_id=%s", request_id)
            response = JSONResponse(
                status_code=500, content={"detail": "Internal server error", "request_id": request_id},
            )
        elapsed = time.perf_counter() - started
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(round(elapsed, 6))
        logger.info(
            "Request completed | request_id=%s | status=%s | time=%.6f",
            request_id, response.status_code, elapsed,
        )
        return response

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": getattr(request.state, "request_id", None)},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422, content={
                "detail": jsonable_encoder(exc.errors()),
                "request_id": getattr(request.state, "request_id", None),
            },
        )

    @app.get("/health", response_model=HealthResponse, summary="Проверка состояния API",
             description="Проверяет, что HTTP API запущен и отвечает.")
    async def health():
        return {"status": "ok"}

    @app.get("/model-info", response_model=ModelInfoResponse, summary="Информация о модели",
             description="Название, версия, тип и готовность скоринговой модели.")
    async def model_info():
        return {"model_name": MODEL_NAME, "model_version": MODEL_VERSION, "model_type": MODEL_TYPE,
                "status": "ready" if settings.model_ready else "unavailable"}

    @app.post("/predict", response_model=PredictionResponse, summary="Одиночный инференс",
              description="Рассчитывает агрориск и возвращает HTTP 200 после сохранения результата в SQLite.",
              responses=errors(400, 422, 503, 504))
    async def predict(data: FarmRequest, request: Request):
        return await request.app.state.service.predict(data, request.state.request_id)

    @app.post("/batch-predict", response_model=BatchPredictionResponse, summary="Явный пакетный инференс",
              description="Один вызов модели для 1–8 хозяйств; одна транзакция SQLite для всего пакета.",
              responses=errors(400, 422, 503, 504))
    async def batch_predict(batch: BatchRequest, request: Request):
        return await request.app.state.service.predict_batch(batch.items, request.state.request_id)

    @app.post("/predict-queued", response_model=PredictionResponse, summary="Инференс через очередь",
              description="Помещает одно хозяйство в ограниченную очередь с динамическим объединением запросов.",
              responses=errors(400, 422, 429, 503, 504))
    async def predict_queued(data: FarmRequest, request: Request):
        return await request.app.state.service.predict_queued(data, request.state.request_id)

    @app.get("/queue-status", response_model=QueueStatusResponse, summary="Состояние очереди",
             description="Текущая длина очереди и ограничения batching и инференса.")
    async def queue_status(request: Request):
        return request.app.state.batches.status()

    @app.get("/predictions", response_model=list[PredictionResponse], summary="История прогнозов",
             description="Читает прогнозы из SQLite: новые записи первыми, фильтрация выполняется до LIMIT.",
             responses=errors(400, 422, 503))
    async def get_predictions(
        request: Request, limit: int = Query(default=10, ge=1, le=100),
        risk_level: str | None = Query(default=None, description="low, medium или high"),
    ):
        return await request.app.state.service.list_predictions(limit, risk_level)

    @app.get("/requests/{request_id}/predictions", response_model=list[PredictionResponse],
             summary="Все прогнозы одного HTTP-запроса",
             description="Возвращает все строки с указанным request_id, включая явный batch.",
             responses=errors(404, 422, 503))
    async def get_request_predictions(request_id: str, request: Request):
        return await request.app.state.service.request_predictions(request_id)

    @app.get("/predictions/by-weekday", response_model=PaginatedPredictionsResponse,
             summary="Пагинация прогнозов по дню недели",
             description="Фильтрация по created_at в UTC; LIMIT/OFFSET, COUNT и ссылки на соседние страницы.",
             responses=errors(400, 404, 422, 503))
    async def get_predictions_by_weekday(
        request: Request,
        weekday: str = Query(..., description="monday, tuesday, wednesday, thursday, friday, saturday, sunday"),
        page: int = Query(default=1, ge=1), per_page: int = Query(default=5, ge=1, le=100),
    ):
        return await request.app.state.service.weekday_page(request.url, weekday, page, per_page)

    @app.get("/predictions/{request_id}", response_model=PredictionResponse,
             summary="Получить одиночный прогноз",
             description="Возвращает 404 при отсутствии данных и 409 при нескольких прогнозах с одним request_id.",
             responses=errors(404, 409, 422, 503))
    async def get_prediction(request_id: str, request: Request):
        return await request.app.state.service.get_prediction(request_id)

    @app.post("/llm/chat", response_model=LLMChatResponse, summary="Прямой запрос к Vireonix LLM",
              description="Асинхронный вызов OpenAI-совместимого API без API-ключа и Authorization.",
              responses=errors(422, 502, 503, 504))
    async def llm_chat(data: LLMChatRequest, request: Request):
        return await request.app.state.llm.chat(
            data.prompt, request.state.request_id, data.model, data.temperature, data.max_tokens,
        )

    @app.post("/predict-with-llm", response_model=PredictWithLLMResponse,
              summary="Агроскоринг с объяснением LLM",
              description=(
                  "Сохраняет детерминированный прогноз, затем запрашивает объяснение. "
                  "LLM не рассчитывает риск; farm_id и исходные финансовые признаки ей не передаются."
              ), responses=errors(400, 422, 502, 503, 504))
    async def predict_with_llm(data: FarmRequest, request: Request):
        return await request.app.state.service.predict_with_llm(data, request.state.request_id, request.app.state.llm)

    @app.post("/demo/seed-weekdays", response_model=SeedResponse, summary="Данные для проверки пагинации",
              description="Создаёт демонстрационные прогнозы за последние 21 день, как в примере преподавателя.",
              responses=errors(422, 503))
    async def seed_demo_weekdays(request: Request, count: int = Query(default=70, ge=7, le=1000)):
        return await request.app.state.service.seed_weekdays(count)

    @app.get("/", response_model=dict[str, str], summary="Информация о сервисе",
             description="Ссылки на Swagger и проверку работоспособности API.")
    async def root():
        return {"message": "Agro Scoring API is running", "docs": "/docs", "health": "/health"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("lab2.main:app", host="127.0.0.1", port=8002, reload=True)
