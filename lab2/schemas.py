from pydantic import BaseModel, Field

from .config import LLM_DEFAULT_MAX_TOKENS, LLM_DEFAULT_TEMPERATURE, LLM_MODEL, MAX_BATCH_SIZE


class FarmRequest(BaseModel):
    farm_id: str = Field(..., min_length=1, description="Идентификатор хозяйства")
    region: str = Field(..., min_length=1, description="Регион")
    crop_type: str = Field(..., min_length=1, description="Основная сельскохозяйственная культура")
    area_ha: float = Field(..., gt=0, description="Площадь посевов, га")
    temperature_avg: float = Field(..., ge=-60, le=60, description="Средняя температура, °C")
    precipitation_mm: float = Field(..., ge=0, description="Количество осадков, мм")
    payment_delay_days: int = Field(..., ge=0, description="Количество дней просрочки платежа")
    previous_defaults: int = Field(..., ge=0, description="Количество предыдущих дефолтов")
    debt: float = Field(..., ge=0, description="Текущая задолженность")


class PredictionResponse(BaseModel):
    request_id: str
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    model_type: str
    status: str


class BatchRequest(BaseModel):
    items: list[FarmRequest] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description="Список объектов для пакетного инференса",
    )


class BatchItemResponse(BaseModel):
    farm_id: str
    risk_score: float
    risk_level: str
    recommendation: str
    model_version: str


class BatchPredictionResponse(BaseModel):
    request_id: str
    batch_size: int
    processing_time_ms: float
    items: list[BatchItemResponse]


class QueueStatusResponse(BaseModel):
    queue_size: int
    queue_max_size: int
    max_batch_size: int
    max_batch_wait_ms: int
    inference_timeout_seconds: float


class PaginationMeta(BaseModel):
    weekday: str
    current_page: int
    per_page: int
    total_pages: int
    total_items: int
    next_page: str | None
    prev_page: str | None


class PaginatedPredictionsResponse(BaseModel):
    data: list[PredictionResponse]
    meta: PaginationMeta


class LLMChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000, description="Текст запроса к LLM")
    model: str = Field(default=LLM_MODEL, min_length=1, description="Модель Vireonix")
    temperature: float = Field(default=LLM_DEFAULT_TEMPERATURE, ge=0, le=2)
    max_tokens: int = Field(default=LLM_DEFAULT_MAX_TOKENS, ge=1, le=2000)


class LLMChatResponse(BaseModel):
    request_id: str
    requested_model: str
    returned_model: str | None
    answer: str
    reasoning: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None
    latency_ms: float


class PredictWithLLMResponse(BaseModel):
    prediction: PredictionResponse
    llm: LLMChatResponse


class ErrorResponse(BaseModel):
    detail: str | list[dict]
    request_id: str | None


class SeedResponse(BaseModel):
    status: str
    created: int
    message: str
