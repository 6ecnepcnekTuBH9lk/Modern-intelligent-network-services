from dataclasses import dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"
MAX_BATCH_SIZE = 8
LLM_MODEL = "auto"
LLM_DEFAULT_TEMPERATURE = 0.3
LLM_DEFAULT_MAX_TOKENS = 300
ALLOWED_REGIONS = {"Krasnodar", "Rostov", "Stavropol"}
WEEKDAY_MAP = {
    "sunday": "0", "monday": "1", "tuesday": "2", "wednesday": "3",
    "thursday": "4", "friday": "5", "saturday": "6",
}


@dataclass(frozen=True)
class Settings:
    database_path: Path = BASE_DIR / "agro_scoring.db"
    db_busy_timeout_ms: int = 5000
    model_ready: bool = True
    max_batch_size: int = MAX_BATCH_SIZE
    max_batch_wait_ms: int = 50
    queue_max_size: int = 100
    inference_timeout_seconds: float = 2.0
    model_fixed_overhead_seconds: float = 0.020
    model_per_item_seconds: float = 0.003
    llm_base_url: str = "https://vireonix.ai/v1"
    llm_timeout_seconds: float = 60.0
