import asyncio

from .config import Settings
from .schemas import FarmRequest


def calculate_risk(data: FarmRequest) -> float:
    score = 0.1
    if data.payment_delay_days > 30:
        score += 0.3
    if data.previous_defaults > 0:
        score += 0.3
    if data.debt > 5_000_000:
        score += 0.2
    if data.precipitation_mm < 100:
        score += 0.1
    return round(min(score, 1.0), 2)


class InferenceEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model_lock = asyncio.Lock()

    async def model_predict_single(self, data: FarmRequest) -> float:
        async with self.model_lock:
            await asyncio.sleep(
                self.settings.model_fixed_overhead_seconds + self.settings.model_per_item_seconds
            )
            return calculate_risk(data)

    async def model_predict_batch(self, items: list[FarmRequest]) -> list[float]:
        async with self.model_lock:
            await asyncio.sleep(
                self.settings.model_fixed_overhead_seconds
                + self.settings.model_per_item_seconds * len(items)
            )
            return [calculate_risk(item) for item in items]
