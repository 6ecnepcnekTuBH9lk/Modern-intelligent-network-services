import asyncio
import logging
from dataclasses import dataclass

from fastapi import HTTPException

from .config import Settings
from .model import InferenceEngine
from .schemas import FarmRequest

logger = logging.getLogger(__name__)


@dataclass
class InferenceTask:
    data: FarmRequest
    request_id: str
    future: asyncio.Future


class BatchProcessor:
    def __init__(self, settings: Settings, engine: InferenceEngine):
        self.settings = settings
        self.engine = engine
        self.queue = asyncio.Queue(maxsize=settings.queue_max_size)
        self.worker: asyncio.Task | None = None

    def start(self) -> None:
        self.worker = asyncio.create_task(self.run(), name="agro-batch-worker")

    async def close(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            try:
                await self.worker
            except asyncio.CancelledError:
                pass
        while not self.queue.empty():
            task = self.queue.get_nowait()
            if not task.future.done():
                task.future.set_exception(HTTPException(503, "Inference worker is unavailable"))
            self.queue.task_done()

    async def submit(self, data: FarmRequest, request_id: str) -> float:
        future = asyncio.get_running_loop().create_future()
        try:
            self.queue.put_nowait(InferenceTask(data, request_id, future))
        except asyncio.QueueFull as exc:
            raise HTTPException(
                429, "Inference queue is full. Try again later.", headers={"Retry-After": "1"},
            ) from exc
        logger.info("Request queued | request_id=%s | queue_size=%s", request_id, self.queue.qsize())
        try:
            return await asyncio.wait_for(future, timeout=self.settings.inference_timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(504, "Inference timeout") from exc

    async def run(self) -> None:
        logger.info("Batch worker started")
        while True:
            batch = []
            try:
                first = await self.queue.get()
                batch.append(first)
                if first.future.done():
                    continue
                loop = asyncio.get_running_loop()
                deadline = loop.time() + self.settings.max_batch_wait_ms / 1000
                while len(batch) < self.settings.max_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        task = await asyncio.wait_for(self.queue.get(), timeout=remaining)
                    except TimeoutError:
                        break
                    if task.future.done():
                        self.queue.task_done()
                    else:
                        batch.append(task)
                active = [task for task in batch if not task.future.done()]
                if not active:
                    continue
                logger.info(
                    "Dynamic batch formed | size=%s | queue_size=%s | request_ids=%s",
                    len(active), self.queue.qsize(), [task.request_id for task in active],
                )
                scores = await asyncio.wait_for(
                    self.engine.model_predict_batch([task.data for task in active]),
                    timeout=self.settings.inference_timeout_seconds,
                )
                for task, score in zip(active, scores, strict=True):
                    if not task.future.done():
                        task.future.set_result(score)
            except asyncio.CancelledError:
                for task in batch:
                    if not task.future.done():
                        task.future.set_exception(HTTPException(503, "Inference worker is unavailable"))
                raise
            except Exception as exc:
                logger.exception("Batch inference failed")
                for task in batch:
                    if not task.future.done():
                        task.future.set_exception(exc)
            finally:
                for _ in batch:
                    self.queue.task_done()

    def status(self) -> dict:
        return {
            "queue_size": self.queue.qsize(),
            "queue_max_size": self.settings.queue_max_size,
            "max_batch_size": self.settings.max_batch_size,
            "max_batch_wait_ms": self.settings.max_batch_wait_ms,
            "inference_timeout_seconds": self.settings.inference_timeout_seconds,
        }
