import asyncio
import time

from lab2.batching import InferenceTask
from lab2.model import calculate_risk
from lab2.schemas import FarmRequest
from lab2.tests.support import APICase, FARM


class BatchingTests(APICase):
    async def test_dynamic_batches_preserve_future_mapping_and_size(self):
        calls = []
        original = self.app.state.engine.model_predict_batch
        async def observed(items):
            calls.append([item.farm_id for item in items])
            return await original(items)
        self.app.state.engine.model_predict_batch = observed
        payloads = [{**FARM, "farm_id": f"QUEUED-{i}", "debt": 6000000 if i % 2 else 0} for i in range(17)]
        responses = await asyncio.gather(*[
            self.client.post("/predict-queued", json=payload, headers={"X-Request-ID": f"queued-{i}"})
            for i, payload in enumerate(payloads)
        ])
        for response, payload in zip(responses, payloads, strict=True):
            result = self.check(response, 200)
            self.assertEqual(result["farm_id"], payload["farm_id"])
            self.assertEqual(result["risk_score"], calculate_risk(FarmRequest(**payload)))
        self.assertEqual(sum(map(len, calls)), 17)
        self.assertTrue(all(len(items) <= 8 for items in calls))
        self.assertTrue(any(len(items) > 1 for items in calls))
        self.assertLess(len(calls), 17)
        await asyncio.wait_for(self.app.state.batches.queue.join(), 1)
        rows = await self.rows()
        self.assertEqual(len(rows), 17)
        self.assertTrue(all(row["source"] == "queued" for row in rows))

    async def test_explicit_batch_uses_one_model_call(self):
        calls = []
        original = self.app.state.engine.model_predict_batch
        async def observed(items):
            calls.append(len(items))
            return await original(items)
        self.app.state.engine.model_predict_batch = observed
        self.check(await self.client.post("/batch-predict", json={"items": [FARM] * 8}), 200)
        self.assertEqual(calls, [8])

    async def test_partial_batch_finishes_after_wait_window(self):
        start = time.perf_counter()
        self.check(await self.client.post("/predict-queued", json=FARM), 200)
        self.assertGreaterEqual(time.perf_counter() - start, self.settings.max_batch_wait_ms / 1000)

    async def test_queue_capacity_429_retry_header_and_status(self):
        processor = self.app.state.batches
        processor.worker.cancel()
        try:
            await processor.worker
        except asyncio.CancelledError:
            pass
        futures = []
        for i in range(100):
            future = asyncio.get_running_loop().create_future()
            futures.append(future)
            processor.queue.put_nowait(InferenceTask(FarmRequest(**FARM), str(i), future))
        status = self.check(await self.client.get("/queue-status"), 200)
        self.assertEqual(status, {"queue_size": 100, "queue_max_size": 100, "max_batch_size": 8,
                                  "max_batch_wait_ms": 50, "inference_timeout_seconds": 2.0})
        response = await self.client.post("/predict-queued", json=FARM)
        self.check(response, 429)
        self.assertEqual(response.headers["Retry-After"], "1")
        for future in futures:
            future.cancel()
        await processor.close()
        await asyncio.wait_for(processor.queue.join(), 1)
        self.assertEqual(processor.queue.qsize(), 0)
        self.assertEqual(await self.rows(), [])

    async def test_worker_recovers_after_model_exception(self):
        original = self.app.state.engine.model_predict_batch
        async def fail(items):
            raise RuntimeError("model test failure")
        self.app.state.engine.model_predict_batch = fail
        self.check(await self.client.post("/predict-queued", json=FARM), 500)
        self.app.state.engine.model_predict_batch = original
        self.check(await self.client.post("/predict-queued", json=FARM), 200)
        self.assertFalse(self.app.state.batches.worker.done())

    async def test_worker_shutdown_resolves_pending_future(self):
        processor = self.app.state.batches
        pending = asyncio.create_task(processor.submit(FarmRequest(**FARM), "shutdown"))
        await asyncio.sleep(0.005)
        await processor.close()
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as error:
            await pending
        self.assertEqual(error.exception.status_code, 503)
        await asyncio.wait_for(processor.queue.join(), 1)
