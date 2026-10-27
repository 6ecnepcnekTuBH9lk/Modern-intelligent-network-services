"""Проверка настоящего LLM API через запущенное приложение, без mock-ответов."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from uuid import uuid4

import httpx

from lab2.config import BASE_DIR
from lab2.tests.support import HIGH_FARM


async def check(base_url):
    async with httpx.AsyncClient(base_url=base_url, timeout=70, trust_env=False) as client:
        async def call(path, payload):
            request_id = "live-" + str(uuid4())
            response = await client.post(path, json=payload, headers={"X-Request-ID": request_id})
            result = {
                "endpoint": path, "http_status": response.status_code, "request_id": request_id,
                "response": response.json(), "x_process_time": response.headers.get("X-Process-Time"),
            }
            if path == "/predict-with-llm":
                stored = await client.get("/predictions/" + request_id)
                result["stored_prediction_status"] = stored.status_code
                result["stored_prediction"] = stored.json()
            return result
        results = await asyncio.gather(
            call("/llm/chat", {"prompt": "Ответь одним словом: работает", "model": "auto", "max_tokens": 30}),
            call("/predict-with-llm", HIGH_FARM),
        )
    output = BASE_DIR / "checks" / "live_llm.json"
    output.parent.mkdir(exist_ok=True)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "mock_used": False, "results": results}
    if output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        report["previous_checks"] = previous.get("previous_checks", []) + [
            {key: value for key, value in previous.items() if key != "previous_checks"}
        ]
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8002")
    asyncio.run(check(parser.parse_args().url))
