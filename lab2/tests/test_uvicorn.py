import asyncio
from collections import Counter
from contextlib import closing, contextmanager
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest

import httpx

from lab2.config import BASE_DIR
from lab2.tests.support import FARM, HIGH_FARM


@contextmanager
def running_server(directory: Path, **settings):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings_code = ", ".join(f"{key}={value!r}" for key, value in settings.items())
    code = (
        "from pathlib import Path; import uvicorn; "
        "from lab2.main import create_app; from lab2.config import Settings; "
        f"app=create_app(Settings(database_path=Path({str(directory / 'test.db')!r}), {settings_code})); "
        f"uvicorn.run(app, host='127.0.0.1', port={port})"
    )
    url = f"http://127.0.0.1:{port}"
    with (directory / "uvicorn.log").open("w+", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", code], cwd=BASE_DIR.parent, stdout=log, stderr=log,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            with httpx.Client(base_url=url, trust_env=False, timeout=1) as client:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.RequestError:
                        time.sleep(0.1)
                else:
                    raise RuntimeError("uvicorn startup timeout")
                if process.poll() is not None:
                    log.seek(0)
                    raise RuntimeError(log.read())
            yield url
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    # Виртуальное окружение Windows запускает Python через дочерний процесс.
                    subprocess.run(
                        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
                    )
                else:
                    process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


class UvicornTests(unittest.TestCase):
    def setUp(self):
        parent = BASE_DIR / ".test-data"
        parent.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def check(self, response, code):
        self.assertEqual(response.status_code, code, response.text)
        self.assertTrue(response.headers["X-Request-ID"])
        self.assertGreaterEqual(float(response.headers["X-Process-Time"]), 0)
        if code >= 400:
            self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])
        return response.json()

    def test_real_http_endpoints_sqlite_and_restart_persistence(self):
        with running_server(self.directory) as url, httpx.Client(base_url=url, trust_env=False, timeout=10) as client:
            self.check(client.get("/health"), 200)
            self.check(client.get("/model-info"), 200)
            docs = client.get("/docs")
            self.assertEqual(docs.status_code, 200)
            self.assertIn("SwaggerUIBundle", docs.text)
            self.assertEqual(len(self.check(client.get("/openapi.json"), 200)["paths"]), 14)
            prediction = self.check(client.post("/predict", json=HIGH_FARM, headers={"X-Request-ID": "http-single"}), 200)
            self.assertEqual(prediction["risk_score"], 0.9)
            self.check(client.post("/predict", json={**FARM, "area_ha": -100}), 422)
            self.check(client.post("/predict", json={**FARM, "region": "bad"}), 400)
            self.check(client.get("/predictions/http-single"), 200)
            self.check(client.get("/predictions/missing"), 404)
            self.check(client.get("/predictions?limit=-5"), 422)
            self.check(client.get("/predictions?risk_level=high"), 200)
            self.check(client.post("/batch-predict", json={"items": [FARM, HIGH_FARM]},
                headers={"X-Request-ID": "http-batch"}), 200)
            self.check(client.post("/batch-predict", json={"items": [FARM] * 9}), 422)
            self.check(client.get("/predictions/http-batch"), 409)
            self.assertEqual(len(self.check(client.get("/requests/http-batch/predictions"), 200)), 2)
            self.check(client.post("/predict-queued", json=FARM), 200)
            self.assertEqual(self.check(client.get("/queue-status"), 200)["queue_size"], 0)
            self.check(client.post("/demo/seed-weekdays?count=70"), 200)
            self.assertEqual(len(self.check(client.get("/predictions?limit=2"), 200)), 2)
            self.check(client.get("/predictions/by-weekday?weekday=monday&page=1&per_page=5"), 200)
            self.check(client.get("/predictions/by-weekday?weekday=monday&page=-1"), 422)
            self.check(client.get("/predictions/by-weekday?weekday=bad"), 400)
        with closing(sqlite3.connect(self.directory / "test.db")) as db:
            self.assertEqual(db.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(db.execute("SELECT source FROM predictions WHERE request_id='http-single'").fetchone()[0], "single")
            count = db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        with running_server(self.directory) as url, httpx.Client(base_url=url, trust_env=False) as client:
            self.assertEqual(self.check(client.get("/predictions/http-single"), 200), prediction)
        with closing(sqlite3.connect(self.directory / "test.db")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], count)

    def test_real_http_inference_timeout(self):
        with running_server(self.directory, inference_timeout_seconds=0.001) as url:
            with httpx.Client(base_url=url, trust_env=False) as client:
                for path in ["/predict", "/batch-predict", "/predict-queued", "/predict-with-llm"]:
                    payload = {"items": [FARM]} if path == "/batch-predict" else FARM
                    self.check(client.post(path, json=payload), 504)
            with closing(sqlite3.connect(self.directory / "test.db")) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], 0)

    def test_real_http_backpressure_and_concurrent_future_mapping(self):
        async def load(url):
            async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=20,
                limits=httpx.Limits(max_connections=180, max_keepalive_connections=180)) as client:
                return await asyncio.gather(*[
                    client.post("/predict-queued", json={**FARM, "farm_id": f"LOAD-{i}"},
                                headers={"X-Request-ID": f"load-{i}"}) for i in range(160)
                ])
        with running_server(self.directory, model_fixed_overhead_seconds=0.3, inference_timeout_seconds=15) as url:
            responses = asyncio.run(load(url))
            counts = Counter(response.status_code for response in responses)
            self.assertGreater(counts[200], 0)
            self.assertGreater(counts[429], 0)
            self.assertEqual(counts[200] + counts[429], 160)
            for i, response in enumerate(responses):
                self.check(response, response.status_code)
                self.assertEqual(response.headers["X-Request-ID"], f"load-{i}")
                if response.status_code == 200:
                    self.assertEqual(response.json()["farm_id"], f"LOAD-{i}")
                else:
                    self.assertEqual(response.headers["Retry-After"], "1")
        with closing(sqlite3.connect(self.directory / "test.db")) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0], counts[200])
        log = (self.directory / "uvicorn.log").read_text(encoding="utf-8")
        self.assertIn("Dynamic batch formed | size=8", log)
        print("HTTP backpressure: " + json.dumps(dict(counts), sort_keys=True), flush=True)
