import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, ProxyHandler
from uuid import UUID

ROOT = Path(__file__).resolve().parent
OPENER = build_opener(ProxyHandler({}))
FARM = {
    "farm_id": "FARM-001",
    "region": "Krasnodar",
    "crop_type": "wheat",
    "area_ha": 2500,
    "temperature_avg": 24.3,
    "precipitation_mm": 320,
    "payment_delay_days": 15,
    "previous_defaults": 0,
    "debt": 1500000,
}


def request(base_url, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(base_url + path, data=data, headers={"Content-Type": "application/json"})
    try:
        response = OPENER.open(req, timeout=5)
    except HTTPError as error:
        response = error
    with response:
        raw = response.read().decode("utf-8")
        body = json.loads(raw) if "application/json" in response.headers.get("Content-Type", "") else raw
        return response.status, body, response.headers


@contextmanager
def running_server(ready=True):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    if ready:
        command = [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", str(port)]
    else:
        command = [
            sys.executable,
            "-c",
            "import main, uvicorn; main.MODEL_READY = False; "
            f"uvicorn.run(main.app, host='127.0.0.1', port={port})",
        ]
    with tempfile.TemporaryFile() as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=log,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    break
                try:
                    if request(url, "/health")[0] == 200:
                        yield url
                        return
                except (URLError, OSError):
                    time.sleep(0.1)
            log.seek(0)
            raise RuntimeError("uvicorn failed to start:\n" + log.read().decode("utf-8", errors="replace"))
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = running_server()
        cls.url = cls.server.__enter__()
        cls.addClassCleanup(cls.server.__exit__, None, None, None)
        cls.samples = []
        for payload in [
            FARM,
            {**FARM, "payment_delay_days": 45},
            {
                **FARM,
                "payment_delay_days": 45,
                "previous_defaults": 1,
                "debt": 6500000,
            },
        ]:
            code, body, _ = request(cls.url, "/predict", payload)
            if code != 200:
                raise AssertionError(body)
            cls.samples.append(body)

    def check(self, path, expected, payload=None):
        code, body, headers = request(self.url, path, payload)
        self.assertEqual(code, expected, body)
        self.assertGreaterEqual(float(headers["X-Process-Time"]), 0)
        return body

    def test_01_health_200(self):
        self.assertEqual(self.check("/health", 200), {"status": "ok"})

    def test_02_model_info_200(self):
        self.assertEqual(
            self.check("/model-info", 200),
            {
                "model_name": "agro-risk-model",
                "model_version": "1.0",
                "model_type": "risk-scoring",
                "status": "ready",
            },
        )

    def test_03_predict_200_and_risk_profiles(self):
        for sample, score, level, recommendation in zip(
            self.samples,
            [0.1, 0.4, 0.9],
            ["low", "medium", "high"],
            [
                "Стандартное рассмотрение",
                "Требуется дополнительная проверка",
                "Высокий риск. Требуется ручное рассмотрение",
            ],
        ):
            with self.subTest(level=level):
                self.assertEqual(sample["risk_score"], score)
                self.assertEqual(sample["risk_level"], level)
                self.assertEqual(sample["recommendation"], recommendation)
                self.assertEqual(UUID(sample["request_id"]).version, 4)
                self.assertEqual(
                    set(sample),
                    {
                        "request_id",
                        "farm_id",
                        "risk_score",
                        "risk_level",
                        "recommendation",
                        "model_version",
                    },
                )
        result = self.check("/predict", 200, FARM)
        self.assertNotIn(result["request_id"], [sample["request_id"] for sample in self.samples])

    def test_04_negative_area_422(self):
        body = self.check("/predict", 422, {**FARM, "area_ha": -100})
        self.assertEqual(body["detail"][0]["loc"], ["body", "area_ha"])

    def test_05_unknown_region_400(self):
        body = self.check("/predict", 400, {**FARM, "region": "Unknown"})
        self.assertEqual(
            body["detail"],
            "Unknown region: Unknown. Allowed regions: ['Krasnodar', 'Rostov', 'Stavropol']",
        )

    def test_06_existing_prediction_200(self):
        sample = self.samples[2]
        self.assertEqual(self.check("/predictions/" + sample["request_id"], 200), sample)

    def test_07_unknown_prediction_404(self):
        self.assertEqual(
            self.check("/predictions/not-found", 404),
            {"detail": "Prediction not found"},
        )

    def test_08_limit_two_200(self):
        self.assertEqual(self.check("/predictions?limit=2", 200), self.samples[:2])

    def test_09_filter_high_200(self):
        self.assertEqual(self.check("/predictions?risk_level=high", 200), [self.samples[2]])

    def test_10_negative_limit_422(self):
        self.check("/predictions?limit=-5", 422)

    def test_11_swagger_html_200(self):
        self.assertIn("SwaggerUIBundle", self.check("/docs", 200))

    def test_12_openapi_contract_200(self):
        api = self.check("/openapi.json", 200)
        self.assertEqual(
            set(api["paths"]),
            {
                "/health",
                "/model-info",
                "/predict",
                "/predictions",
                "/predictions/{request_id}",
            },
        )
        self.assertEqual(set(api["components"]["schemas"]["FarmRequest"]["required"]), set(FARM))
        expected_codes = {
            ("/predict", "post"): {"200", "400", "422", "503"},
            ("/predictions", "get"): {"200", "400", "422"},
            ("/predictions/{request_id}", "get"): {"200", "404", "422"},
        }
        for path, methods in api["paths"].items():
            for method, operation in methods.items():
                self.assertTrue(operation["summary"])
                self.assertTrue(operation["description"])
                self.assertIn("schema", operation["responses"]["200"]["content"]["application/json"])
                if (path, method) in expected_codes:
                    self.assertEqual(set(operation["responses"]), expected_codes[path, method])

    def test_13_query_validation(self):
        self.check("/predictions?risk_level=critical", 400)
        for limit in [0, 101, "abc"]:
            with self.subTest(limit=limit):
                self.check(f"/predictions?limit={limit}", 422)
        self.assertEqual(len(self.check("/predictions?limit=1", 200)), 1)
        self.check("/predictions?limit=100", 200)
        self.assertEqual(
            self.check("/predictions?risk_level=medium&limit=5", 200),
            [self.samples[1]],
        )

    def test_14_structural_validation_no_storage_mutation(self):
        before = self.check("/predictions?limit=100", 200)
        for field, value in [
            ("area_ha", 0),
            ("precipitation_mm", -1),
            ("payment_delay_days", -1),
            ("previous_defaults", -1),
            ("debt", -1),
            ("temperature_avg", -61),
            ("temperature_avg", 61),
            ("farm_id", ""),
            ("region", ""),
            ("crop_type", ""),
            ("area_ha", "not-a-number"),
            ("previous_defaults", 1.5),
        ]:
            with self.subTest(field=field, value=value):
                self.check("/predict", 422, {**FARM, field: value})
        missing = dict(FARM)
        missing.pop("debt")
        self.check("/predict", 422, missing)
        self.check("/predict", 400, {**FARM, "region": "unknown"})
        self.assertEqual(self.check("/predictions?limit=100", 200), before)

    def test_15_scoring_boundaries(self):
        cases = [
            ({"payment_delay_days": 30, "debt": 5000000, "precipitation_mm": 100}, 0.1, "low"),
            ({"payment_delay_days": 31}, 0.4, "medium"),
            ({"debt": 5000001}, 0.3, "medium"),
            ({"precipitation_mm": 99}, 0.2, "low"),
            ({"payment_delay_days": 31, "previous_defaults": 1}, 0.7, "high"),
            (
                {
                    "payment_delay_days": 31,
                    "previous_defaults": 1,
                    "debt": 5000001,
                    "precipitation_mm": 99,
                },
                1.0,
                "high",
            ),
        ]
        for changes, score, level in cases:
            with self.subTest(changes=changes):
                body = self.check("/predict", 200, {**FARM, **changes})
                self.assertEqual(body["risk_score"], score)
                self.assertEqual(body["risk_level"], level)
        for region in ["Rostov", "Stavropol"]:
            self.check("/predict", 200, {**FARM, "region": region})

    def test_16_unavailable_model_503(self):
        with running_server(ready=False) as url:
            self.assertEqual(request(url, "/health")[1], {"status": "ok"})
            self.assertEqual(request(url, "/model-info")[1]["status"], "unavailable")
            for payload in [FARM, {**FARM, "region": "unknown"}]:
                code, body, headers = request(url, "/predict", payload)
                self.assertEqual(code, 503)
                self.assertEqual(body, {"detail": "Model is temporarily unavailable"})
                self.assertGreaterEqual(float(headers["X-Process-Time"]), 0)
            self.assertEqual(request(url, "/predict", {**FARM, "area_ha": -100})[0], 422)
            self.assertEqual(request(url, "/predictions")[1], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
