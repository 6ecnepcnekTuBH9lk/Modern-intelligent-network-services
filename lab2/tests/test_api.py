import ast
import asyncio
from itertools import product
from uuid import UUID

from lab2.config import BASE_DIR
from lab2.model import calculate_risk
from lab2.schemas import FarmRequest
from lab2.tests.support import APICase, FARM, HIGH_FARM


class APITests(APICase):
    async def test_health_model_and_generated_request_id(self):
        response = await self.client.get("/health")
        self.assertEqual(self.check(response, 200), {"status": "ok"})
        self.assertEqual(UUID(response.headers["X-Request-ID"]).version, 4)
        model = self.check(await self.client.get("/model-info"), 200)
        self.assertEqual(model, {"model_name": "agro-risk-model", "model_version": "1.0",
                                 "model_type": "risk-scoring", "status": "ready"})

    async def test_request_id_in_response_logs_and_database(self):
        with self.assertLogs("lab2", level="INFO") as logs:
            response = await self.client.post("/predict", json=HIGH_FARM, headers={"X-Request-ID": "crm-001"})
        body = self.check(response, 200)
        self.assertEqual(body["request_id"], "crm-001")
        self.assertEqual(response.headers["X-Request-ID"], "crm-001")
        self.assertTrue(any("request_id=crm-001" in entry for entry in logs.output))
        self.assertEqual(body["risk_score"], 0.9)
        self.assertEqual(body["risk_level"], "high")
        self.assertEqual(body["recommendation"], "Высокий риск. Требуется ручное рассмотрение")
        rows = await self.rows()
        self.assertEqual(rows[0]["request_id"], "crm-001")
        self.assertEqual(rows[0]["source"], "single")
        self.assertTrue(rows[0]["created_at"])

    async def test_algorithm_matches_first_lab_for_all_combinations(self):
        tree = ast.parse((BASE_DIR.parent / "model.py").read_text(encoding="utf-8"))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "calculate_risk")
        scope = {"FarmRequest": FarmRequest}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "lab1-model", "exec"), scope)
        for delay, defaults, debt, rain in product([30, 31], [0, 1], [5000000, 5000001], [99, 100]):
            farm = FarmRequest(**{**FARM, "payment_delay_days": delay, "previous_defaults": defaults,
                                  "debt": debt, "precipitation_mm": rain})
            self.assertEqual(calculate_risk(farm), scope["calculate_risk"](farm))

    async def test_structural_validation_and_malformed_json(self):
        for field, value in [
            ("area_ha", -100), ("area_ha", 0), ("precipitation_mm", -1),
            ("payment_delay_days", -1), ("previous_defaults", -1), ("previous_defaults", 1.5),
            ("debt", -1), ("temperature_avg", -61), ("temperature_avg", 61),
            ("farm_id", ""), ("crop_type", ""), ("region", ""), ("area_ha", "bad"),
        ]:
            with self.subTest(field=field, value=value):
                self.check(await self.client.post("/predict", json={**FARM, field: value}), 422)
        self.check(await self.client.post("/predict", json={"area_ha": 10}), 422)
        self.check(await self.client.post("/predict", content="{", headers={"Content-Type": "application/json"}), 422)
        self.assertEqual(await self.rows(), [])

    async def test_business_validation_for_all_inference_paths(self):
        for path in ["/predict", "/predict-queued", "/predict-with-llm", "/batch-predict"]:
            payload = {**FARM, "region": "Unknown"}
            if path == "/batch-predict":
                payload = {"items": [FARM, payload]}
            with self.subTest(path=path):
                body = self.check(await self.client.post(path, json=payload), 400)
                self.assertIn("Allowed regions", body["detail"])
        self.assertEqual(await self.rows(), [])

    async def test_allowed_regions_and_boundary_values(self):
        for region in ["Krasnodar", "Rostov", "Stavropol"]:
            self.check(await self.client.post("/predict", json={**FARM, "region": region,
                "temperature_avg": -60, "precipitation_mm": 0, "payment_delay_days": 0,
                "previous_defaults": 0, "debt": 0}), 200)

    async def test_history_filter_order_limit_and_lookup(self):
        low = self.check(await self.client.post("/predict", json=FARM), 200)
        high = self.check(await self.client.post("/predict", json=HIGH_FARM), 200)
        self.assertEqual(self.check(await self.client.get("/predictions?limit=2"), 200), [high, low])
        self.assertEqual(self.check(await self.client.get("/predictions?risk_level=high"), 200), [high])
        self.assertEqual(self.check(await self.client.get("/predictions/" + low["request_id"]), 200), low)
        self.assertEqual(self.check(await self.client.get("/requests/" + low["request_id"] + "/predictions"), 200), [low])
        for limit in [-5, 0, 101, "abc"]:
            self.check(await self.client.get(f"/predictions?limit={limit}"), 422)
        self.check(await self.client.get("/predictions?risk_level=bad"), 400)
        self.check(await self.client.get("/predictions?limit=100"), 200)
        self.check(await self.client.get("/predictions/unknown"), 404)
        self.check(await self.client.get("/requests/unknown/predictions"), 404)

    async def test_explicit_batch_and_ambiguous_request_id(self):
        response = await self.client.post("/batch-predict", json={"items": [FARM, HIGH_FARM]}, headers={"X-Request-ID": "batch-1"})
        body = self.check(response, 200)
        self.assertEqual(body["batch_size"], 2)
        self.assertEqual(body["request_id"], "batch-1")
        self.assertEqual([item["risk_score"] for item in body["items"]], [0.1, 0.9])
        self.assertGreater(body["processing_time_ms"], 0)
        self.check(await self.client.get("/predictions/batch-1"), 409)
        results = self.check(await self.client.get("/requests/batch-1/predictions"), 200)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(row["source"] == "batch" for row in await self.rows()))

    async def test_batch_size_and_single_item_batch(self):
        for items in [[], [FARM] * 9]:
            self.check(await self.client.post("/batch-predict", json={"items": items}), 422)
        one = self.check(await self.client.post("/batch-predict", json={"items": [FARM]}), 200)
        self.check(await self.client.get("/predictions/" + one["request_id"]), 200)
        self.assertEqual(self.check(await self.client.post("/batch-predict", json={"items": [FARM] * 8}), 200)["batch_size"], 8)

    async def test_reusing_client_request_id_is_not_idempotent(self):
        for _ in range(2):
            self.check(await self.client.post("/predict", json=FARM, headers={"X-Request-ID": "same"}), 200)
        self.check(await self.client.get("/predictions/same"), 409)

    async def test_weekday_pagination_and_links(self):
        self.check(await self.client.post("/demo/seed-weekdays?count=70"), 200)
        rows = await self.rows()
        import datetime
        weekday = datetime.datetime.fromisoformat(rows[0]["created_at"]).strftime("%A").lower()
        first = self.check(await self.client.get(f"/predictions/by-weekday?weekday={weekday.upper()}&per_page=2"), 200)
        meta = first["meta"]
        self.assertEqual(meta["weekday"], weekday)
        self.assertEqual(meta["current_page"], 1)
        self.assertEqual(len(first["data"]), 2)
        self.assertGreater(meta["total_items"], 2)
        self.assertIsNone(meta["prev_page"])
        second = self.check(await self.client.get(meta["next_page"]), 200)
        self.assertEqual(second["meta"]["current_page"], 2)
        self.assertIsNotNone(second["meta"]["prev_page"])
        self.assertNotEqual(first["data"], second["data"])
        self.check(await self.client.get(f"/predictions/by-weekday?weekday={weekday}&page=999"), 404)
        self.check(await self.client.get("/predictions/by-weekday?weekday=bad"), 400)
        self.check(await self.client.get("/predictions/by-weekday?weekday=monday&page=-1"), 422)
        self.check(await self.client.get("/predictions/by-weekday?weekday=monday&per_page=101"), 422)
        self.check(await self.client.get("/predictions/by-weekday"), 422)

    async def test_empty_weekday(self):
        result = self.check(await self.client.get("/predictions/by-weekday?weekday=monday"), 200)
        self.assertEqual(result["data"], [])
        self.assertEqual(result["meta"]["total_pages"], 0)
        self.assertEqual(result["meta"]["total_items"], 0)
        self.check(await self.client.get("/predictions/by-weekday?weekday=monday&page=2"), 404)

    async def test_database_failure_and_atomic_batch_rollback(self):
        await self.sql("""CREATE TRIGGER fail_farm BEFORE INSERT ON predictions
            WHEN NEW.farm_id = 'FAIL' BEGIN SELECT RAISE(ABORT, 'test constraint'); END;""")
        body = self.check(await self.client.post("/batch-predict", json={"items": [FARM, {**FARM, "farm_id": "FAIL"}]}), 503)
        self.assertEqual(body["detail"], "Prediction storage is unavailable")
        self.assertEqual(await self.rows(), [])
        await self.sql("DROP TABLE predictions;")
        self.check(await self.client.post("/predict", json=FARM), 503)
        self.check(await self.client.get("/predictions"), 503)
        self.check(await self.client.get("/predictions/unknown"), 503)
        self.check(await self.client.get("/predictions/by-weekday?weekday=monday"), 503)
        self.check(await self.client.get("/health"), 200)

    async def test_database_locked_returns_503_without_partial_write(self):
        from dataclasses import replace
        import aiosqlite
        self.app.state.database.settings = replace(self.settings, db_busy_timeout_ms=10)
        async with aiosqlite.connect(self.db_path) as blocker:
            await blocker.execute("BEGIN IMMEDIATE")
            self.check(await self.client.post("/predict", json=FARM), 503)
            await blocker.rollback()
        self.assertEqual(await self.rows(), [])
        self.check(await self.client.post("/predict", json=FARM), 200)

    async def test_unhandled_error_is_500_and_service_recovers(self):
        engine = self.app.state.engine
        original = engine.model_predict_single
        async def fail(data):
            raise RuntimeError("internal test detail")
        engine.model_predict_single = fail
        response = await self.client.post("/predict", json=FARM)
        self.assertEqual(self.check(response, 500)["detail"], "Internal server error")
        self.assertNotIn("internal test detail", response.text)
        engine.model_predict_single = original
        self.check(await self.client.post("/predict", json=FARM), 200)
        self.check(await self.client.get("/no-such-route"), 404)
        self.check(await self.client.post("/health"), 405)

    async def test_openapi_and_swagger(self):
        self.assertIn("SwaggerUIBundle", (await self.client.get("/docs")).text)
        api = self.check(await self.client.get("/openapi.json"), 200)
        self.assertEqual(len(api["paths"]), 14)
        self.assertEqual(set(api["components"]["schemas"]["FarmRequest"]["required"]), set(FARM))
        for path, methods in api["paths"].items():
            for method, operation in methods.items():
                with self.subTest(path=path, method=method):
                    self.assertTrue(operation["summary"])
                    self.assertTrue(operation["description"])
                    self.assertIn("schema", operation["responses"]["200"]["content"]["application/json"])
                    self.assertIn("500", operation["responses"])
        responses = api["paths"]["/predict-queued"]["post"]["responses"]
        self.assertTrue({"200", "400", "422", "429", "500", "503", "504"} <= set(responses))


class UnavailableModelTests(APICase):
    settings_overrides = {"model_ready": False}

    async def test_model_not_ready_on_every_inference_path(self):
        for path in ["/predict", "/batch-predict", "/predict-queued", "/predict-with-llm"]:
            payload = {"items": [FARM]} if path == "/batch-predict" else FARM
            self.check(await self.client.post(path, json=payload), 503)
        self.check(await self.client.post("/predict", json={**FARM, "region": "bad"}), 503)
        self.check(await self.client.post("/predict", json={**FARM, "area_ha": -100}), 422)
        self.assertEqual(self.check(await self.client.get("/model-info"), 200)["status"], "unavailable")
        self.check(await self.client.get("/health"), 200)
        self.assertEqual(await self.rows(), [])


class InferenceTimeoutTests(APICase):
    settings_overrides = {"inference_timeout_seconds": 0.001}

    async def test_timeout_on_every_inference_path_and_queue_cleanup(self):
        for path in ["/predict", "/batch-predict", "/predict-queued", "/predict-with-llm"]:
            payload = {"items": [FARM]} if path == "/batch-predict" else FARM
            self.check(await self.client.post(path, json=payload), 504)
        await asyncio.wait_for(self.app.state.batches.queue.join(), timeout=1)
        self.assertFalse(self.app.state.batches.worker.done())
        self.assertEqual(await self.rows(), [])
