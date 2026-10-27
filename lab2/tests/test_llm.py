import json

import httpx

from lab2.tests.support import APICase, FARM, HIGH_FARM


class LLMTests(APICase):
    def transport(self, handler):
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(client.aclose)
        self.app.state.llm.client = client

    @staticmethod
    def completion(answer="Тестовое объяснение"):
        return {"model": "test-model", "choices": [{"message": {
            "role": "assistant", "content": answer, "reasoning_content": "test reasoning",
        }, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}

    async def test_chat_success_openai_payload_and_no_authorization(self):
        captured = []
        def provider(request):
            captured.append(request)
            return httpx.Response(200, json=self.completion())
        self.transport(provider)
        result = self.check(await self.client.post("/llm/chat", json={"prompt": "Тестовый вопрос"},
            headers={"X-Request-ID": "llm-1"}), 200)
        self.assertEqual(result["request_id"], "llm-1")
        self.assertEqual(result["requested_model"], "auto")
        self.assertEqual(result["returned_model"], "test-model")
        self.assertEqual(result["answer"], "Тестовое объяснение")
        self.assertEqual(result["reasoning"], "test reasoning")
        self.assertGreaterEqual(result["latency_ms"], 0)
        self.assertEqual(str(captured[0].url), "https://vireonix.ai/v1/chat/completions")
        self.assertNotIn("Authorization", captured[0].headers)
        payload = json.loads(captured[0].content)
        self.assertEqual([message["role"] for message in payload["messages"]], ["system", "user"])
        self.assertEqual(payload["temperature"], 0.3)
        self.assertEqual(payload["max_tokens"], 300)

    async def test_provider_http_errors_are_502(self):
        for code in [400, 401, 429, 500, 503]:
            with self.subTest(code=code):
                self.transport(lambda request, code=code: httpx.Response(code, text="upstream failure"))
                body = self.check(await self.client.post("/llm/chat", json={"prompt": "test"}), 502)
                self.assertIn(str(code), body["detail"])

    async def test_invalid_llm_responses_are_502(self):
        responses = [httpx.Response(200, text="not json")]
        for data in [[], None, {}, {"choices": []}, {"choices": "bad"}, {"choices": [None]},
                     {"choices": [{"message": {}}]}, {"choices": [{"message": {"content": " "}}]},
                     {"choices": [{"message": {"content": {"bad": "shape"}}}]},
                     {**self.completion(), "usage": []}]:
            responses.append(httpx.Response(200, json=data))
        for response in responses:
            with self.subTest(body=response.text):
                self.transport(lambda request, response=response: response)
                self.check(await self.client.post("/llm/chat", json={"prompt": "test"}), 502)

    async def test_llm_timeout_is_504(self):
        def timeout(request):
            raise httpx.ReadTimeout("test timeout", request=request)
        self.transport(timeout)
        self.assertEqual(self.check(await self.client.post("/llm/chat", json={"prompt": "test"}), 504)["detail"], "LLM request timeout")

    async def test_llm_network_failure_is_503(self):
        def unavailable(request):
            raise httpx.ConnectError("test network error", request=request)
        self.transport(unavailable)
        self.check(await self.client.post("/llm/chat", json={"prompt": "test"}), 503)

    async def test_combined_response_keeps_scoring_and_excludes_input_features(self):
        captured = []
        def provider(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=self.completion("Тестовый текст; исходный риск не меняется."))
        self.transport(provider)
        result = self.check(await self.client.post("/predict-with-llm", json={**HIGH_FARM, "farm_id": "PRIVATE-FARM"},
            headers={"X-Request-ID": "combined-1"}), 200)
        self.assertEqual(result["prediction"]["risk_score"], 0.9)
        self.assertEqual(result["prediction"]["risk_level"], "high")
        self.assertEqual(result["llm"]["request_id"], result["prediction"]["request_id"])
        text = json.dumps(captured[0], ensure_ascii=False)
        for excluded in ["PRIVATE-FARM", "farm_id", "debt", "6500000", "payment_delay_days", "previous_defaults"]:
            self.assertNotIn(excluded, text)
        self.assertIn("0.9", text)
        self.assertEqual((await self.rows())[0]["source"], "single+llm")

    async def test_combined_failure_preserves_committed_prediction(self):
        self.transport(lambda request: httpx.Response(503))
        self.check(await self.client.post("/predict-with-llm", json=HIGH_FARM,
            headers={"X-Request-ID": "saved-before-llm"}), 502)
        result = self.check(await self.client.get("/predictions/saved-before-llm"), 200)
        self.assertEqual(result["risk_score"], 0.9)
        self.assertEqual(result["risk_level"], "high")

    async def test_llm_structural_validation_does_not_call_provider(self):
        def forbidden(request):
            raise AssertionError("Provider must not be called")
        self.transport(forbidden)
        for changes in [{"prompt": ""}, {"prompt": "x" * 5001}, {"model": ""},
                        {"temperature": -0.1}, {"temperature": 2.1}, {"max_tokens": 0}, {"max_tokens": 2001}]:
            self.check(await self.client.post("/llm/chat", json={"prompt": "test", **changes}), 422)
