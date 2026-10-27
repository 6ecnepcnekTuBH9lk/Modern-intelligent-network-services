import logging
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import aiosqlite
import httpx

from lab2.config import BASE_DIR, Settings
from lab2.main import create_app

logging.getLogger("lab2").setLevel(logging.CRITICAL)
logging.getLogger("httpx").setLevel(logging.WARNING)
FARM = {
    "farm_id": "FARM-001", "region": "Krasnodar", "crop_type": "wheat", "area_ha": 2500,
    "temperature_avg": 24.3, "precipitation_mm": 320, "payment_delay_days": 15,
    "previous_defaults": 0, "debt": 1500000,
}
HIGH_FARM = {**FARM, "payment_delay_days": 45, "previous_defaults": 1, "debt": 6500000}


class APICase(unittest.IsolatedAsyncioTestCase):
    settings_overrides = {}

    async def asyncSetUp(self):
        temporary_root = BASE_DIR / ".test-data"
        temporary_root.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=temporary_root)
        self.addCleanup(self.temp.cleanup)
        self.db_path = Path(self.temp.name) / "test.db"
        self.settings = replace(Settings(), database_path=self.db_path, **self.settings_overrides)
        self.app = create_app(self.settings)
        self.lifecycle = self.app.router.lifespan_context(self.app)
        await self.lifecycle.__aenter__()
        self.addAsyncCleanup(self.lifecycle.__aexit__, None, None, None)
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app, raise_app_exceptions=False),
            base_url="http://testserver",
        )
        self.addAsyncCleanup(self.client.aclose)

    def check(self, response, status):
        self.assertEqual(response.status_code, status, response.text)
        self.assertTrue(response.headers["X-Request-ID"])
        self.assertGreaterEqual(float(response.headers["X-Process-Time"]), 0)
        if status >= 400:
            self.assertEqual(response.json()["request_id"], response.headers["X-Request-ID"])
        return response.json()

    async def rows(self):
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT * FROM predictions ORDER BY id") as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def sql(self, sql):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(sql)
            await db.commit()
