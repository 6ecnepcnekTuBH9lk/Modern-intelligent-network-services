import logging
import math
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import HTTPException

from .config import Settings

logger = logging.getLogger(__name__)
PREDICTION_COLUMNS = "request_id, farm_id, risk_score, risk_level, recommendation, model_version"
INSERT_PREDICTION = f"INSERT INTO predictions ({PREDICTION_COLUMNS}, source) VALUES (?, ?, ?, ?, ?, ?, ?)"


class Database:
    def __init__(self, settings: Settings):
        self.settings = settings

    @asynccontextmanager
    async def connect(self):
        try:
            async with aiosqlite.connect(
                str(self.settings.database_path), timeout=self.settings.db_busy_timeout_ms / 1000,
            ) as db:
                db.row_factory = aiosqlite.Row
                await db.execute(f"PRAGMA busy_timeout={self.settings.db_busy_timeout_ms}")
                await db.execute("PRAGMA synchronous=NORMAL")
                yield db
        except aiosqlite.Error as exc:
            logger.exception("Prediction storage error | path=%s", self.settings.database_path)
            raise HTTPException(503, "Prediction storage is unavailable") from exc

    async def initialize(self) -> None:
        async with self.connect() as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    request_id TEXT NOT NULL,
                    farm_id TEXT NOT NULL,
                    risk_score REAL NOT NULL,
                    risk_level TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    source TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions(request_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_predictions_risk_level ON predictions(risk_level)")
            await db.commit()
        logger.info("SQLite initialized | path=%s", self.settings.database_path)

    @staticmethod
    def row(result: dict, source: str) -> tuple:
        return tuple(result[key] for key in (
            "request_id", "farm_id", "risk_score", "risk_level", "recommendation", "model_version",
        )) + (source,)

    async def save_prediction(self, result: dict, source: str) -> None:
        async with self.connect() as db:
            await db.execute(INSERT_PREDICTION, self.row(result, source))
            await db.commit()
        logger.info("Prediction saved | request_id=%s | source=%s", result["request_id"], source)

    async def save_predictions_batch(self, request_id: str, items: list[dict]) -> None:
        rows = [self.row({**item, "request_id": request_id}, "batch") for item in items]
        async with self.connect() as db:
            await db.executemany(INSERT_PREDICTION, rows)
            await db.commit()
        logger.info("Batch saved | request_id=%s | batch_size=%s", request_id, len(items))

    async def get_predictions(self, limit: int, risk_level: str | None) -> list[dict]:
        sql = f"SELECT {PREDICTION_COLUMNS} FROM predictions"
        params = []
        if risk_level is not None:
            sql += " WHERE risk_level = ?"
            params.append(risk_level)
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        async with self.connect() as db:
            async with db.execute(sql, params) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def get_by_request_id(self, request_id: str) -> list[dict]:
        async with self.connect() as db:
            async with db.execute(
                f"SELECT {PREDICTION_COLUMNS} FROM predictions WHERE request_id = ? ORDER BY id",
                (request_id,),
            ) as cursor:
                return [dict(row) for row in await cursor.fetchall()]

    async def get_weekday_page(self, weekday_number: str, page: int, per_page: int) -> tuple[list[dict], int, int]:
        async with self.connect() as db:
            # COUNT и выборка страницы должны видеть один снимок данных.
            await db.execute("BEGIN")
            async with db.execute(
                "SELECT COUNT(*) FROM predictions WHERE strftime('%w', created_at) = ?",
                (weekday_number,),
            ) as cursor:
                total_items = (await cursor.fetchone())[0]
            async with db.execute(
                f"SELECT {PREDICTION_COLUMNS} FROM predictions "
                "WHERE strftime('%w', created_at) = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (weekday_number, per_page, (page - 1) * per_page),
            ) as cursor:
                data = [dict(row) for row in await cursor.fetchall()]
        return data, total_items, math.ceil(total_items / per_page)

    async def seed_weekdays(self, rows: list[tuple]) -> None:
        async with self.connect() as db:
            await db.executemany(
                f"INSERT INTO predictions ({PREDICTION_COLUMNS}, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', ?))",
                rows,
            )
            await db.commit()
