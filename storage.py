"""Временное хранилище: результаты существуют только в памяти процесса."""

from schemas import PredictionResponse

predictions: dict[str, PredictionResponse] = {}
