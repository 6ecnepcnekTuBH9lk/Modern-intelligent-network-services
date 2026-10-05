# Agro Scoring API

REST API для оценки риска сельскохозяйственных предприятий на FastAPI.

## Структура проекта

```text
agro_api/
├── main.py
├── schemas.py
├── model.py
├── services.py
├── storage.py
├── requirements.txt
├── test_api.py
└── README.md
```

Назначение модулей:

- `main.py` — создание FastAPI-приложения, endpoints, middleware и логирование;
- `schemas.py` — Pydantic-модели запросов и ответов;
- `model.py` — расчет оценки риска и метаданные модели;
- `services.py` — бизнес-валидация, постобработка и работа с прогнозами;
- `storage.py` — временное хранение результатов в памяти;
- `test_api.py` — интеграционные тесты API;
- `requirements.txt` — зависимости проекта.

## Установка и запуск

Создание виртуального окружения:

```powershell
python -m venv .venv
```

Активация окружения:

```powershell
.\.venv\Scripts\Activate.ps1
```

Установка зависимостей:

```powershell
python -m pip install -r requirements.txt
```

Запуск приложения:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Если порт `8000` недоступен, можно использовать другой, например `8001`:

```powershell
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8001
```

После запуска доступны:

- Swagger UI: `http://127.0.0.1:8000/docs`;
- OpenAPI: `http://127.0.0.1:8000/openapi.json`.

При использовании другого порта его необходимо заменить в адресе.

## API

Поддерживаются следующие endpoints:

| Метод | Endpoint | Назначение |
|---|---|---|
| `GET` | `/health` | Проверка состояния API |
| `GET` | `/model-info` | Информация о модели |
| `POST` | `/predict` | Расчет риска хозяйства |
| `GET` | `/predictions/{request_id}` | Получение прогноза по идентификатору |
| `GET` | `/predictions` | Получение списка прогнозов |

## Входные данные

Пример запроса `POST /predict`:

```json
{
  "farm_id": "FARM-001",
  "region": "Krasnodar",
  "crop_type": "wheat",
  "area_ha": 2500,
  "temperature_avg": 24.3,
  "precipitation_mm": 320,
  "payment_delay_days": 45,
  "previous_defaults": 1,
  "debt": 6500000
}
```

Основные ограничения входных данных:

- `area_ha > 0`;
- `precipitation_mm >= 0`;
- `payment_delay_days >= 0`;
- `previous_defaults >= 0`;
- `debt >= 0`;
- `temperature_avg` находится в диапазоне от `-60` до `60`;
- `farm_id`, `region` и `crop_type` не могут быть пустыми.

Допустимые регионы:

- `Krasnodar`;
- `Rostov`;
- `Stavropol`.

## Расчет риска

Начальное значение оценки риска равно `0.1`.

| Условие | Изменение оценки |
|---|---:|
| `payment_delay_days > 30` | `+0.3` |
| `previous_defaults > 0` | `+0.3` |
| `debt > 5_000_000` | `+0.2` |
| `precipitation_mm < 100` | `+0.1` |

Итоговая оценка ограничивается значением `1.0`.

Категории риска:

- `risk_score < 0.3` — `low`;
- `0.3 <= risk_score < 0.7` — `medium`;
- `risk_score >= 0.7` — `high`.

Для каждой категории формируется рекомендация:

- `low` — стандартное рассмотрение;
- `medium` — дополнительная проверка;
- `high` — ручное рассмотрение.

## Пример ответа

```json
{
  "request_id": "2ea8766d-391c-410b-81a2-75b7a9da89b4",
  "farm_id": "FARM-001",
  "risk_score": 0.9,
  "risk_level": "high",
  "recommendation": "Высокий риск. Требуется ручное рассмотрение",
  "model_version": "1.0"
}
```

## Фильтрация прогнозов

Параметр `limit` задает максимальное количество возвращаемых результатов:

```text
GET /predictions?limit=2
```

Допустимый диапазон: от `1` до `100`.

Фильтр `risk_level` позволяет получать прогнозы определенной категории:

```text
GET /predictions?risk_level=high
```

Также параметры можно использовать совместно:

```text
GET /predictions?risk_level=medium&limit=5
```

## HTTP-коды

Основные ответы API:

- `200 OK` — успешный запрос;
- `400 Bad Request` — ошибка бизнес-валидации;
- `404 Not Found` — прогноз не найден;
- `422 Unprocessable Entity` — ошибка валидации входных данных;
- `503 Service Unavailable` — модель временно недоступна.

## Хранение данных

Прогнозы сохраняются во временном словаре в оперативной памяти. После перезапуска приложения сохраненные результаты удаляются.

## Тестирование

Для запуска интеграционных тестов:

```powershell
python test_api.py
```

Тесты проверяют основные endpoints, HTTP-коды, валидацию данных, фильтрацию, UUID, OpenAPI и состояние доступности модели.
