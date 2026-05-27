# Project Tests

This folder contains **tests** for verifying code correctness:

- unit tests for functions and classes from `src/`;
- integration tests for API endpoints;
- mock tests for isolated verification.

---

## Test Structure

```
tests/
├── __init__.py
├── test_utils.py               # Unit tests for utilities (inverse_dict, JSON, logging)
├── test_data.py                # Unit tests for data processing (preprocess_tags, filtering)
├── test_models.py              # Unit tests for models (recommendations, save/load)
└── test_api/                   # API integration tests
    ├── __init__.py
    ├── test_health.py          # Tests for /health endpoint
    └── test_predict.py         # Tests for /predict endpoint
```

---

## Test Types

| Type | Files | Description |
|------|-------|-------------|
| **Unit tests** | `test_utils.py`, `test_data.py`, `test_models.py` | Isolated function testing |
| **Integration tests** | `test_api/test_health.py`, `test_api/test_predict.py` | API endpoint testing |
| **Mock tests** | `test_models.py`, `test_api/test_predict.py` | Testing with dependency substitution |

**Total tests:** 55

---

## What Is Tested

### `test_utils.py` — Utilities

| Class/Function | What it tests |
|---------------|----------------|
| `TestInverseDict` | Dictionary inversion, duplicates, errors |
| `TestJsonOperations` | JSON save/load, error handling |
| `TestSetupLogging` | Log directory creation, logger return |

### `test_data.py` — Data

| Class/Function | What it tests |
|---------------|----------------|
| `TestPreprocessTags` | Tag cleaning (punctuation, case, NaN, None, Russian letters) |
| `TestGetRatingsByThreshold` | Rating filtering by threshold, edge cases |
| `TestSaveLoadData` | DataFrame/numpy array save/load, error handling |

### `test_models.py` — Models

| Class/Function | What it tests |
|---------------|----------------|
| `TestRecommendByMovieId` | Recommendations for 1 or N movies, k validation, errors |
| `TestRecommendByUserId` | User recommendations, movie exclusion |
| `TestSaveLoadModel` | Pipeline and ALS model save/load |
| `TestBuildCBPipeline` | Pipeline creation with different parameters |
| `TestTrainCBPipeline` | Training on Series and DataFrame, error handling |
| `TestBuildALSEstimator` | ALS creation with different parameters |

### `test_api/test_health.py` — Health Endpoint

| Test | What it tests |
|------|----------------|
| `test_health_endpoint_returns_200` | Response status 200 |
| `test_health_response_format` | JSON response structure |
| `test_root_endpoint` | Root endpoint `/` |

### `test_api/test_predict.py` — Predict Endpoint

| Test | What it tests |
|------|----------------|
| `test_predict_without_ids` | Error when both movieIds and userIds are missing |
| `test_predict_with_both_ids` | Error when both ID types are provided |
| `test_predict_with_empty_movieIds` | Error when movieIds list is empty |
| `test_predict_with_empty_userIds` | Error when userIds list is empty |
| `test_predict_with_negative_ids` | Error when IDs are negative |
| `test_predict_with_k_out_of_range` | Error when k is outside range (1-100) |
| `test_predict_response_format` | Response structure (if models are loaded) |

---

## Running Tests

### All tests
```bash
.venv\Scripts\activate      # Windows
# or
source .venv/bin/activate   # Linux/macOS

pip install -e ".[dev, api]"

pytest tests/ -v
```

### With coverage report
```bash
pytest tests/ --cov=mrh --cov-report=term
```

### Detailed report with missing lines
```bash
pytest tests/ --cov=mrh --cov-report=term-missing
```

### Unit tests only
```bash
pytest tests/test_utils.py tests/test_data.py tests/test_models.py -v
```

### API tests only
```bash
pytest tests/test_api/ -v
```

### Specific test class
```bash
pytest tests/test_utils.py::TestInverseDict -v
```

### Specific test
```bash
pytest tests/test_data.py::TestPreprocessTags::test_normal_string -v
```

---

## Test Results

```
============================= 55 passed, 1 warning in 16.79s =============================

Name                           Stmts   Miss  Cover
-------------------------------------------------------
src\mrh\api\dependencies.py        78     10    87%
src\mrh\api\endpoints\health.py    16      1    94%
src\mrh\api\endpoints\predict.py   42     13    69%
src\mrh\api\main.py                36     17    53%
src\mrh\api\schemas.py             34      0   100%
src\mrh\config.py                  15      0   100%
src\mrh\data.py                   211    137    35%
src\mrh\models.py                 270    133    51%
src\mrh\utils.py                   65     11    83%
-------------------------------------------------------
TOTAL                             767    322    58%
```

---

## Known Warnings

When running tests, the following warning may appear:

```
RuntimeWarning: OpenBLAS is configured to use 12 threads. 
It is highly recommended to disable its internal threadpool...
```

**This is normal for testing.** In Docker, the variables are already set to `OPENBLAS_NUM_THREADS=1`.
