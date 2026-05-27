# Project Source Code

This folder contains the **core project code** used for:

- data preparation;
- model training;
- inference (getting predictions);
- running services (API).

---

## `mrh` Module Structure

```
src/mrh/
├── __init__.py              # Package initialization
├── config.py                # Environment variables handling
├── data.py                  # Data loading, preparation and processing
├── models.py                # Models: training, recommendations, save/load
├── utils.py                 # Helper functions (logging, JSON, dictionaries)
└── api/                     # FastAPI application
    ├── __init__.py
    ├── dependencies.py      # Dependencies: model loading, caching
    ├── main.py              # FastAPI app creation (lifespan, CORS, routers)
    ├── schemas.py           # Pydantic models for request/response validation
    └── endpoints/           # API endpoints
        ├── __init__.py
        ├── health.py        # /health — service health check
        └── predict.py       # /predict — get recommendations
```

---

## Modules and Their Purpose

### 1. `config.py` — Configuration

**Purpose:** Load environment variables and validate required parameters.

**What it does:**
- Loads `.env` file (if it exists)
- Provides constants: `APP_HOST`, `APP_PORT`, `LOG_LEVEL`, `CB_THRESHOLD`, `CF_THRESHOLD`, `CONFIGS_DIR`

**Usage example:**
```python
from mrh.config import APP_HOST, APP_PORT, LOG_LEVEL
```

---

### 2. `data.py` — Data Handling

**Purpose:** Load, preprocess and save data.

**Main functions:**
| Function | Description |
|----------|-------------|
| `download_data(url, path)` | Download file from URL |
| `extract_archive(path_from, directory, expected_path)` | Extract ZIP with path traversal protection |
| `preprocess_tags(tags_string)` | Clean tags (lowercase, remove special characters) |
| `feature_preparation_cb(movies, tags, ratings)` | Prepare text features for movies |
| `get_ratings_by_threshold(ratings, threshold)` | Filter ratings by threshold |
| `data_preparation_cf(ratings, K1, B)` | Create BM25-weighted user-item matrix |
| `save_data(path, data)` | Save data (CSV for tables, NPZ for matrices) |
| `load_data(path)` | Load data with automatic format detection |

---

### 3. `models.py` — Models

**Purpose:** Build, train and use models.

**Main components:**

#### Content-based Model
| Function | Description |
|---------|-------------|
| `build_cb_pipeline(parameters)` | Create pipeline (TF-IDF + TruncatedSVD) |
| `train_cb_pipeline(pipeline, data, column)` | Train pipeline on text data |
| `apply_fitted_cb_pipeline(pipeline, data, column)` | Apply trained pipeline |
| `recommend_by_movieId(movieIds, movie_features, ...)` | Find similar movies via cosine similarity |

#### Collaborative Filtering Model (ALS)
| Function | Description |
|---------|-------------|
| `build_als_estimator(parameters)` | Create ALS model |
| `train_als_estimator(als_estimator, data)` | Train ALS on sparse matrix |
| `recommend_by_userId(userIds, als_model, ...)` | Get personalized recommendations |

#### Save/Load
| Function | Description |
|---------|-------------|
| `save_model(model, path)` | Save model via joblib |
| `load_model(path)` | Load model from file |

---

### 4. `utils.py` — Helper Functions

**Purpose:** Utilities used throughout the project.

| Function | Description |
|----------|-------------|
| `setup_logging(level, log_dir, log_filename)` | Configure logging (console + file) |
| `load_json(path)` | Load JSON file |
| `save_json(data, path)` | Save data to JSON |
| `inverse_dict(dictionary)` | Invert dictionary (keys ↔ values) |

---

### 5. `api/` — FastAPI Application

#### `api/dependencies.py` — Dependencies
| Function | Description |
|----------|-------------|
| `is_models_ready()` | Check if models are loaded in cache |
| `get_config_paths()` | Load `paths.json` (cached) |
| `load_cb_model()` | Load CB model (cached) |
| `load_cf_model()` | Load CF model (cached) |
| `get_cb_model_dep()` | Dependency for endpoints |
| `get_cf_model_dep()` | Dependency for endpoints |

#### `api/schemas.py` — Pydantic Models
| Model | Description |
|--------|-------------|
| `PredictRequest` | Request validation: `movieIds`/`userIds`, `k` |
| `PredictResponse` | Response structure with recommendations |
| `HealthResponse` | `/health` response structure |
| `RecommendationItem` | Single recommendation (movieId, score) |

#### `api/endpoints/health.py` — `/health` Endpoint
- `GET /health` — check service status and model loading

#### `api/endpoints/predict.py` — `/predict` Endpoint
- `POST /predict` — get recommendations by `movieIds` or `userIds`

#### `api/main.py` — FastAPI Application
| Function | Description |
|----------|-------------|
| `lifespan(app)` | Lifecycle management (load/clear models) |
| `create_app()` | Create and configure FastAPI application |

---

## How to Use Modules

### Import in Code

```python
# Data imports
from mrh.data import load_data, save_data, preprocess_tags

# Model imports
from mrh.models import build_als_estimator, recommend_by_movieId

# Utility imports
from mrh.utils import setup_logging, load_json, save_json, inverse_dict

# API imports
from mrh.api.main import create_app
from mrh.api.schemas import PredictRequest, PredictResponse
```

---

## Security and Error Handling

The modules implement:

- **Path traversal protection** — in `extract_archive()`
- **Type hints** — full type annotations
- **Input validation** — type and boundary checks
- **Error handling** — specific exceptions with clear messages
- **Logging** — informative messages at all stages

---

## Testing

Tests in `tests/` are used to test the modules:

```bash
# Test data.py
pytest tests/test_data.py -v

# Test models.py
pytest tests/test_models.py -v

# Test utils.py
pytest tests/test_utils.py -v

# Test API
pytest tests/test_api/ -v
```
