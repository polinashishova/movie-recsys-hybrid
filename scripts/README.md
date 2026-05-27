# Project Scripts

This folder contains scripts for automating the main tasks of the project: data loading, model training, downloading artifacts, and running the API service.

---

## Scripts Overview

- `load_and_prepare_data.py` - download and prepare MovieLens 32M data;
- `train_models.py` - train and save models;
- `download_models.py` - download pre-trained models from Hugging Face Hub;
- `run_api.py` - run the FastAPI application.

---


## Installing Dependencies

```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# For running the API only (minimal)
pip install -e ".[api]"

# For full cycle (training + API)
pip install -e ".[dev,api]"
```

## Typical Scenarios

### Scenario 1: Run service only (minimal)

```bash
# 1. Download pre-trained models and data
python scripts/download_models.py

# 2. Run the API
python scripts/run_api.py
```

#### Scenario 2: Full cycle (training + run)

```bash
# 1. Download and prepare data
python scripts/load_and_prepare_data.py

# 2. Train models
python scripts/train_models.py

# 3. Run the API
python scripts/run_api.py
```

#### Scenario 3: Experiments only (notebooks)

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run Jupyter
jupyter lab notebooks/
```

---

## Detailed Script Descriptions

### 1. `load_and_prepare_data.py`

**Purpose:** Download the MovieLens 32M dataset and prepare data for training.

**What it does:**
1. Downloads `ml-32m.zip` from the official website.
2. Extracts the archive.
3. Filters ratings by thresholds (CB_THRESHOLD, CF_THRESHOLD from `.env`).
4. Prepares text features for movies (genres, year, decade, tags).
5. Creates ID → index mappings.
6. Saves processed data to `data/processed/`.

**Usage:**

```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

python scripts/load_and_prepare_data.py
```

**Output files:**
- `data/processed/movies_processed.csv` — movie features;
- `data/processed/*_to_idx.json` — ID mappings;
- `data/processed/cf_user_item_matrix.npz` — user-item matrix.

**Configuration via `.env`:**
```bash
CB_THRESHOLD=3.0    # threshold for content-based model
CF_THRESHOLD=4.0    # threshold for collaborative filtering
```
**Note:** the script skips existing files.

---

### 2. `train_models.py`

**Purpose:** Train and save content-based and collaborative filtering models.

**What it does:**
1. Loads processed data.
2. Trains the CB pipeline (TF-IDF + TruncatedSVD).
3. Trains the ALS model (Alternating Least Squares).
4. Saves models to `artifacts/models/`.

**Usage:**
```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

python scripts/train_models.py
```

**Output files:**
- `artifacts/models/cb_pipeline.joblib` — trained CB pipeline;
- `artifacts/models/als_model.joblib` — trained ALS model;
- `data/features/cb_features.csv` — movie features.

**Model configuration:** `configs/models.json`
```json
{
    "cb": {
        "svd": { "n_components": 100, "random_state": 42 }
    },
    "cf": {
        "als": { "factors": 100, "regularization": 0.1, "iterations": 50, "alpha": 1.0, "random_state": 42 }
    }
}
```
**Note:** the script skips existing files.

---

### 3. `download_models.py`

**Purpose:** Download pre-trained models and processed data from Hugging Face Hub.

**What it does:**
1. Checks if files exist locally.
2. If missing, downloads from the `shajeless/movie-recsys-hybrid` repository.
3. Distributes files to the appropriate directories.

**Usage:**
```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

python scripts/download_models.py
```

**What gets downloaded:**
| From | To |
|------|-----|
| `models/cb_pipeline.joblib` | `artifacts/models/` |
| `models/als_model.joblib` | `artifacts/models/` |
| `data/processed/*` | `data/processed/` |
| `data/features/*` | `data/features/` |

**Note:** the script skips existing files.

---

### 4. `run_api.py`

**Purpose:** Run the FastAPI server for the recommendation service.

**What it does:**
1. Loads models into memory (on startup).
2. Starts the Uvicorn server.
3. Handles requests to `/predict`, `/health`, `/docs` endpoints.

**Usage:**
```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Basic launch
python scripts/run_api.py

# With parameters
python scripts/run_api.py --host 0.0.0.0 --port 8000 --workers 2 --reload
```

**Command line parameters:**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--host` | `0.0.0.0` | Server host |
| `--port` | `8000` | Server port |
| `--workers` | `1` | Number of workers |
| `--reload` | `False` | Auto-reload on code changes (development only) |

**Verification:** see README.md section Request Examples

---

## Environment Variables

The scripts use variables from the `.env` file:

```bash
# .env
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
CONFIGS_DIR=./configs/

# Model parameters
CB_FEATURE_COLUMN=genres_decade_tags
CB_THRESHOLD=3.0
CF_THRESHOLD=4.0

# BLAS optimizations
PYTHONUNBUFFERED=1
OPENBLAS_NUM_THREADS=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
```

---

## Notes

- Scripts assume they are run from the **root directory**.
- GPU training is not required for ALS (CPU version from `implicit.cpu`).
