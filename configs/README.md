# Configuration Files

## Configuration Files

| File | Format | Purpose |
|------|--------|---------|
| `paths.json` | JSON | Paths to project directories and files |
| `models.json` | JSON | Model hyperparameters (SVD, ALS) |
| `.env.example` | ENV | Environment variables template |

## File Contents

**`paths.json`** — path configuration:
```json
{
    "data_dir": "data",
    "data_raw_dir": "data/raw",
    "data_processed_dir": "data/processed",
    "data_features_dir": "data/features",
    "artifacts_dir": "artifacts",
    "models_dir": "models",
    "notebooks_artifacts_dir": "notebooks_artifacts",
    "data_url": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "ml_32m_zip": "ml_32m.zip",
    "ml_32m": "ml-32m",
    "ratings_path": "ratings.csv",
    "movies_path": "movies.csv",
    "tags_path": "tags.csv",
    "cb_pipeline": "cb_pipeline.joblib",
    "als_model": "als_model.joblib",
    "cb_features": "cb_features.csv",
    "cf_user_item_matrix": "cf_user_item_matrix.npz"
}
```

**`models.json`** — model hyperparameters:
```json
{
    "cb": {
        "svd": {
            "n_components": 100,
            "random_state": 42
        }
    },
    "cf": {
        "als": {
            "factors": 100,
            "regularization": 0.1,
            "iterations": 50,
            "alpha": 1.0,
            "random_state": 42
        }
    }
}
```

**`.env.example`** — environment variables template:
```bash
APP_HOST=0.0.0.0
APP_PORT=8000
LOG_LEVEL=INFO
CONFIGS_DIR=./configs/
CB_FEATURE_COLUMN=genres_decade_tags
CB_THRESHOLD=3.0
CF_THRESHOLD=4.0
PYTHONUNBUFFERED=1
OPENBLAS_NUM_THREADS=1
```

## How to Use

1. **To run the project:**
```bash
# Copy the template and fill if necessary
cp configs/.env.example .env
```

2. **To change hyperparameters:**
   - Edit `configs/models.json`
   - Delete (if they exist) the files:
     - `artifacts/models/cb_pipeline.joblib`;
     - `artifacts/models/als_model.joblib`;
     - `data/features/cb_features.csv`.
  
```bash
rm -f artifacts/models/cb_pipeline.joblib
rm -f artifacts/models/als_model.joblib
rm -f data/features/cb_features.csv

# PowerShell
ri data/features/cb_features.csv 
ri artifacts/models/als_model.joblib
ri data/features/cb_features.csv
```

   - Retrain the models: `python scripts/train_models.py` (if there's no data, first run `python scripts/load_and_prepare_data.py`).
 ```bash
python scripts/load_and_prepare_data.py
python scripts/train_models.py
```

3. **To change paths:**
   - Edit `configs/paths.json`

## Loading Configuration in Code

```python
# Example from mrh/config.py
from dotenv import load_dotenv
load_dotenv(Path.cwd() / ".env")

APP_HOST = os.environ.get('APP_HOST')
APP_PORT = int(os.environ.get('APP_PORT'))
```

```python
# Example from mrh/api/dependencies.py
paths = load_json(root_path / CONFIGS_DIR / 'paths.json')
models_config = load_json(root_path / CONFIGS_DIR / 'models.json')
```

## What Can Be Configured Without Changing Code

| What is configured | File | Parameter |
|------------------|------|-----------|
| Server host and port | `.env` | `APP_HOST`, `APP_PORT` |
| Logging level | `.env` | `LOG_LEVEL` |
| Rating thresholds | `.env` | `CB_THRESHOLD`, `CF_THRESHOLD` |
| Text feature column | `.env` | `CB_FEATURE_COLUMN` |
| Data paths | `paths.json` | all paths |
| Embedding dimension | `models.json` | `cf.als.factors` |
| ALS regularization strength | `models.json` | `cf.als.regularization` |
| Number of SVD components | `models.json` | `cb.svd.n_components` |
| Number of ALS iterations | `models.json` | `cf.als.iterations` |

---