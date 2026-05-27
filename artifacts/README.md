# Project Artifacts

## Saved Models
| File | Size | Description |
|------|------|-------------|
| `models/cb_pipeline.joblib` | ~40 MB | TF-IDF + TruncatedSVD (100 components) pipeline for content-based recommendations |
| `models/als_model.joblib` | ~100 MB | Trained ALS model (100 factors) |

## Inference Data
| File | Description |
|------|-------------|
| `data/processed/cb_movieId_to_idx.json` | movieId → index mapping for CB model |
| `data/processed/cf_movieId_to_idx.json` | movieId → index mapping for CF model |
| `data/processed/cf_userId_to_idx.json` | userId → index mapping for CF model |
| `data/processed/cf_user_item_matrix.npz` | Sparse user-item matrix (BM25 weighted) |
| `data/features/cb_features.csv` | Movie features after SVD (100 components) |

## Experiment Results

Located in the `notebooks_artifacts/` folder.

| File | Description |
|------|-------------|
| `notebooks_artifacts/baseline_metrics.json` | Random baseline model metrics |
| `notebooks_artifacts/content_based_metrics.json` | Results of all CB experiments |
| `notebooks_artifacts/collaborative_filtering_metrics.json` | ALS hyperparameter results |
| `notebooks_artifacts/best_cb_model_config.json` | Best CB model configuration |
| `notebooks_artifacts/best_cf_model_config.json` | Best CF model configuration |

## Visualizations
| File | Description |
|------|-------------|
| `notebooks_artifacts/hist_movie_year_plot.png` | Histogram of movie distribution by year |

## Logging

When running scripts, logs are written to `artifacts/logs/logs.log` (the name can be changed). This file is not included in the repository.

---

## External Storage

Final models and processed data are uploaded to **Hugging Face Hub**:

- Repository: `shajeless/movie-recsys-hybrid`
- Link: https://huggingface.co/shajeless/movie-recsys-hybrid

To download the models, use the `scripts/download_models.py` script.

```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
# Download models and required data
python scripts/download_models.py
```

---

## Note

Artifacts (especially models and `.npz` files) are not stored in the Git repository due to their large size. 
To reproduce the results, use the download script or run data preparation and training via `scripts/load_and_prepare_data.py` and `scripts/train_models.py`.

```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate
# Load and prepare data
python scripts/load_and_prepare_data.py
# Train models
python scripts/train_models.py
```

You can also obtain trained models and inference data by running the notebooks.

```bash
pip install -e ".[dev]"
jupyter lab notebooks/
```