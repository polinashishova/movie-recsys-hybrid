# Project Notebooks

## Overview

- `01_eda.ipynb` - exploratory data analysis of MovieLens 32M;
- `02_preprocessing_and_experiments.ipynb` - data preparation, model experiments, hyperparameter tuning.

---

### 1. `01_eda.ipynb` — Exploratory Data Analysis

**Objective:** Study the dataset structure, identify features and issues.

**What it does:**
- Downloads and extracts the dataset.
- Analyzes `ratings.csv`, `movies.csv`, `tags.csv` files.
- Identifies missing values and duplicates.
- Visualizes movie distribution by year.

**Key findings:**
- 32 million ratings, 200,948 users, 87,585 movies.
- 17 missing values in tags (replaced with empty strings).
- 615 movies missing release year.
- Rating range: 0.5 - 5.0.
- Trend: number of movies increases with each decade.

**Saved artifacts:**
- `artifacts/notebooks_artifacts/hist_movie_year_plot.png` — histogram of movie distribution by year

---

### 2. `02_preprocessing_and_experiments.ipynb` — Data Preparation and Experiments

**Objective:** Prepare data for models, conduct experiments, select best models.

**Main stages:**

#### 2.1. Feature Preprocessing
- Extract release year from movie title.
- Generate decades (1990s, 2000s, etc.).
- Clean tags (remove special characters, lowercase).
- Create combined text features:
  - `genres_decade` — genres + decade;
  - `genres_decade_tags` — genres + decade + tags.

#### 2.2. Dataset Preparation
Three dataset variants for content-based experiments:
- All ratings (0.5-5.0).
- Non-negative (3.0-5.0).
- Positive (4.0-5.0).

#### 2.3. Content-based Model Experiments
Comparison of different vectorization methods:
- One-hot genres;
- TF-IDF on genres + decade;
- TF-IDF + SVD (95% variance);
- TF-IDF + SVD (100 components).

#### 2.4. Collaborative Filtering Model Experiments
- ALS (Alternating Least Squares) from the `implicit` library.
- BM25 weighting of user-item matrix.
- Hyperparameter tuning (factors, regularization, iterations, alpha).

**Saved artifacts:**
- `data/processed/movies_processed.csv` — processed movie features;
- `data/processed/*_to_idx.json` — ID → index mappings;
- `data/processed/cf_user_item_matrix.npz` — weighted user-item matrix;
- `data/features/cb_features.csv` — movie features;
- `artifacts/notebooks_artifacts/baseline_metrics.json`;
- `artifacts/notebooks_artifacts/content_based_metrics.json`;
- `artifacts/notebooks_artifacts/collaborative_filtering_metrics.json`;
- `artifacts/notebooks_artifacts/best_cb_model_config.json`;
- `artifacts/notebooks_artifacts/best_cf_model_config.json`.

**Final Models:**

**Content-based**: TF-IDF + TruncatedSVD (n_components=100), trained on non-negative ratings (3.0-5.0).
**Collaborative**: ALS (factors=100, reg=0.1, iter=50, alpha=1.0), BM25 weight, trained on positive ratings (4.0-5.0).

---

## Running Notebooks

```bash
# Activate the environment:
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dev dependencies
pip install -e ".[dev]"

# Run Jupyter
jupyter lab
# or
jupyter notebook
```

**Execution order:**
1. First `01_eda.ipynb`.
2. Then `02_preprocessing_and_experiments.ipynb`.

**Execution time:**
- `01_eda.ipynb`: ~2-3 minutes
- `02_preprocessing_and_experiments.ipynb`: ~1-2 hours (mostly due to experiments)

---

## Code Migration to `src/`

Logic from notebooks that is used in the service has been extracted into modules.
