# Project Data

## Data Source

**MovieLens 32M** — open dataset collected by GroupLens Research (University of Minnesota).

- **Link:** https://grouplens.org/datasets/movielens/32m/
- **Size:** 239 MB (compressed), 911 MB (uncompressed)
- **License:** Free for academic use

---

## Data Structure

### `data/raw/` — raw data (after extraction)

| File | Size (uncompressed) | Description | Used |
|------|---------------------|-------------|------|
| `ratings.csv` | 850 MB | 32 million ratings (userId, movieId, rating, timestamp) | Yes |
| `movies.csv` | 4 MB | 87 thousand movies (movieId, title, genres) | Yes |
| `tags.csv` | 71 MB | 2 million tags (userId, movieId, tag, timestamp) | Yes |
| `links.csv` | 2 MB | Links to IMDb/TMDB | No |
| `README.md` | — | Dataset information | No |
| `checksums.txt` | — | File checksums | No |

### `data/processed/` — prepared data for models

| File | Description | Size |
|------|-------------|------|
| `movies_processed.csv` | Movies with features (year, genres, decade, tags) | ~80 MB |
| `cb_movieId_to_idx.json` | movieId → index mapping for CB model | ~1.5 MB |
| `cf_movieId_to_idx.json` | movieId → index mapping for CF model | ~1 MB |
| `cf_userId_to_idx.json` | userId → index mapping for CF model | ~4 MB |
| `cf_user_item_matrix.npz` | Sparse user-item matrix (BM25 weighted) | ~150 MB |
| `neg_cf_movieIds.json` | Negative movies to exclude from recommendations | ~250 MB |

### `data/features/` — inference features

| File | Description | Size |
|------|-------------|------|
| `cb_features.csv` | Movie feature matrix (TF-IDF + SVD, 100 components) | ~150 MB |

---

## How to Obtain the Data

### Method 1: Automatic Download (recommended)

```bash
python scripts/load_and_prepare_data.py
```

**What the script does:**
1. Downloads `ml-32m.zip` from the official website to `data/raw/`
2. Extracts the archive
3. Performs preprocessing and filtering by thresholds
4. Saves processed data to `data/processed/`

### Method 2: Manual Download

1. Go to https://grouplens.org/datasets/movielens/32m/
2. Download `ml-32m.zip`
3. Extract to `data/raw/ml-32m/`
4. Run `python scripts/load_and_prepare_data.py` for processing
   
To obtain the `cb_features.csv` features, you also need to run `python scripts/train_models.py`.

### Method 3: Use Ready-made Artifacts from Hugging Face

```bash
python scripts/download_models.py
```
Downloads only processed data (`data/processed/`, `data/features/`) and models, without raw CSVs.

---

## Directory Structure After Preparation

```
data/
├── raw/                           # raw data (after extraction)
│   └── ml-32m/
│       ├── ratings.csv            # 32 million ratings
│       ├── movies.csv             # 87 thousand movies
│       ├── tags.csv               # 2 million tags
│       ├── links.csv              # (not used)
│       ├── README.md              # (not used)
│       └── checksums.txt          # (not used)
├── processed/                     # prepared data (for models)
│   ├── movies_processed.csv       # movie features
│   ├── cb_movieId_to_idx.json     # CB mapping
│   ├── cf_movieId_to_idx.json     # CF movie mapping
│   ├── cf_userId_to_idx.json      # CF user mapping
│   ├── cf_user_item_matrix.npz    # BM25-weighted matrix
│   └── neg_cf_movieIds.json       # negative movies
└── features/                      # inference features
    └── cb_features.csv            # TF-IDF + SVD features
```

---

## Why Data Is Not Stored in the Repository

| Reason | Explanation |
|---------|-------------|
| **Size** | Raw data ~1 GB (uncompressed) |
| **GitHub Limits** | GitHub is not designed for storing large files |
| **Reproducibility** | Download scripts ensure experiment cleanliness |

---

## Reproducibility

### For full experiment reproduction:

```bash
# 1. Obtain data
python scripts/load_and_prepare_data.py

# 2. Train models
python scripts/train_models.py

# 3. Run notebooks (optional)
jupyter lab notebooks/
```

### For API-only (without retraining):

```bash
# 1. Download pre-trained models and data
python scripts/download_models.py

# 2. Run the service
python scripts/run_api.py
```

---

## Notes

- For API operation, only `data/processed/` and `data/features/` are needed (can be obtained via `download_models.py`).
- Full training requires all raw data (~1 GB free space).
