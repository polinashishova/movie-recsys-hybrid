# Movie Hybrid RecSys API

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)
[![Tests](https://img.shields.io/badge/tests-55%20passed-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-58%25-yellow.svg)]()
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A hybrid movie recommendation API service combining **content-based filtering** (using genres, tags, and release year) and **collaborative filtering** (using user ratings). The service handles both cold-start scenarios (recommendations by movie ID) and personalized recommendations (by user ID).

## Features

- **Cold-start recommendations** – Get similar movies based on content features (TF-IDF + SVD)
- **Personalized recommendations** – ALS collaborative filtering with BM25 weighting
- **REST API** – FastAPI with automatic Swagger documentation
- **Docker support** – Multi-stage build for production deployment
- **Production-ready** – Model caching, health checks, comprehensive logging

## Quick Start

### Docker (recommended)

```bash
# Clone the repository
git clone https://github.com/polinashishova/movie-recsys-hybrid.git
cd movie-recsys-hybrid

# Build and run
docker build -t movie-recsys-hybrid .
docker run -d -p 8000:8000 --name movie-recsys-hybrid movie-recsys-hybrid
```

### Local installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install with API dependencies
pip install -e ".[api]"

# Download pre-trained models
python scripts/download_models.py

# Start the server
python scripts/run_api.py
```

## API Reference

The service runs on `http://localhost:8000` by default.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service information |
| `/health` | GET | Health check |
| `/predict` | POST | Get recommendations |
| `/docs` | GET | Swagger UI documentation |
| `/redoc` | GET | ReDoc documentation |

### Request Examples

**Cold-start recommendations (by movie ID)**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"movieIds": [1225, 329], "k": 5}'
```

**Personalized recommendations (by user ID)**
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"userIds": [214, 154198], "k": 10}'
```

### Response Format

```json
{
  "recommendations": {
    "1225": [
      {"movieId": 1234, "score": 0.95},
      {"movieId": 5678, "score": 0.89}
    ]
  },
  "model_used": "content_based",
  "timestamp": "2025-05-26T10:30:00Z",
  "request_id": "abc12345"
}
```

## Models

### Content-based Model
- **Features**: TF-IDF on genres + decade + user tags
- **Dimensionality reduction**: TruncatedSVD (100 components)
- **Similarity**: Cosine similarity
- **Dataset**: Non-negative ratings (3.0–5.0)

### Collaborative Filtering Model
- **Algorithm**: ALS (Alternating Least Squares)
- **Library**: `implicit` with CPU backend
- **Matrix weighting**: BM25 (K1=100, B=0.8)
- **Hyperparameters**: factors=100, reg=0.1, iter=50, alpha=1.0
- **Dataset**: Positive ratings (4.0–5.0), binarized

### Model Performance (k=10)

| Model | precision@10 | recall@10 | hit_rate@10 | ndcg@10 |
|-------|--------------|-----------|-------------|---------|
| Random Baseline | 0.05% | 0.01% | 0.46% | 0.04% |
| Content-based | 1.50% | 1.03% | 12.26% | 1.46% |
| **Collaborative (ALS)** | **8.86%** | **9.09%** | **51.21%** | **11.33%** |

## Testing

```bash
pip install -e ".[dev]"
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=mrh --cov-report=term

# Unit tests only
pytest tests/test_utils.py tests/test_data.py tests/test_models.py -v

# API tests only
pytest tests/test_api/ -v
```

**Results:** 55 passed, 58% coverage

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.12 |
| API Framework | FastAPI + Uvicorn |
| ML Libraries | scikit-learn, implicit, numpy, scipy, pandas |
| Validation | Pydantic v2 |
| Container | Docker (multi-stage build) |
| Testing | pytest, pytest-cov, pytest-mock |
| Dependency Management | pyproject.toml (PEP 621) |
| Model Storage | Hugging Face Hub + joblib |

## Project Structure

```
movie-recsys-hybrid/
├── src/mrh/              # Main source code
│   ├── api/              # FastAPI endpoints, schemas, dependencies
│   ├── data.py           # Data loading and preprocessing
│   ├── models.py         # Model training and inference
│   ├── config.py         # Environment configuration
│   └── utils.py          # Helper functions
├── scripts/              # Utility scripts
│   ├── load_and_prepare_data.py
│   ├── train_models.py
│   ├── download_models.py
│   └── run_api.py
├── configs/              # Configuration files
│   ├── paths.json
│   ├── models.json
│   └── .env.example
├── tests/                # Unit and integration tests (55 tests)
├── notebooks/            # EDA and experimentation
├── data/                 # Data directory (not in Git)
├── artifacts/            # Models and logs (not in Git)
├── dockerfile
├── pyproject.toml
├── LICENSE               # MIT License
└── README.md
```

## Configuration

Copy the environment template and adjust as needed:

```bash
cp configs/.env.example .env
```

Key configuration options:

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_HOST` | 0.0.0.0 | Server host |
| `APP_PORT` | 8000 | Server port |
| `LOG_LEVEL` | INFO | Logging level |
| `CB_THRESHOLD` | 3.0 | Rating threshold for content-based |
| `CF_THRESHOLD` | 4.0 | Rating threshold for collaborative filtering |
| `CB_FEATURE_COLUMN` | genres_decade_tags | Text feature column for CB |

Model hyperparameters can be tuned in `configs/models.json`.

Paths can be changed in `config/paths.json`.

## Development

### Install with dev dependencies

```bash
pip install -e ".[dev,api]"
```

### Run Jupyter for experimentation

```bash
jupyter lab notebooks/
```

### Train models from scratch

```bash
python scripts/load_and_prepare_data.py
python scripts/train_models.py
```

## Data

The project uses the **MovieLens 32M** dataset from GroupLens Research:

- **Size**: 32 million ratings, 87,585 movies, 200,948 users
- **Source**: [MovieLens 32M](https://grouplens.org/datasets/movielens/32m/)
- **License**: Free for academic use

Data is automatically downloaded and preprocessed when running `scripts/load_and_prepare_data.py`. Pre-trained models and processed data are also available on [Hugging Face Hub](https://huggingface.co/shajeless/movie-recsys-hybrid).

## Limitations & Future Work

**Current limitations:**
- Offline training only (no real-time updates)
- Limited to MovieLens dataset format
- No rate limiting or request caching
- No recommendation explanations
- No horizontal scaling

**Planned improvements:**
- Online embedding updates
- Neural network-based approaches (NeuMF, Two-Tower)
- Recommendation explanations (LIME, SHAP)
- Redis caching for popular queries
- Rate limiting for production
- Support for new movie catalogs

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.


## Author

**Polina Shishova**  
- Telegram: [@waste_magpie](https://t.me/waste_magpie)
- Email: timewastingnonsense@gmail.com
- GitHub: [polinashishova](https://github.com/polinashishova)
