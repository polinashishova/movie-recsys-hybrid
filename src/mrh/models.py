"""src/mrh/models.py
Recommendation system models: building models (content-based for user cold-start 
and collaborative filtering for personalized recommendations); training and applying models;
obtaining recommendations; saving and loading models.
"""

import logging
from pathlib import Path
from typing import Any, Hashable, Optional

from implicit.cpu.als import AlternatingLeastSquares
import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def build_cb_pipeline(pipeline_parameters: Optional[dict[str, Any]] = None) -> Pipeline:
    """
    Build a pipeline for the content-based model.
    
    This function creates a pipeline consisting of two stages:
    1. TfidfVectorizer - converts text features to TF-IDF matrix
    2. TruncatedSVD - dimensionality reduction to reduce computational complexity
    
    Parameters
    ----------
    pipeline_parameters : dict, default=None
        Dictionary with parameters for pipeline components.
        Supported keys:
        - 'tfidf': parameters for TfidfVectorizer
        - 'svd': parameters for TruncatedSVD
        If parameters are not specified, default values are used.
    
    Returns
    -------
    Pipeline
        Scikit-learn pipeline with configured components:
        - 'tfidf': TfidfVectorizer
        - 'svd': TruncatedSVD
    
    Raises
    ------
    TypeError
        If pipeline_parameters is not a dict or not None
    """
    
    if pipeline_parameters is not None and not isinstance(pipeline_parameters, dict):
        raise TypeError(
            f"pipeline_parameters must be dict or None, got {type(pipeline_parameters).__name__}"
        )
    
    if pipeline_parameters is None:
        pipeline_parameters = {}
        logger.debug('pipeline_parameters not specified, using default parameters')
    
    tfidf_params = pipeline_parameters.get('tfidf', {})
    svd_params = pipeline_parameters.get('svd', {})
    
    logger.info(
        'Creating CB pipeline with parameters: TfidfVectorizer(%s), TruncatedSVD(%s)',
        tfidf_params,
        svd_params
    )
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(**tfidf_params)),
        ('svd', TruncatedSVD(**svd_params))
    ])
    
    logger.info('Pipeline successfully created')
    
    return pipeline


def train_cb_pipeline(
    pipeline: Pipeline, 
    data: pd.Series | pd.DataFrame, 
    column: Optional[Hashable] = None
) -> Pipeline:
    """
    Train a content-based pipeline on text data.
    
    This function takes a pipeline (TfidfVectorizer + TruncatedSVD)
    and trains it on text data. Supports input data as pandas Series or DataFrame with column specification.
    
    Parameters
    ----------
    pipeline : Pipeline
        Scikit-learn pipeline with 'tfidf' and 'svd' components
    data : pd.Series or pd.DataFrame
        Input data:
        - If pd.Series: used directly as text data
        - If pd.DataFrame: uses the column specified in the column parameter
    column : Hashable, default=None
        Column name in DataFrame containing text data.
        Required if data is a DataFrame.
    
    Returns
    -------
    Pipeline
        Trained pipeline (fit model)
    
    Raises
    ------
    TypeError
        If pipeline is not a Pipeline instance
        If data is not pd.Series or pd.DataFrame
    ValueError
        If data is a DataFrame and column is not specified
        If column is not found in DataFrame
        If data is empty after transformation
    """
    
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"pipeline must be sklearn.pipeline.Pipeline, got {type(pipeline).__name__}"
        )
    
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"data must be pd.Series or pd.DataFrame, got {type(data).__name__}"
        )
    
    if isinstance(data, pd.Series):
        train_data = data.values.tolist()
        logger.info("Training on Series with %d texts", len(train_data))
        
    elif isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError(
                "For DataFrame, the column parameter with column name must be specified"
            )
        
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        train_data = data[column].values.tolist()
        logger.info(
            "Training on column '%s' of DataFrame: %d texts from %d rows",
            column, len(train_data), len(data)
        )
    
    if len(train_data) == 0:
        raise ValueError("No data for training")
    
    try:
        logger.info("Starting pipeline training")
        trained_pipeline = pipeline.fit(train_data)
        logger.info("Pipeline successfully trained")
        return trained_pipeline
    except Exception as e:
        logger.exception("Error training pipeline")
        raise RuntimeError(f"Failed to train pipeline: {e}") from e


def apply_fitted_cb_pipeline(
    pipeline: Pipeline, 
    data: pd.Series | pd.DataFrame, 
    column: Optional[Hashable] = None
) -> np.ndarray:
    """
    Apply a trained content-based pipeline to data.
    
    This function applies a trained pipeline (TfidfVectorizer + TruncatedSVD)
    to transform text data into feature space.
    
    Parameters
    ----------
    pipeline : Pipeline
        Trained scikit-learn pipeline with 'tfidf' and 'svd' components
    data : pd.Series or pd.DataFrame
        Input data:
        - If pd.Series: used directly as text data
        - If pd.DataFrame: uses the column specified in the column parameter
    column : Optional[Hashable], default=None
        Column name in DataFrame containing text data.
        Required if data is a DataFrame.
    
    Returns
    -------
    np.ndarray
        Feature matrix of shape (n_samples, n_components) after applying
        TF-IDF and SVD transformations.
    
    Raises
    ------
    TypeError
        If pipeline is not a Pipeline instance
        If data is not pd.Series or pd.DataFrame
    ValueError
        If data is a DataFrame and column is not specified
        If column is not found in DataFrame
        If data is empty after transformation
    RuntimeError
        If an error occurs when applying the pipeline
    """
    
    if not isinstance(pipeline, Pipeline):
        raise TypeError(
            f"pipeline must be sklearn.pipeline.Pipeline, got {type(pipeline).__name__}"
        )
    
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError(
            f"data must be pd.Series or pd.DataFrame, got {type(data).__name__}"
        )
    
    if isinstance(data, pd.Series):
        transform_data = data.values.tolist()
        logger.info("Applying pipeline to Series with %d texts", len(transform_data))
        
    elif isinstance(data, pd.DataFrame):
        if column is None:
            raise ValueError(
                "For DataFrame, the column parameter with column name must be specified"
            )
        
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame. Available columns: {list(data.columns)}")
        
        transform_data = data[column].values.tolist()
        logger.info(
            "Applying pipeline to column '%s' of DataFrame: %d texts from %d rows",
            column, len(transform_data), len(data)
        )
    
    if len(transform_data) == 0:
        raise ValueError("No data for transformation")
    
    try:
        logger.info("Applying trained pipeline to data")
        transformed_features = pipeline.transform(transform_data)
        logger.info(
            "Transformation completed: feature matrix shape = %s",
            transformed_features.shape
        )
        return transformed_features
    except Exception as e:
        logger.exception("Error applying pipeline")
        raise RuntimeError(f"Failed to apply pipeline: {e}") from e


def build_als_estimator(parameters: Optional[dict[str, Any]] = None) -> AlternatingLeastSquares:
    """
    Create an ALS (Alternating Least Squares) Estimator for collaborative filtering.
    
    This function creates and configures an AlternatingLeastSquares model from the implicit library.
    
    Parameters
    ----------
    parameters : Optional[dict], default=None
        Dictionary with parameters for the AlternatingLeastSquares model.
        If None, default parameters are used.
        
        Main ALS parameters:
        - factors : int, default=100
            Number of latent factors (embedding dimension)
        - regularization : float, default=0.01
            Regularization coefficient
        - iterations : int, default=15
            Number of training iterations
        - alpha : float, default=1.0
            Confidence parameter for implicit data  
    
    Returns
    -------
    AlternatingLeastSquares
        Configured ALS model for collaborative filtering
    
    Raises
    ------
    TypeError
        If parameters is not a dict or None
    """
    
    if parameters is not None and not isinstance(parameters, dict):
        raise TypeError(
            f"parameters must be dict or None, got {type(parameters).__name__}"
        )
    
    if parameters is None:
        parameters = {}
        logger.debug("Using default ALS parameters")
    else:
        logger.info(
            "Creating ALS model with parameters: %s",
            {k: v for k, v in parameters.items() if k != 'random_state'}
        )
    
    try:
        model = AlternatingLeastSquares(**parameters)
        logger.info("ALS model successfully created")
        return model
    except Exception as e:
        logger.exception("Error creating ALS model")
        raise RuntimeError(f"Failed to create ALS model: {e}") from e


def train_als_estimator(
    als_estimator: AlternatingLeastSquares, 
    data: sparse.csr_matrix
) -> AlternatingLeastSquares:
    """
    Train an ALS (Alternating Least Squares) model on a sparse matrix.
    
    Parameters
    ----------
    als_estimator : AlternatingLeastSquares
        Untrained or partially trained ALS model
    data : sparse.csr_matrix
        Sparse user-item matrix in CSR format.
        Shape: (n_users, n_items)
    
    Returns
    -------
    AlternatingLeastSquares
        Trained ALS model
    
    Raises
    ------
    TypeError
        If als_estimator is not an AlternatingLeastSquares
        If data is not a scipy.sparse.csr_matrix
    ValueError
        If the data matrix is empty or has incorrect dimensions
    RuntimeError
        If an error occurs during training
    """
    
    if not isinstance(als_estimator, AlternatingLeastSquares):
        raise TypeError(
            f"als_estimator must be AlternatingLeastSquares, got {type(als_estimator).__name__}"
        )
    
    if not isinstance(data, sparse.csr_matrix):
        raise TypeError(
            f"data must be scipy.sparse.csr_matrix, got {type(data).__name__}"
        )
    
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError(
            f"Data matrix has invalid dimensions: {data.shape}. "
            "Expected (n_users, n_items) with n_users > 0 and n_items > 0"
        )
    
    if data.nnz == 0:
        raise ValueError(
            f"Data matrix contains no non-zero elements (nnz = 0). "
            "Cannot train ALS model on an empty matrix"
        )
    
    try:
        logger.info(
            "Starting ALS model training. Matrix size: %d users, %d items, %d interactions",
            data.shape[0], data.shape[1], data.nnz
        )
        als_estimator.fit(data)
        logger.info("ALS model successfully trained")
        return als_estimator
    except Exception as e:
        logger.exception("Error training ALS model")
        raise RuntimeError(f"Failed to train ALS model: {e}") from e


def recommend_by_movieId(
    movieIds: int | list[int], 
    movie_features: np.ndarray, 
    movieId_to_idx: dict[int, int], 
    idx_to_movieId: dict[int, int], 
    k: int = 10
) -> tuple[tuple[list[int], list[float]], ...]:
    """
    Get similar movie recommendations based on movieId.
    
    This function computes cosine similarity between requested movies
    and all movies in the feature matrix, returning the top-k most similar.
    
    Parameters
    ----------
    movieIds : int or list[int]
        Movie ID or list of movie IDs to find similar movies for
    movie_features : np.ndarray
        Movie feature matrix of shape (n_movies, n_features)
    movieId_to_idx : dict[int, int]
        Dictionary mapping movieId -> index in matrix
    idx_to_movieId : dict[int, int]
        Dictionary mapping index -> movieId
    k : int, default=10
        Number of recommendations to return for each movie
    
    Returns
    -------
    tuple[tuple[list[int], list[float]], ...]
        Nested tuple of recommendations. For each requested movie,
        returns a tuple of pairs (movieId, similarity_score).
        
        Example: (( (101, 0.95), (102, 0.89) ), ( (201, 0.92), (202, 0.85) ))
    
    Raises
    ------
    TypeError
        If movieIds is not int or list[int]
        If movie_features is not np.ndarray
        If k is not int
    ValueError
        If k <= 0
        If movie_features is empty
    KeyError
        If movieId not found in movieId_to_idx
        If index not found in idx_to_movieId
    """
    
    if not isinstance(movieIds, (int, list)):
        raise TypeError(
            f"movieIds must be int or list[int], got {type(movieIds).__name__}"
        )
    
    if not isinstance(movie_features, np.ndarray):
        raise TypeError(
            f"movie_features must be np.ndarray, got {type(movie_features).__name__}"
        )
    
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    
    if k <= 0:
        raise ValueError(f"k must be greater than 0, got {k}")
    
    if len(movie_features) == 0:
        raise ValueError("movie_features matrix is empty")
    
    if isinstance(movieIds, int):
        movieIds = [movieIds]

    indices = []
    for movieId in movieIds:
        if movieId not in movieId_to_idx:
            logger.error("Movie %d not found in movieId_to_idx", movieId)
            raise KeyError(f"Movie {movieId} not found")
        indices.append(movieId_to_idx[movieId])
    
    for idx in indices:
        if idx in idx_to_movieId:
            if idx_to_movieId[idx] in movieIds:
                continue
            logger.error('Movie %d does not match requested', idx_to_movieId[idx])
            raise KeyError(f'Movie {idx_to_movieId[idx]} does not match requested')
        else:
            logger.error('Index %d not found', idx)
            raise KeyError(f'Index {idx} not found')
    
    logger.info("Searching for recommendations for %d movies, k=%d, total movies: %d",
                len(indices), k, len(movie_features))
    
    query_movies = movie_features[indices]
    
    similarities = cosine_similarity(query_movies, movie_features)
    
    n_movies = len(movie_features)
    top_k_indices = []
    
    for i in range(len(indices)):
        query_idx = indices[i]
        sim_row = similarities[i]
        
        sim_row[query_idx] = -1
    
        if n_movies <= k:
            top_idx = np.argsort(sim_row)[::-1][:n_movies-1]
        else:
            top_idx = np.argpartition(sim_row, -k)[-k:]
            top_idx = top_idx[np.argsort(sim_row[top_idx])[::-1]]
        
        top_k_indices.append(top_idx)
    
    recommendations = []
    for i, top_idx in enumerate(top_k_indices):
        rec_movieIds = [idx_to_movieId[idx] for idx in top_idx]
        rec_scores = similarities[i][top_idx].tolist()
        recommendations.append((rec_movieIds, rec_scores))
    
    logger.info(
        "Recommendations received: %d requests, %d recommendations each",
        len(recommendations),
        len(recommendations[0][0]) if recommendations else 0
    )
    
    return tuple(recommendations)


def recommend_by_userId(
    userIds: int | list[int], 
    als_model: AlternatingLeastSquares, 
    user_item_matrix: sparse.csr_matrix,
    movieId_to_idx: dict[int, int], 
    userId_to_idx: dict[int, int], 
    idx_to_movieId: dict[int, int], 
    idx_to_userId: dict[int, int], 
    k: int = 10,
    neg_movieIds: Optional[int | list[int] | dict[int, int | list[int]]] = None
) -> tuple[tuple[list[int], list[float]], ...]:
    """
    Get recommendations for user(s) based on a trained ALS model.
    
    This function returns a list of recommended movies for each specified user,
    with the ability to exclude negative movies (that should not be recommended).
    
    Parameters
    ----------
    userIds : int or list[int]
        User ID or list of user IDs to get recommendations for
    als_model : AlternatingLeastSquares
        Trained ALS model from the implicit library
    user_item_matrix : sparse.csr_matrix
        Sparse user-item matrix (users × items)
    movieId_to_idx : dict[int, int]
        Dictionary mapping movieId -> index in matrix
    userId_to_idx : dict[int, int]
        Dictionary mapping userId -> index in matrix
    idx_to_movieId : dict[int, int]
        Dictionary mapping index -> movieId
    idx_to_userId : dict[int, int]
        Dictionary mapping index -> userId
    k : int, default=10
        Number of recommendations to return for each user
    neg_movieIds : Optional[int | list[int] | dict[int, int | list[int]]], default=None
        Movies to exclude from recommendations.
        Can be:
        - int: single movie for the first user
        - list[int]: list of movies for the first user
        - dict[int, int | list[int]]: dictionary {userId: movie(s)} for different users
    
    Returns
    -------
    tuple[tuple[list[int], list[float]], ...]
        Tuple of results for each user. Each result is a tuple of two elements:
        - list[int]: list of recommended movie IDs
        - list[float]: list of scores
    
    Raises
    ------
    TypeError
        If input parameters have incorrect types
    KeyError
        If userId not found in userId_to_idx
        If movieId from neg_movieIds not found in movieId_to_idx
        If index not found in idx_to_movieId or idx_to_userId
    ValueError
        If k <= 0
    """

    if not isinstance(userIds, (int, list)):
        raise TypeError(
            f"userIds must be int or List[int], got {type(userIds).__name__}"
        )
    
    if not isinstance(als_model, AlternatingLeastSquares):
        raise TypeError(
            f"als_model must be AlternatingLeastSquares, got {type(als_model).__name__}"
        )
    
    if not isinstance(user_item_matrix, sparse.csr_matrix):
        raise TypeError(
            f"user_item_matrix must be scipy.sparse.csr_matrix, got {type(user_item_matrix).__name__}"
        )
    
    if not isinstance(k, int):
        raise TypeError(f"k must be int, got {type(k).__name__}")
    
    if k <= 0:
        raise ValueError(f"k must be greater than 0, got {k}")
    
    if isinstance(userIds, int):
        userIds = [userIds]
    
    if neg_movieIds is None:
        neg_movieIds = {}
    elif isinstance(neg_movieIds, int):
        neg_movieIds = {userIds[0]: [neg_movieIds]}
    elif isinstance(neg_movieIds, list):
        neg_movieIds = {userIds[0]: neg_movieIds}
    elif not isinstance(neg_movieIds, dict):
        raise TypeError(
            f"neg_movieIds must be int, list, dict or None, got {type(neg_movieIds).__name__}"
        )
    
    normalized_neg_movieIds = {}
    for uid, movies in neg_movieIds.items():
        if isinstance(movies, int):
            normalized_neg_movieIds[uid] = [movies]
        elif isinstance(movies, list):
            normalized_neg_movieIds[uid] = movies
        else:
            raise TypeError(
                f"Values in neg_movieIds must be int or list, got {type(movies).__name__}"
            )
    neg_movieIds = normalized_neg_movieIds
    
    logger.info(
        "Getting recommendations for %d users, k=%d",
        len(userIds), k
    )

    user_indices = []
    for userId in userIds:
        if userId not in userId_to_idx:
            logger.error("User %d not found in userId_to_idx", userId)
            raise KeyError(f"User {userId} not found")
        user_indices.append(userId_to_idx[userId])
    
    for idx in user_indices:
        if idx not in idx_to_userId:
            logger.error("Index %d not found in idx_to_userId", idx)
            raise KeyError(f"Index {idx} not found")
        if idx_to_userId[idx] not in userIds:
            logger.warning(
                "User %d (index %d) does not match requested %s",
                idx_to_userId[idx], idx, userIds
            )
    
    user_filters = {}
    for userId in userIds:
        if userId in neg_movieIds:
            movie_indices = []
            for movieId in neg_movieIds[userId]:
                if movieId not in movieId_to_idx:
                    logger.error("Movie %d not found in movieId_to_idx", movieId)
                    raise KeyError(f"Movie {movieId} not found")
                movie_idx = movieId_to_idx[movieId]
                
                if movie_idx not in idx_to_movieId:
                    logger.error("Index %d not found in idx_to_movieId", movie_idx)
                    raise KeyError(f"Index {movie_idx} not found")
                if idx_to_movieId[movie_idx] != movieId:
                    logger.warning(
                        "Mismatch: index %d -> %d, expected %d",
                        movie_idx, idx_to_movieId[movie_idx], movieId
                    )
                
                movie_indices.append(movie_idx)
            
            user_filters[userId] = movie_indices
            logger.debug(
                "For user %d, %d movies will be excluded",
                userId, len(movie_indices)
            )
    
    recommendations = []
    for i, user_idx in enumerate(user_indices):
        userId = userIds[i]
        filter_list = user_filters.get(userId, None)
        
        logger.debug("Requesting recommendations for user %d (index %d)",
                     userId, user_idx)
        
        try:
            ids, scores = als_model.recommend(
                user_idx, 
                user_item_matrix[user_idx], 
                N=k,
                filter_items=filter_list,
                filter_already_liked_items=True
            )
            
            rel_movieIds = [idx_to_movieId[idx] for idx in ids]
            recommendations.append((rel_movieIds, scores.tolist()))
            
            logger.debug(
                "For user %d received %d recommendations, average score: %.4f",
                userId, len(rel_movieIds), scores.mean()
            )
            
        except Exception as e:
            logger.exception("Error getting recommendations for user %d", userId)
            raise RuntimeError(f"Failed to get recommendations for user {userId}: {e}") from e
    
    logger.info("Successfully received recommendations for %d users", len(recommendations))
    
    return tuple(recommendations)


def save_model(model: Pipeline | AlternatingLeastSquares, path: Path) -> None:
    """
    Save a model to a file via joblib.
    
    Parameters
    ----------
    model : Pipeline or AlternatingLeastSquares
        Trained model to save
    path : Path
        Path to save to. Extension will be automatically changed to .joblib
    
    Raises
    ----------
    TypeError, OSError, RuntimeError
    """
    
    if not isinstance(model, (Pipeline, AlternatingLeastSquares)):
        raise TypeError(
            f"model must be Pipeline or AlternatingLeastSquares, "
            f"got {type(model).__name__}"
        )

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.exception("Failed to create directory %s", path.parent)
        raise OSError(f"Failed to create directory: {e}") from e
    
    if path.exists():
        logger.info("File %s already exists, skipping save", path)
        return
    
    if path.suffix != '.joblib':
        logger.info("Extension changed from %s to .joblib", path.suffix)
        path = path.with_suffix('.joblib')
    
    logger.info("Saving model %s to %s", type(model).__name__, path)
    
    try:
        joblib.dump(model, path)
        logger.info("Model successfully saved to %s", path)
    except Exception as e:
        logger.exception("Error saving model to %s", path)
        raise RuntimeError(f"Failed to save model: {e}") from e


def load_model(path: Path) -> Pipeline | AlternatingLeastSquares:
    """
    Load a model from a file via joblib.
    
    Parameters
    ----------
    path : Path
        Path to the model file (expected extension .joblib)
    
    Returns
    -------
    Pipeline or AlternatingLeastSquares
    
    Raises
    ----------
    FileNotFoundError, TypeError, RuntimeError
    """
    
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    if path.suffix != '.joblib':
        logger.warning("Expected .joblib extension, but got %s. Attempting to load...", path.suffix)
    
    logger.info("Loading model from %s", path)
    
    try:
        model = joblib.load(path)
        
        if not isinstance(model, (Pipeline, AlternatingLeastSquares)):
            raise TypeError(
                f"Loaded object has type {type(model).__name__}, "
                "expected Pipeline or AlternatingLeastSquares"
            )
        
        if isinstance(model, AlternatingLeastSquares):
            if model.user_factors is None or model.item_factors is None:
                raise RuntimeError(
                    "ALS model loaded but user_factors or item_factors = None. "
                    "Retraining required or correct saving via joblib."
                )
            logger.info(
                "ALS model loaded: user_factors=%s, item_factors=%s",
                model.user_factors.shape,
                model.item_factors.shape
            )
        
        logger.info("Model successfully loaded: %s", type(model).__name__)
        return model
        
    except (FileNotFoundError, TypeError, RuntimeError):
        raise
    except Exception as e:
        logger.exception("Error loading model from %s", path)
        raise RuntimeError(f"Failed to load model: {e}") from e