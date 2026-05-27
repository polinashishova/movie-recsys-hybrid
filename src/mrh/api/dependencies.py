""" src/mrh/api/dependencies.py
API application dependencies: checking model availability, loading models into memory.
"""

from functools import lru_cache
from pathlib import Path
from fastapi import HTTPException, status
from typing import Optional

from sklearn.pipeline import Pipeline
from implicit.cpu.als import AlternatingLeastSquares
import logging
import numpy as np
from scipy.sparse import csr_matrix

from mrh.models import load_model
from mrh.data import load_data
from mrh.utils import load_json, inverse_dict
from mrh.config import CONFIGS_DIR


logger = logging.getLogger(__name__) 


def is_models_ready() -> bool:
    """
    Check if models are ready by accessing the cache.
    """
    try:
        return (load_cb_model.cache_info().currsize > 0 and load_cf_model.cache_info().currsize > 0)
    except (AttributeError, TypeError, RuntimeError):
        return False


@lru_cache(maxsize=1)
def get_config_paths() -> dict[str, str]:
    """
    Load configuration of file paths.
    
    Returns
    -------
    dict[str, str]
        Dictionary with file and directory paths from the configuration
    """

    root_path = Path.cwd()
    return load_json(root_path / CONFIGS_DIR / 'paths.json')


@lru_cache(maxsize=1)
def load_cb_model() -> tuple[Pipeline, 
           np.ndarray,
           dict[int, int],
           dict[int, int]]:
    """
    Load the content-based model and associated data.
    
    Returns
    -------
    tuple[Pipeline, np.ndarray, dict[int, int], dict[int, int]]
        Tuple of four elements:
        - cb_pipeline : Pipeline
            Trained pipeline (TfidfVectorizer + TruncatedSVD)
        - cb_features : np.ndarray
            Movie feature matrix (n_movies, n_components)
        - cb_movieId_to_idx : dict[int, int]
            Dictionary mapping movieId -> index in matrix
        - cb_idx_to_movieId : dict[int, int]
            Dictionary mapping index -> movieId
    
    Raises
    ------
    HTTPException (503 SERVICE UNAVAILABLE)
        If the model or associated files could not be loaded
    """

    root_path = Path.cwd()
    
    paths = get_config_paths()
    data_processed_dir = root_path / paths['data_processed_dir']
    data_features_dir = root_path / paths['data_features_dir']
    models_dir = root_path / paths['artifacts_dir'] / paths['models_dir']

    cb_pipeline_path = models_dir / paths['cb_pipeline']
    cb_features_path = data_features_dir / paths['cb_features']
    cb_movieId_to_idx_path = data_processed_dir / paths['cb_movieId_to_idx']
    

    try:
        logger.info('Loading content-based model ...')
        cb_pipeline = load_model(cb_pipeline_path)
        cb_features = load_data(cb_features_path)
        if hasattr(cb_features, 'to_numpy'):
            features_array = cb_features.to_numpy()
        else:
            features_array = np.asarray(cb_features)
        cb_movieId_to_idx = {int(k): int(v) for k, v in load_json(cb_movieId_to_idx_path).items()}
        cb_idx_to_movieId = inverse_dict(cb_movieId_to_idx)

        logger.info('Content-based model loaded: %d movies, %d features',
                    len(cb_movieId_to_idx), cb_features.shape[1])
        
        return cb_pipeline, features_array, cb_movieId_to_idx, cb_idx_to_movieId
    
    except Exception as e:
        logger.exception('Error loading content-based model')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Failed to load content-based model: {e}'
        )
    

@lru_cache(maxsize=1)
def load_cf_model() -> tuple[AlternatingLeastSquares, 
           csr_matrix, 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[Optional[int], Optional[list[int]]]]:
    """
    Load the ALS collaborative filtering model and associated data.
    
    Returns
    -------
    tuple[AlternatingLeastSquares, csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[Optional[int], Optional[list[int]]]]
        Tuple of seven elements:
        - cf_model : AlternatingLeastSquares
            Trained ALS model
        - cf_user_item_matrix: csr_matrix
            Matrix of training user-item interactions
        - cf_movieId_to_idx : dict[int, int]
            Dictionary mapping movieId -> index in matrix
        - cf_userId_to_idx : dict[int, int]
            Dictionary mapping userId -> index in matrix
        - cf_idx_to_movieId : dict[int, int]
            Dictionary mapping index -> movieId
        - cf_idx_to_userId : dict[int, int]
            Dictionary mapping index -> userId
        - cf_neg_movieIds: dict[Optional[int], Optional[list[int]]]
            Dictionary of movies to avoid recommending to specific users
    
    Raises
    ------
    HTTPException (503 SERVICE UNAVAILABLE)
        If the model or associated files could not be loaded
    """

    root_path = Path.cwd()
    
    paths = get_config_paths()
    data_processed_dir = root_path / paths['data_processed_dir']
    models_dir = root_path / paths['artifacts_dir'] / paths['models_dir']

    cf_model_path = models_dir / paths['als_model']
    cf_user_item_path = data_processed_dir / paths['cf_user_item_matrix']
    cf_movieId_to_idx_path = data_processed_dir / paths['cf_movieId_to_idx']
    cf_userId_to_idx_path = data_processed_dir / paths['cf_userId_to_idx']
    cf_neg_movieIds_path = data_processed_dir / paths['neg_cf_movieIds']
    

    try:
        logger.info('Loading collaborative filtering model ...')
        cf_model = load_model(cf_model_path)
        cf_user_item_matrix = load_data(cf_user_item_path)
        cf_movieId_to_idx = {int(k): int(v) for k, v in load_json(cf_movieId_to_idx_path).items()}
        cf_userId_to_idx = {int(k): int(v) for k, v in load_json(cf_userId_to_idx_path).items()}
        cf_idx_to_movieId = inverse_dict(cf_movieId_to_idx)
        cf_idx_to_userId = inverse_dict(cf_userId_to_idx)
        if cf_neg_movieIds_path.exists():
            cf_neg_movieIds = {int(k): v for k, v in load_json(cf_neg_movieIds_path).items()}
        else:
            cf_neg_movieIds = {}

        logger.info('Collaborative filtering model loaded: %d users, %d movies',
                    len(cf_userId_to_idx), len(cf_movieId_to_idx))
        
        return cf_model, cf_user_item_matrix, cf_movieId_to_idx, cf_userId_to_idx, cf_idx_to_movieId, cf_idx_to_userId, cf_neg_movieIds
    
    except Exception as e:
        logger.exception('Error loading collaborative filtering model')
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Failed to load collaborative filtering model: {e}'
        )


def get_cb_model_dep() -> tuple[Pipeline, 
           np.ndarray, 
           dict[int, int], 
           dict[int, int]]:
    """
    Dependency for injecting the content-based model into FastAPI endpoints.
    
    Returns
    -------
    tuple[Pipeline, np.ndarray, dict[int, int], dict[int, int]]
        Content-based model data (see load_cb_model)
    """

    return load_cb_model()

def get_cf_model_dep() -> tuple[AlternatingLeastSquares, 
           csr_matrix, 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[int, int], 
           dict[Optional[int], Optional[list[int]]]]:
    """
    Dependency for injecting the collaborative filtering model into FastAPI endpoints.
    
    Returns
    -------
    tuple[AlternatingLeastSquares, csr_matrix, dict[int, int], dict[int, int], dict[int, int], dict[int, int], dict[Optional[int], Optional[list[int]]]]
        Collaborative filtering model data (see load_cf_model)
    """

    return load_cf_model()