""" src/mrh/api/endpoints/predict.py
Endpoint for predicting recommendations either by movie IDs or user IDs.
"""

import uuid
import logging
from fastapi import APIRouter, Depends, HTTPException, status
import numpy as np
from typing import Annotated, Optional

from sklearn.pipeline import Pipeline
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix

from mrh.api.schemas import PredictRequest, PredictResponse, RecommendationItem
from mrh.api.dependencies import get_cb_model_dep, get_cf_model_dep
from mrh.models import recommend_by_movieId, recommend_by_userId

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/predict', tags=['predictions'])


@router.post("", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    cb_model_data: Annotated[
        tuple[Pipeline, 
              np.ndarray,
              dict[int, int],
              dict[int, int]], 
        Depends(get_cb_model_dep)],
    cf_model_data: Annotated[
        tuple[AlternatingLeastSquares, 
              csr_matrix,
              dict[int, int],
              dict[int, int],
              dict[int, int],
              dict[int, int],
              dict[Optional[int], Optional[list[int]]]], 
        Depends(get_cf_model_dep)]
) -> PredictResponse:
    """
    Endpoint for getting movie recommendations.
    
    Supports two types of requests:
    - By movieIds: returns similar movies (content-based)
    - By userIds: returns personalized recommendations (collaborative filtering)
    
    Parameters
    ----------
    request : PredictRequest
        Request body with parameters (movieIds or userIds, k)

    Dependencies
    ------------------------------------
    cb_model_data 
        Content-based model data (loaded from cache)
    cf_model_data
        Collaborative filtering model data (loaded from cache)
    
    Returns
    -------
    PredictResponse
        Response object with recommendations and metadata
    
    Raises
    ------
    HTTPException (404)
        If the requested movieId or userId is not found in the model
    HTTPException (500)
        If an internal error occurs during processing
    HTTPException (400)
        If movieIds or userIds is empty
    """
     
    request_id = str(uuid.uuid4())[:8]
    logger.info('Request %s: k=%d, movieIds=%s, userIds=%s',
                request_id, request.k, request.movieIds, request.userIds)
    if request.movieIds is not None and len(request.movieIds) == 0:
        raise HTTPException(
            status_code=400,
            detail="movieIds cannot be an empty list"
        )
    
    elif request.userIds is not None and len(request.userIds) == 0:
        raise HTTPException(
            status_code=400,
            detail="userIds cannot be an empty list"
        )
    
    try:
        if request.movieIds is not None:
                
            cb_pipeline, movie_features, movieId_to_idx, idx_to_movieId = cb_model_data

            recs = recommend_by_movieId(
                movieIds=request.movieIds,
                movie_features=movie_features,
                movieId_to_idx=movieId_to_idx,
                idx_to_movieId=idx_to_movieId,
                k=request.k
            )

            predictions = {}
            for (rec_ids, scores), req_id in zip(recs, request.movieIds):
                predictions[req_id] = [
                    RecommendationItem(movieId=mid, score=float(score))
                    for mid, score in zip(rec_ids, scores)
                ]
            
            return PredictResponse(
                recommendations=predictions,
                model_used='content_based',
                request_id=request_id
            )
        
        elif request.userIds is not None:
            
            cf_model, user_item_matrix, movieId_to_idx, userId_to_idx, idx_to_movieId, idx_to_userId, cf_neg_movieIds = cf_model_data

            recs = recommend_by_userId(
                userIds=request.userIds, 
                als_model=cf_model, 
                user_item_matrix=user_item_matrix,
                movieId_to_idx=movieId_to_idx, 
                userId_to_idx=userId_to_idx, 
                idx_to_movieId=idx_to_movieId, 
                idx_to_userId=idx_to_userId, 
                k=request.k,
                neg_movieIds=cf_neg_movieIds
            )

            predictions = {}
            for (rec_ids, scores), req_id in zip(recs, request.userIds):
                predictions[req_id] = [
                    RecommendationItem(movieId=mid, score=float(score))
                    for mid, score in zip(rec_ids, scores)
                ]
            
            return PredictResponse(
                recommendations=predictions,
                model_used='collaborative_filtering',
                request_id=request_id
            )
        
    except KeyError as e:
        logger.warning("Request %s: not found %s", request_id, e)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Not found: {e}"
        )
    except Exception as e:
        logger.exception("Request %s: processing error", request_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Request processing error"
        )