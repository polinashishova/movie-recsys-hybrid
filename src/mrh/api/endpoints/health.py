""" src/mrh/api/endpoints/health.py
Endpoint for service health check.
"""

from fastapi import APIRouter, status, Response
from fastapi.responses import JSONResponse
import logging

from mrh.api.schemas import HealthResponse
from mrh.api.dependencies import is_models_ready


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/health", tags=['health'])

@router.get("", response_model=HealthResponse)
async def health_check(response: Response) -> HealthResponse | JSONResponse:
    """
    Endpoint for checking service status.
    
    Returns
    -------
    HealthResponse
        With status 'healthy' (HTTP 200) or 'unhealthy' (HTTP 503)
    
    Response Codes
    --------------
    - 200 OK: Service is healthy, models are loaded
    - 503 Service Unavailable: Service is running but models are not loaded
    """

    models_ok = is_models_ready()
    
    result = HealthResponse(
        status='healthy' if models_ok else 'unhealthy',
        service='movie-recsys-hybrid',
        models_loaded=models_ok
    )

    if not models_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        logger.warning('Health check failed: models are not ready')
    else:
        logger.info('Health check passed: models are loaded')
    return result