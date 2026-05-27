""" src/mrh/api/main.py
API application: application lifecycle management and creation.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from mrh.api.endpoints import health, predict
from mrh.api.dependencies import load_cb_model, load_cf_model
from mrh.utils import setup_logging

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage the lifecycle of the FastAPI application.
    
    Executed when the server starts and stops:
    - On startup: configure logging, load models into memory
    - On shutdown: clear model cache, release resources
    
    Parameters
    ----------
    app : FastAPI
        FastAPI application instance
    """

    setup_logging()
    logger.info('Starting Movie RecSys API')

    try:
        load_cb_model()
        load_cf_model()
        logger.info('Models loaded into memory')
    except Exception as e:
        logger.warning('Failed to load models into memory: %s', e)
    
    yield

    logger.info('Stopping server and cleaning resources')
    try:
        load_cb_model.cache_clear()
        load_cf_model.cache_clear()
        logger.info('Model cache cleared')
    except Exception as e:
        logger.exception('Error clearing model cache')
    
    logger.info('Server stopped')


def create_app() -> FastAPI:
    """
    Create and configure a FastAPI application instance.
    
    Configures:
    - Application metadata (title, version, description)
    - Lifespan manager for model management
    - CORS middleware for cross-domain requests
    - Routes (routers) for endpoints
    - Root endpoint with service information
    
    Returns
    -------
    FastAPI
        Configured FastAPI application instance
    """

    app = FastAPI(
        title='Movie RecSys Hybrid API',
        description='Hybrid movie recommendation system',
        version='0.1.0',
        lifespan=lifespan,
        docs_url='/docs',
        redoc_url='/redoc',
        openapi_tags=[
            {'name': 'Root', 'description': 'Basic service information'},
            {'name': 'health', 'description': 'Service availability and health checks'},
            {'name': 'predictions', 'description': 'Get recommendations'},
        ]
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=False,
        allow_methods=['*'],
        allow_headers=['*']
    )

    app.include_router(health.router)
    app.include_router(predict.router)

    @app.get('/', tags=['Root'])
    async def root() -> dict:
        """
        Root endpoint with service information.
        
        Returns
        -------
        dict
            Service information, version and available endpoints
        """
        return {
            'service': 'Movie RecSys Hybrid API',
            'version': '0.1.0',
            'docs': '/docs',
            'health': '/health',
            'predict': '/predict'
        }
    
    return app