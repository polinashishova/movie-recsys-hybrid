""" src/mrh/api/schemas.py
Request and response schemas.
"""

from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Optional, Literal, Annotated
from datetime import datetime, timezone


class PredictRequest(BaseModel):
    """
    Request for movie recommendations.
    
    Can be of two types:
    - By movie IDs (cold-start): returns similar movies
    - By user IDs: returns personalized recommendations
    ID must be a positive number
    You can request recommendations for no more than 500 IDs at a time
    """

    movieIds: Annotated[
        Optional[list[int]], 
        Field(
            default=None, 
            description='Movie IDs for cold-start recommendations (similar movies)',
            examples=[[1225, 329]],
            max_length=500
        )
    ]

    userIds: Annotated[
        Optional[list[int]],
        Field(
            default=None,
            description='User IDs for personalized movie recommendations',
            examples=[[214, 154198]],
            max_length=500
        )
    ]

    k: Annotated[
        int,
        Field(
            default=10,
            description='Number of movies to recommend',
            ge=1, 
            le=100
        )
    ]

    @field_validator('movieIds', 'userIds', mode='after')
    @classmethod
    def validate_ids(cls, v):
        if v is not None:
            if any(id < 0 for id in v):
                raise ValueError('ID must be a positive number')
        return v

    @model_validator(mode='after')
    def check_either_movies_or_users(self):
        """Check that exactly one of the parameters is specified: movieIds or userIds."""
        if self.movieIds is not None and self.userIds is not None:
            raise ValueError('Specify either only movieIds or only userIds. Both were specified')
        elif self.movieIds is None and self.userIds is None:
            raise ValueError('Specify either movieIds or userIds')
        return self
    
    
class RecommendationItem(BaseModel):
    """
    Recommendation item: movie with its relevance score.
    """

    movieId: Annotated[
        int, 
        Field(description='ID of the recommended movie')
    ]

    score: Annotated[
        float, 
        Field(description='Relevance score of the recommended movie (higher is better)')
    ]


class PredictResponse(BaseModel):
    """
    Response with movie recommendations.
    """

    recommendations: Annotated[
        dict[int, list[RecommendationItem]],
        Field(description='Recommendation dictionary: {"requested ID": [recommendations]}')
    ]

    model_used: Annotated[
        Literal['content_based', 'collaborative_filtering'],
        Field(description='Type of model used for prediction')
    ]

    timestamp: Annotated[
        str,
        Field(
            default_factory=lambda: datetime.now(timezone.utc).isoformat(),
            description='Response time'
        )
    ]

    request_id: Annotated[
        Optional[str],
        Field(
            default=None,
            description='Request ID'
        )
    ]


class HealthResponse(BaseModel):
    """
    Service health check response.
    """

    status: Annotated[
        Literal['healthy', 'unhealthy'],
        Field(description='Service status')
    ]

    service: Annotated[
        Literal['movie-recsys-hybrid'],
        Field(
            default='movie-recsys-hybrid',
            description='Service name'
        )
    ]

    models_loaded: Annotated[
        bool,
        Field(description='Flag indicating whether models are loaded')
    ]

    timestamp: Annotated[
        str,
        Field(
            default_factory=lambda: datetime.now(timezone.utc).isoformat(),
            description='Response time'
        )
    ]