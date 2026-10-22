from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint exposing runtime metadata."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "inference_mode": settings.INFERENCE_MODE,
        "api_prefix": settings.API_V1_PREFIX,
    }
