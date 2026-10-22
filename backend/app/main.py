from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import logger, setup_logging


def create_application() -> FastAPI:
    """Factory function for FastAPI application initialization."""
    setup_logging()

    application = FastAPI(
        title=settings.APP_NAME,
        description=(
            "Production API for Hierarchical Multimodal Multi-label Video Moderation. "
            "Detects Harmful Videos (Stage 1 Gatekeeper) and classifies violations across "
            "7 fine-grained Harm Categories (Stage 2 Multilabel)."
        ),
        version="1.0.0",
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
    )

    # CORS configuration
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include main API v1 router
    application.include_router(api_router, prefix=settings.API_V1_PREFIX)

    @application.on_event("startup")
    async def startup_event():
        logger.info(f"Starting {settings.APP_NAME} in '{settings.INFERENCE_MODE}' inference mode")

    @application.on_event("shutdown")
    async def shutdown_event():
        logger.info(f"Shutting down {settings.APP_NAME}")

    return application


app = create_application()
