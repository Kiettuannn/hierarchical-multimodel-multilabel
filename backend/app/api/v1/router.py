from fastapi import APIRouter
from app.api.v1.endpoints import health, moderation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(moderation.router)
