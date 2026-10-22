from functools import lru_cache

from app.core.config import settings
from app.domain.interfaces.inference import IInferenceService
from app.domain.interfaces.storage import IVideoStorage
from app.infrastructure.inference.mock_inference_service import MockInferenceService
from app.infrastructure.inference.pipeline_inference_service import PipelineInferenceService
from app.infrastructure.storage.local_storage import LocalVideoStorage
from app.services.moderation_service import ModerationService


@lru_cache()
def get_storage() -> IVideoStorage:
    """Provide singleton storage instance."""
    return LocalVideoStorage(base_dir=settings.UPLOAD_DIR)


@lru_cache()
def get_inference_service() -> IInferenceService:
    """Provide inference service adapter based on application settings."""
    if settings.INFERENCE_MODE.lower() == "pipeline":
        return PipelineInferenceService()
    return MockInferenceService()


# Shared application service instance
_moderation_service_instance: ModerationService | None = None


def get_moderation_service() -> ModerationService:
    """Dependency provider for ModerationService."""
    global _moderation_service_instance
    if _moderation_service_instance is None:
        _moderation_service_instance = ModerationService(
            inference_service=get_inference_service(),
            storage=get_storage(),
        )
    return _moderation_service_instance
