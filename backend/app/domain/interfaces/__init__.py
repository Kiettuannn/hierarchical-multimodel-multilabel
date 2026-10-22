"""Domain service contracts and interfaces."""
from app.domain.interfaces.inference import IInferenceService
from app.domain.interfaces.storage import IVideoStorage

__all__ = ["IInferenceService", "IVideoStorage"]
