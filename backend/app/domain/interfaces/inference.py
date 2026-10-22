from abc import ABC, abstractmethod
from app.domain.models import ModerationResult


class IInferenceService(ABC):
    """Abstract interface defining the contract for multimodal video inference."""

    @abstractmethod
    async def predict(self, job_id: str, video_path: str, filename: str) -> ModerationResult:
        """Run hierarchical multimodal classification on a video file.

        Args:
            job_id: Unique identifier for the moderation job.
            video_path: Absolute or local path to the target .mp4 video.
            filename: Original uploaded filename.

        Returns:
            ModerationResult containing Stage1 and Stage2 predictions.
        """
        raise NotImplementedError
