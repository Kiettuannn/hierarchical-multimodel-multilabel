import uuid
from datetime import datetime
from typing import BinaryIO, Dict, List, Optional

from app.core.logging import logger
from app.domain.interfaces.inference import IInferenceService
from app.domain.interfaces.storage import IVideoStorage
from app.domain.models import JobStatus, ModerationResult


class ModerationService:
    """Use case coordinator managing video moderation lifecycle."""

    def __init__(self, inference_service: IInferenceService, storage: IVideoStorage):
        self._inference_service = inference_service
        self._storage = storage
        # In-memory store; can be swapped for Redis / PostgreSQL repository adapter
        self._jobs: Dict[str, ModerationResult] = {}

    async def submit_job(self, filename: str, file_stream: BinaryIO) -> tuple[str, str]:
        """Save video to storage and register a new pending moderation job."""
        job_id = uuid.uuid4().hex
        stored_path = await self._storage.save(filename, file_stream)

        initial_result = ModerationResult(
            job_id=job_id,
            video_filename=filename,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
        )
        self._jobs[job_id] = initial_result
        logger.info(f"Created moderation job {job_id} for file {filename}")
        return job_id, stored_path

    async def execute_job(self, job_id: str, video_path: str, filename: str) -> None:
        """Run the inference pipeline and update job status upon completion."""
        job = self._jobs.get(job_id)
        if not job:
            logger.error(f"Cannot execute job {job_id}: job not found")
            return

        job.status = JobStatus.PROCESSING
        try:
            result = await self._inference_service.predict(
                job_id=job_id,
                video_path=video_path,
                filename=filename,
            )
            self._jobs[job_id] = result
            logger.info(f"Successfully processed job {job_id}")
        except Exception as exc:
            logger.exception(f"Error processing job {job_id}: {exc}")
            job.status = JobStatus.FAILED
            job.error_message = str(exc)
            job.completed_at = datetime.utcnow()
        finally:
            # Clean up the temporary video file to conserve disk space
            await self._storage.delete(video_path)

    def get_job(self, job_id: str) -> Optional[ModerationResult]:
        """Retrieve moderation status and result for a specific job."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[ModerationResult]:
        """List all tracked moderation jobs."""
        return list(self._jobs.values())
