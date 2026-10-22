import asyncio
import random
from datetime import datetime
from app.core.logging import logger
from app.domain.interfaces.inference import IInferenceService
from app.domain.models import (
    HarmCategory,
    JobStatus,
    ModerationResult,
    Stage1Result,
    Stage2Result,
)


class MockInferenceService(IInferenceService):
    """Mock inference service for local development without GPU dependencies.

    Generates realistic predictions conforming strictly to the HMM-Net taxonomy.
    """

    async def predict(self, job_id: str, video_path: str, filename: str) -> ModerationResult:
        logger.info(f"[MockInference] Starting mock prediction for {filename} (job: {job_id})")

        # Simulate processing delay across 4 modalities
        await asyncio.sleep(2.0)

        # Deterministic simulation based on filename cues
        lower_name = filename.lower()
        if "normal" in lower_name or "clean" in lower_name or "safe" in lower_name:
            stage1 = Stage1Result(is_harmful=False, confidence=0.96)
            stage2 = Stage2Result(
                predicted_categories=[],
                category_scores={cat: round(random.uniform(0.01, 0.15), 4) for cat in HarmCategory},
            )
        else:
            stage1 = Stage1Result(is_harmful=True, confidence=0.88)
            predicted = [HarmCategory.CLICKBAIT_HARM, HarmCategory.INFORMATION_HARM]
            scores = {cat: round(random.uniform(0.05, 0.25), 4) for cat in HarmCategory}
            scores[HarmCategory.CLICKBAIT_HARM] = 0.89
            scores[HarmCategory.INFORMATION_HARM] = 0.76

            stage2 = Stage2Result(
                predicted_categories=predicted,
                category_scores=scores,
            )

        logger.info(f"[MockInference] Completed prediction for {filename}. Harmful: {stage1.is_harmful}")

        return ModerationResult(
            job_id=job_id,
            video_filename=filename,
            status=JobStatus.COMPLETED,
            stage1=stage1,
            stage2=stage2,
            completed_at=datetime.utcnow(),
        )
