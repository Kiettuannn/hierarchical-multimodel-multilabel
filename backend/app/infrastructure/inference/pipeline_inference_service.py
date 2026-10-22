from app.core.logging import logger
from app.domain.interfaces.inference import IInferenceService
from app.domain.models import ModerationResult


class PipelineInferenceService(IInferenceService):
    """Production inference adapter integrating the 4-branch multimodal cascade.

    Note: This class serves as the clean architecture seam. The heavy model weights
    (Demucs, ChunkFormer, PaddleOCR, SigLIP2, PhoBERT, CLAP) and FusionMLP checkpoints
    are wired here or delegated to a remote GPU execution worker.
    """

    def __init__(self, stage1_checkpoint: str = "", stage2_checkpoint: str = ""):
        self.stage1_checkpoint = stage1_checkpoint
        self.stage2_checkpoint = stage2_checkpoint
        logger.info("Initialized PipelineInferenceService skeleton")

    async def predict(self, job_id: str, video_path: str, filename: str) -> ModerationResult:
        """Execute multimodal inference pipeline.

        To be implemented when deploying to a GPU worker environment.
        For logic reference, inspect `experiments/inference/end_to_end_demo.py`.
        """
        raise NotImplementedError(
            "PipelineInferenceService requires GPU runtime dependencies. "
            "Set INFERENCE_MODE=mock for local development."
        )
