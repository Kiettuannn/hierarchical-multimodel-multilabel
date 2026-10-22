"""Inference service adapters."""
from app.infrastructure.inference.mock_inference_service import MockInferenceService
from app.infrastructure.inference.pipeline_inference_service import PipelineInferenceService

__all__ = ["MockInferenceService", "PipelineInferenceService"]
