from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class HarmCategory(str, Enum):
    """The canonical seven harmful taxonomy labels defined in GLOSSARY.md."""

    INFORMATION_HARM = "information_harm"
    SEXUAL_HARM = "sexual_harm"
    PSYCHOLOGICAL_HARM = "psychological_harm"
    HATE_HARASSMENT_HARM = "hate_harassment_harm"
    CLICKBAIT_HARM = "clickbait_harm"
    ADDICTIVE_HARM = "addictive_harm"
    PHYSICAL_HARM = "physical_harm"


class JobStatus(str, Enum):
    """Status lifecycle for an asynchronous video moderation job."""

    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Stage1Result(BaseModel):
    """Stage 1 gatekeeper binary classification: Normal Content vs Harmful Video."""

    is_harmful: bool = Field(..., description="True if video is classified as Harmful Video")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Binary classification confidence score")


class Stage2Result(BaseModel):
    """Stage 2 fine-grained multi-label classification across the 7 Harm Categories."""

    predicted_categories: List[HarmCategory] = Field(
        default_factory=list,
        description="Identified harmful categories based on optimal decision thresholds",
    )
    category_scores: Dict[HarmCategory, float] = Field(
        default_factory=dict,
        description="Raw prediction probabilities per harm category",
    )


class ModerationResult(BaseModel):
    """Aggregated hierarchical moderation outcome for a video."""

    job_id: str
    video_filename: str
    status: JobStatus
    stage1: Optional[Stage1Result] = None
    stage2: Optional[Stage2Result] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
