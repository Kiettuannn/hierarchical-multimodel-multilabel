from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.domain.models import HarmCategory, JobStatus


class JobSubmissionResponse(BaseModel):
    """Response returned upon submitting a video for moderation."""

    job_id: str = Field(..., description="Unique job identifier")
    video_filename: str = Field(..., description="Uploaded video filename")
    status: JobStatus = Field(..., description="Initial job status")
    created_at: datetime


class Stage1Response(BaseModel):
    """Gatekeeper binary outcome."""

    is_harmful: bool
    confidence: float


class Stage2Response(BaseModel):
    """Multilabel classification outcome."""

    predicted_categories: List[HarmCategory]
    category_scores: Dict[str, float]


class ModerationJobResponse(BaseModel):
    """Full detail of a moderation job."""

    job_id: str
    video_filename: str
    status: JobStatus
    stage1: Optional[Stage1Response] = None
    stage2: Optional[Stage2Response] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class CategoryInfoResponse(BaseModel):
    """Taxonomy category descriptor."""

    category: HarmCategory
    display_name: str
    description: str
