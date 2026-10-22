from typing import List
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.api.deps import get_moderation_service
from app.api.v1.schemas.moderation import (
    CategoryInfoResponse,
    JobSubmissionResponse,
    ModerationJobResponse,
)
from app.domain.models import HarmCategory
from app.services.moderation_service import ModerationService

router = APIRouter(prefix="/moderation", tags=["Video Moderation"])

TAXONOMY_DESCRIPTIONS = {
    HarmCategory.INFORMATION_HARM: (
        "Information Harms",
        "Content that spreads misinformation, disinformation, conspiracy theories, or unverified claims.",
    ),
    HarmCategory.HATE_HARASSMENT_HARM: (
        "Hate and Harassment Harms",
        "Insults, abusive language, identity attacks, or hate speech targeting individuals or groups.",
    ),
    HarmCategory.ADDICTIVE_HARM: (
        "Addictive Harms",
        "Content promoting addictive behaviors including online gaming addiction, substance use, or gambling.",
    ),
    HarmCategory.CLICKBAIT_HARM: (
        "Clickbait Harms",
        "Sensational claims, fraudulent schemes, rumors, or manipulative engagement tactics.",
    ),
    HarmCategory.PHYSICAL_HARM: (
        "Physical Harms",
        "Content depicting or promoting self-harm, dangerous challenges, or graphic violence.",
    ),
    HarmCategory.SEXUAL_HARM: (
        "Sexual Harms",
        "Sexually explicit material, nudity, sexual exploitation, or sexually suggestive acts.",
    ),
    HarmCategory.PSYCHOLOGICAL_HARM: (
        "Psychological Harms",
        "Content causing distress or fear, including horror, supernatural, or depressing material.",
    ),
}


@router.post(
    "/jobs",
    response_model=JobSubmissionResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a video for hierarchical moderation",
)
async def submit_video_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Target MP4 video file"),
    service: ModerationService = Depends(get_moderation_service),
):
    """Upload a video and trigger asynchronous hierarchical moderation inference."""
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload an MP4 or supported video file.",
        )

    job_id, stored_path = await service.submit_job(
        filename=file.filename,
        file_stream=file.file,
    )

    # Dispatch inference pipeline to background worker
    background_tasks.add_task(
        service.execute_job,
        job_id=job_id,
        video_path=stored_path,
        filename=file.filename,
    )

    job = service.get_job(job_id)
    return JobSubmissionResponse(
        job_id=job.job_id,
        video_filename=job.video_filename,
        status=job.status,
        created_at=job.created_at,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=ModerationJobResponse,
    summary="Get status and results of a moderation job",
)
async def get_job_status(
    job_id: str,
    service: ModerationService = Depends(get_moderation_service),
):
    """Retrieve moderation status and hierarchical predictions for a specific job."""
    job = service.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID '{job_id}' not found.",
        )
    return job


@router.get(
    "/jobs",
    response_model=List[ModerationJobResponse],
    summary="List all moderation jobs",
)
async def list_jobs(service: ModerationService = Depends(get_moderation_service)):
    """Return all recorded moderation jobs."""
    return service.list_jobs()


@router.get(
    "/categories",
    response_model=List[CategoryInfoResponse],
    summary="List all canonical Harm Categories",
)
async def list_categories():
    """Return the predefined seven Harm Categories with their canonical definitions."""
    return [
        CategoryInfoResponse(
            category=cat,
            display_name=display,
            description=desc,
        )
        for cat, (display, desc) in TAXONOMY_DESCRIPTIONS.items()
    ]
