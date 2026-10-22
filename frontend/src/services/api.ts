import { CategoryInfo, JobSubmission, ModerationJob } from '../types/moderation';

const API_BASE = '/api/v1';

export async function submitVideo(file: File): Promise<JobSubmission> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE}/moderation/jobs`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || 'Failed to submit video for moderation');
  }

  return response.json();
}

export async function getJobStatus(jobId: string): Promise<ModerationJob> {
  const response = await fetch(`${API_BASE}/moderation/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error(`Failed to fetch job status for ${jobId}`);
  }
  return response.json();
}

export async function getCategories(): Promise<CategoryInfo[]> {
  const response = await fetch(`${API_BASE}/moderation/categories`);
  if (!response.ok) {
    throw new Error('Failed to fetch taxonomy categories');
  }
  return response.json();
}

export async function listRecentJobs(): Promise<ModerationJob[]> {
  const response = await fetch(`${API_BASE}/moderation/jobs`);
  if (!response.ok) {
    throw new Error('Failed to fetch jobs list');
  }
  return response.json();
}
