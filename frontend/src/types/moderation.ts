/**
 * Canonical seven harm categories from GLOSSARY.md.
 */
export type HarmCategory =
  | 'information_harm'
  | 'sexual_harm'
  | 'psychological_harm'
  | 'hate_harassment_harm'
  | 'clickbait_harm'
  | 'addictive_harm'
  | 'physical_harm';

export type JobStatus = 'PENDING' | 'PROCESSING' | 'COMPLETED' | 'FAILED';

export interface Stage1Result {
  is_harmful: boolean;
  confidence: number;
}

export interface Stage2Result {
  predicted_categories: HarmCategory[];
  category_scores: Record<HarmCategory, number>;
}

export interface ModerationJob {
  job_id: string;
  video_filename: string;
  status: JobStatus;
  stage1: Stage1Result | null;
  stage2: Stage2Result | null;
  error_message: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface JobSubmission {
  job_id: string;
  video_filename: string;
  status: JobStatus;
  created_at: string;
}

export interface CategoryInfo {
  category: HarmCategory;
  display_name: string;
  description: string;
}
