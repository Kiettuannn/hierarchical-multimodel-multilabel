import React from 'react';
import { ModerationJob, HarmCategory } from '../types/moderation';
import { TaxonomyBadge } from './TaxonomyBadge';
import { CheckCircle2, AlertTriangle, Clock, XCircle } from 'lucide-react';

interface JobStatusViewProps {
  job: ModerationJob;
}

const ALL_CATEGORIES: HarmCategory[] = [
  'information_harm',
  'sexual_harm',
  'psychological_harm',
  'hate_harassment_harm',
  'clickbait_harm',
  'addictive_harm',
  'physical_harm',
];

export const JobStatusView: React.FC<JobStatusViewProps> = ({ job }) => {
  const isProcessing = job.status === 'PENDING' || job.status === 'PROCESSING';

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm space-y-6">
      <div className="flex items-center justify-between border-b border-slate-100 pb-4">
        <div>
          <h3 className="font-semibold text-slate-800 text-lg">Moderation Analysis</h3>
          <p className="text-xs text-slate-400 mt-0.5">Job ID: {job.job_id}</p>
        </div>
        <div className="flex items-center gap-2">
          {job.status === 'COMPLETED' && (
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-800 flex items-center gap-1">
              <CheckCircle2 className="w-3.5 h-3.5" /> COMPLETED
            </span>
          )}
          {isProcessing && (
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-amber-100 text-amber-800 flex items-center gap-1">
              <Clock className="w-3.5 h-3.5 animate-spin" /> {job.status}
            </span>
          )}
          {job.status === 'FAILED' && (
            <span className="px-3 py-1 rounded-full text-xs font-semibold bg-red-100 text-red-800 flex items-center gap-1">
              <XCircle className="w-3.5 h-3.5" /> FAILED
            </span>
          )}
        </div>
      </div>

      <div className="text-sm text-slate-600">
        <span className="font-medium text-slate-700">Filename:</span> {job.video_filename}
      </div>

      {isProcessing && (
        <div className="py-8 text-center text-slate-500 space-y-2">
          <div className="w-8 h-8 border-4 border-brand-500 border-t-transparent rounded-full animate-spin mx-auto"></div>
          <p className="text-sm">Running multimodal extraction across Vision, ASR, OCR, and Audio...</p>
        </div>
      )}

      {job.status === 'FAILED' && (
        <div className="p-4 bg-red-50 border border-red-200 rounded text-red-700 text-sm">
          {job.error_message || 'An error occurred during video analysis.'}
        </div>
      )}

      {job.status === 'COMPLETED' && job.stage1 && (
        <div className="space-y-6">
          {/* Stage 1 Gatekeeper */}
          <div
            className={`p-4 rounded-lg border ${
              job.stage1.is_harmful
                ? 'bg-rose-50 border-rose-200 text-rose-900'
                : 'bg-emerald-50 border-emerald-200 text-emerald-900'
            }`}
          >
            <div className="flex items-center gap-3">
              {job.stage1.is_harmful ? (
                <AlertTriangle className="w-6 h-6 text-rose-600 shrink-0" />
              ) : (
                <CheckCircle2 className="w-6 h-6 text-emerald-600 shrink-0" />
              )}
              <div>
                <h4 className="font-semibold text-base">
                  Stage 1 Gatekeeper: {job.stage1.is_harmful ? 'Harmful Video' : 'Normal Content'}
                </h4>
                <p className="text-sm opacity-90">
                  Confidence score: {(job.stage1.confidence * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* Stage 2 Fine-Grained Harm Taxonomy */}
          <div>
            <h4 className="text-sm font-semibold text-slate-700 mb-3">
              Stage 2: Taxonomy Violations Classification (7 Harm Categories)
            </h4>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
              {ALL_CATEGORIES.map((cat) => {
                const isViolation =
                  job.stage2?.predicted_categories?.includes(cat) ?? false;
                const score = job.stage2?.category_scores?.[cat];
                return (
                  <TaxonomyBadge
                    key={cat}
                    category={cat}
                    score={score}
                    highlighted={isViolation}
                  />
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
