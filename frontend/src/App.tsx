import React, { useEffect, useState } from 'react';
import { VideoUploader } from './components/VideoUploader';
import { JobStatusView } from './components/JobStatusView';
import { ModerationJob, CategoryInfo } from './types/moderation';
import { submitVideo, getJobStatus, getCategories } from './services/api';
import { ShieldCheck, Info } from 'lucide-react';

export const App: React.FC = () => {
  const [currentJob, setCurrentJob] = useState<ModerationJob | null>(null);
  const [categories, setCategories] = useState<CategoryInfo[]>([]);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  useEffect(() => {
    getCategories()
      .then(setCategories)
      .catch((err) => console.error('Failed to load categories', err));
  }, []);

  // Polling effect for ongoing job
  useEffect(() => {
    if (!currentJob) return;
    if (currentJob.status !== 'PENDING' && currentJob.status !== 'PROCESSING') return;

    const intervalId = setInterval(async () => {
      try {
        const updated = await getJobStatus(currentJob.job_id);
        setCurrentJob(updated);
        if (updated.status === 'COMPLETED' || updated.status === 'FAILED') {
          clearInterval(intervalId);
        }
      } catch (err) {
        console.error('Polling error', err);
      }
    }, 1500);

    return () => clearInterval(intervalId);
  }, [currentJob?.job_id, currentJob?.status]);

  const handleUpload = async (file: File) => {
    setIsSubmitting(true);
    try {
      const submission = await submitVideo(file);
      // Initialize with pending status
      setCurrentJob({
        job_id: submission.job_id,
        video_filename: submission.video_filename,
        status: submission.status,
        stage1: null,
        stage2: null,
        error_message: null,
        created_at: submission.created_at,
        completed_at: null,
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col">
      {/* Top Navbar */}
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-9 h-9 rounded-lg bg-brand-600 flex items-center justify-center text-white">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-slate-800 leading-tight">HMM-Net</h1>
              <p className="text-xs text-slate-500">
                Hierarchical Multimodal Video Moderation System
              </p>
            </div>
          </div>
          <div className="text-xs text-slate-400">Production Scaffold v1.0</div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-4 py-8 flex-1 w-full grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Left Column: Upload & Live Analysis */}
        <div className="lg:col-span-2 space-y-6">
          <VideoUploader onUpload={handleUpload} isLoading={isSubmitting} />
          {currentJob && <JobStatusView job={currentJob} />}
        </div>

        {/* Right Column: Taxonomy Definitions Reference */}
        <div className="space-y-6">
          <div className="bg-white rounded-lg border border-slate-200 p-5 shadow-sm">
            <div className="flex items-center gap-2 mb-3 text-slate-700">
              <Info className="w-4 h-4 text-brand-600" />
              <h3 className="font-semibold text-sm">Harm Taxonomy Definition</h3>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Vietnamese short-form video policy taxonomy extending the MetaHarm framework across 7
              fine-grained categories.
            </p>
            <div className="space-y-3">
              {categories.map((cat) => (
                <div key={cat.category} className="border-t border-slate-100 pt-2 text-xs">
                  <div className="font-semibold text-slate-800">{cat.display_name}</div>
                  <div className="text-slate-500 mt-0.5">{cat.description}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-4 text-center text-xs text-slate-400">
        HMM-Net Multimodal Cascade — Vision (SigLIP2, Gemma4), Audio (CLAP, Demucs), ASR
        (ChunkFormer), OCR (PaddleOCR, VietOCR)
      </footer>
    </div>
  );
};
