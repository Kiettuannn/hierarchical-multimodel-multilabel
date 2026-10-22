import React, { useRef, useState } from 'react';
import { Upload, FileVideo, AlertCircle } from 'lucide-react';

interface VideoUploaderProps {
  onUpload: (file: File) => Promise<void>;
  isLoading: boolean;
}

export const VideoUploader: React.FC<VideoUploaderProps> = ({ onUpload, isLoading }) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setErrorMessage(null);
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      if (!file.name.match(/\.(mp4|mov|avi|mkv)$/i)) {
        setErrorMessage('Please select a valid video format (.mp4, .mov, .avi, .mkv).');
        return;
      }
      setSelectedFile(file);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFile) return;
    try {
      await onUpload(selectedFile);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = '';
    } catch (err: any) {
      setErrorMessage(err.message || 'Error uploading video');
    }
  };

  return (
    <div className="bg-white rounded-lg border border-slate-200 p-6 shadow-sm">
      <h2 className="text-lg font-semibold text-slate-800 mb-2">Upload Video for Moderation</h2>
      <p className="text-sm text-slate-500 mb-4">
        Supports MP4 video formats. Videos are analyzed through the 4-branch multimodal cascade
        (Vision, ASR, OCR, Audio) to evaluate policy compliance.
      </p>

      {errorMessage && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-red-700 text-sm flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{errorMessage}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed border-slate-300 hover:border-slate-400 rounded-lg p-6 text-center cursor-pointer transition-colors"
        >
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="video/*"
            className="hidden"
          />
          <div className="flex flex-col items-center justify-center space-y-2">
            {selectedFile ? (
              <>
                <FileVideo className="w-10 h-10 text-brand-600" />
                <span className="text-sm font-medium text-slate-800">{selectedFile.name}</span>
                <span className="text-xs text-slate-500">
                  {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                </span>
              </>
            ) : (
              <>
                <Upload className="w-10 h-10 text-slate-400" />
                <span className="text-sm text-slate-600">
                  Click or drag and drop video file here
                </span>
                <span className="text-xs text-slate-400">MP4 format recommended</span>
              </>
            )}
          </div>
        </div>

        <button
          type="submit"
          disabled={!selectedFile || isLoading}
          className={`w-full py-2.5 px-4 rounded font-medium text-white transition-colors ${
            !selectedFile || isLoading
              ? 'bg-slate-400 cursor-not-allowed'
              : 'bg-brand-600 hover:bg-brand-700'
          }`}
        >
          {isLoading ? 'Analyzing Video...' : 'Start Moderation'}
        </button>
      </form>
    </div>
  );
};
