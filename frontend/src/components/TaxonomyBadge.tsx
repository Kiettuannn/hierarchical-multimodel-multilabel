import React from 'react';
import { HarmCategory } from '../types/moderation';

interface TaxonomyBadgeProps {
  category: HarmCategory;
  score?: number;
  highlighted?: boolean;
}

const CATEGORY_NAMES: Record<HarmCategory, string> = {
  information_harm: 'Information Harms',
  sexual_harm: 'Sexual Harms',
  psychological_harm: 'Psychological Harms',
  hate_harassment_harm: 'Hate & Harassment Harms',
  clickbait_harm: 'Clickbait Harms',
  addictive_harm: 'Addictive Harms',
  physical_harm: 'Physical Harms',
};

export const TaxonomyBadge: React.FC<TaxonomyBadgeProps> = ({
  category,
  score,
  highlighted = false,
}) => {
  const name = CATEGORY_NAMES[category] || category;

  return (
    <div
      className={`flex items-center justify-between px-3 py-2 rounded border text-sm font-medium transition-all ${
        highlighted
          ? 'bg-rose-50 border-rose-300 text-rose-800'
          : 'bg-slate-50 border-slate-200 text-slate-700'
      }`}
    >
      <span>{name}</span>
      {score !== undefined && (
        <span
          className={`ml-2 px-2 py-0.5 rounded text-xs ${
            highlighted ? 'bg-rose-200 text-rose-900' : 'bg-slate-200 text-slate-800'
          }`}
        >
          {(score * 100).toFixed(1)}%
        </span>
      )}
    </div>
  );
};
