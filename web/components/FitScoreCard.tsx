import React from 'react';

type Props = { score: number };

export default function FitScoreCard({ score }: Props) {
  const clamped = Math.max(0, Math.min(100, score));
  const tier =
    clamped >= 85 ? 'bg-green-100 text-green-800 border-green-200' :
    clamped >= 70 ? 'bg-emerald-100 text-emerald-800 border-emerald-200' :
    clamped >= 50 ? 'bg-yellow-100 text-yellow-800 border-yellow-200' :
    clamped >= 30 ? 'bg-orange-100 text-orange-800 border-orange-200' :
                    'bg-red-100 text-red-800 border-red-200';

  return (
    <div className="border rounded-2xl p-6 shadow-sm bg-white">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Overall Fit Score</h2>
        <span className={"px-3 py-1 rounded-full text-sm border " + tier}>
          {clamped}%
        </span>
      </div>
      <div className="w-full h-3 bg-gray-200 rounded-full overflow-hidden">
        <div
          className="h-3 bg-indigo-600"
          style={{ width: clamped + '%' }}
          aria-label="Fit score progress"
        />
      </div>
      <p className="text-sm text-gray-600 mt-3">
        This score blends semantic similarity (70%) and keyword coverage (30%).
      </p>
    </div>
  );
}
