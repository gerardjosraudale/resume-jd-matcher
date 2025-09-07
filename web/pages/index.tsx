import React, { useState } from 'react';
import FitScoreCard from '../components/FitScoreCard';
import Tag from '../components/Tag';
import { analyze, generateBullets } from '../lib/api';

export default function Home() {
  const [resume, setResume] = useState('');
  const [jd, setJd] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [regenLoading, setRegenLoading] = useState(false);

  const onAnalyze = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await analyze(resume, jd, 'local');
      setResult(data);
    } catch (e:any) {
      setError(e?.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const onRegenerateBullets = async (tone: 'concise' | 'impact' = 'concise') => {
    setRegenLoading(true);
    setError(null);
    try {
      const data = await generateBullets(resume, jd, tone);
      setResult((prev:any) => ({ ...(prev || {}), tailored_bullets: data.bullets || [] }));
    } catch (e:any) {
      setError(e?.message || 'Something went wrong');
    } finally {
      setRegenLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-8 font-sans bg-gray-50">
      <div className="max-w-4xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold">Resume ↔ JD Matcher (AI)</h1>
          <p className="text-gray-600 mt-1">
            Paste your resume and a job description. Get a fit score, gap highlights, and tailored bullets.
          </p>
        </header>

        <div className="grid md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="block text-sm font-medium mb-1">Resume</label>
            <textarea
              className="w-full p-3 border rounded-xl bg-white"
              rows={10}
              placeholder="Paste your resume text…"
              value={resume}
              onChange={e => setResume(e.target.value)}
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Job Description</label>
            <textarea
              className="w-full p-3 border rounded-xl bg-white"
              rows={10}
              placeholder="Paste the job description…"
              value={jd}
              onChange={e => setJd(e.target.value)}
            />
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            className="px-5 py-2.5 rounded-xl bg-indigo-600 text-white font-medium disabled:opacity-60"
            onClick={onAnalyze}
            disabled={loading || !resume || !jd}
          >
            {loading ? 'Analyzing…' : 'Analyze'}
          </button>

          <button
            className="px-4 py-2 rounded-xl border bg-white disabled:opacity-60"
            onClick={() => onRegenerateBullets('concise')}
            disabled={regenLoading || !resume || !jd}
            title="Generate concise bullets"
          >
            {regenLoading ? 'Generating…' : 'Regenerate bullets'}
          </button>

          <button
            className="px-4 py-2 rounded-xl border bg-white disabled:opacity-60"
            onClick={() => onRegenerateBullets('impact')}
            disabled={regenLoading || !resume || !jd}
            title="Generate impact‑focused bullets"
          >
            {regenLoading ? 'Generating…' : 'Regenerate (impact tone)'}
          </button>
        </div>

        {error && <p className="mt-4 text-sm text-red-600">{error}</p>}

        {result && (
          <section className="mt-8 space-y-6">
            <FitScoreCard score={result.fit_score ?? 0} />

            <div className="grid md:grid-cols-2 gap-6">
              <div className="border rounded-2xl p-5 bg-white">
                <h3 className="font-semibold mb-2">Matched Terms</h3>
                <div>
                  {result.top_overlap?.length
                    ? result.top_overlap.map((t:any, i:number) => (
                        <Tag key={i}>{t.term}</Tag>
                      ))
                    : <p className="text-sm text-gray-500">No matched terms detected.</p>
                  }
                </div>
              </div>

              <div className="border rounded-2xl p-5 bg-white">
                <h3 className="font-semibold mb-2">Gaps</h3>
                <ul className="list-disc pl-5 space-y-1">
                  {result.gaps?.length
                    ? result.gaps.map((g:any, i:number) => (
                        <li key={i} className="text-sm">
                          <span className="font-medium">{g.term}:</span> {g.suggestion}
                        </li>
                      ))
                    : <p className="text-sm text-gray-500">No gaps detected.</p>
                  }
                </ul>
              </div>
            </div>

            <div className="border rounded-2xl p-5 bg-white">
              <h3 className="font-semibold mb-3">Tailored Bullets</h3>
              {result.tailored_bullets?.length ? (
                <div className="space-y-2">
                  {result.tailored_bullets.map((b:string, i:number) => (
                    <div key={i} className="p-3 bg-gray-50 rounded-lg">{b}</div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-gray-500">No bullets generated.</p>
              )}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
