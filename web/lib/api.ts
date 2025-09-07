export const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

export async function analyze(resume: string, job_description: string, model: 'local' | 'openai' = 'local') {
  const res = await fetch(`${API_BASE}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume, job_description, model })
  });
  if (!res.ok) throw new Error('Analyze request failed');
  return res.json();
}

export async function generateBullets(resume: string, job_description: string, tone: 'concise' | 'impact' = 'concise') {
  const res = await fetch(`${API_BASE}/bullets`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ resume, job_description, tone })
  });
  if (!res.ok) throw new Error('Bullets request failed');
  return res.json();
}
