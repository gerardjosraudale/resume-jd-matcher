from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()
USE_OPENAI = os.getenv('USE_OPENAI', 'false').lower() == 'true'

def embed_texts(texts: List[str]) -> np.ndarray:
    if USE_OPENAI:
        raise NotImplementedError("OpenAI embeddings not implemented in demo.")
    else:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return model.encode(texts, normalize_embeddings=True)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def chunk(text: str, max_tokens: int = 256) -> List[str]:
    import re
    sentences = re.split(r'(?<=[\.!?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) < 1200:
            cur += (" " if cur else "") + s
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return chunks or [text]

def extract_skills(text: str) -> List[str]:
    candidates = [
        "python","java","c++","typescript","react","node","fastapi","docker",
        "kubernetes","aws","gcp","azure","terraform","sql","postgres","mongodb",
        "firebase","tailwind","next.js","ci/cd","pytest","jest"
    ]
    lower = text.lower()
    return sorted({c for c in candidates if c in lower})

def coverage(jd_terms: List[str], resume_terms: List[str]) -> float:
    if not jd_terms: return 0.0
    hits = sum(1 for t in jd_terms if t in resume_terms)
    return hits / len(jd_terms)

def fit_score(resume: str, jd: str) -> Dict[str, Any]:
    r_chunks = chunk(resume)
    j_chunks = chunk(jd)
    r_vec = embed_texts([" ".join(r_chunks)]).mean(axis=0)
    j_vec = embed_texts([" ".join(j_chunks)]).mean(axis=0)
    sem = cosine(r_vec, j_vec)

    jd_terms = extract_skills(jd)
    r_terms  = extract_skills(resume)
    cov = coverage(jd_terms, r_terms)

    score = 100.0 * (0.7*sem + 0.3*cov)

    gaps = [{"term": t, "suggestion": f"Add evidence of {t} (project, metric, impact)."} 
            for t in jd_terms if t not in r_terms]
    overlap = [{"term": t, "evidence": "found in resume"} for t in jd_terms if t in r_terms]

    return {
        "fit_score": round(score, 1),
        "top_overlap": overlap[:12],
        "gaps": gaps[:12],
        "explanations": {"methods":["embedding cosine","keyword coverage"],
                         "weights":{"semantic":0.7,"keywords":0.3}}
    }

app = FastAPI(title="Resume↔JD Matcher")

# Enable CORS for local frontend dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeIn(BaseModel):
    resume: str
    job_description: str
    model: str = "local"

@app.post("/analyze")
def analyze(body: AnalyzeIn):
    global USE_OPENAI
    USE_OPENAI = (body.model == "openai")
    result = fit_score(body.resume, body.job_description)
    bullets = [
        "Delivered feature X improving load time by 28% using React and Node.",
        "Automated CI pipeline with Docker and GitHub Actions, cutting deploy time by 60%.",
        "Optimized Postgres queries reducing P95 latency from 420ms to 180ms."
    ]
    result["tailored_bullets"] = bullets
    return result

@app.get("/health")
def health():
    return {"status": "ok"}

class BulletsIn(BaseModel):
    resume: str
    job_description: str
    tone: str = "concise"  # or "impact"

@app.post("/bullets")
def bullets(body: BulletsIn):
    # Placeholder implementation, extend with LLM call later
    return {
        "bullets": [
            "Implemented REST API with FastAPI handling 10k+ daily requests.",
            "Optimized React frontend performance, reducing bundle size by 35%.",
            "Deployed app with Docker + CI/CD pipelines on AWS."
        ]
    }
