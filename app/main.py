# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import numpy as np

# -----------------------------
# Env & configuration
# -----------------------------
load_dotenv()

USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
OPENAI_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

ALLOWED_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN")
if FRONTEND_ORIGIN:
    ALLOWED_ORIGINS.append(FRONTEND_ORIGIN)

CORS_ALLOW_ALL = os.getenv("CORS_ALLOW_ALL", "false").lower() == "true"

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Resume↔JD Matcher")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if CORS_ALLOW_ALL else ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Embeddings
# -----------------------------
_local_model = None  # lazy-loaded


def _normalize(vectors: np.ndarray) -> np.ndarray:
    # L2 normalize rows
    denom = np.linalg.norm(vectors, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return vectors / denom


def _embed_local(texts: List[str]) -> np.ndarray:
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer
        _local_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    vecs = _local_model.encode(texts, normalize_embeddings=False)
    return _normalize(np.array(vecs, dtype=np.float32))


def _embed_openai(texts: List[str]) -> np.ndarray:
    try:
        from openai import OpenAI  # pip install openai
    except Exception as e:
        raise RuntimeError(
            "OpenAI SDK not installed. Run `pip install openai` "
            "or set USE_OPENAI=false."
        ) from e

    client = OpenAI()
    # Create in batches to be safe (simple single batch here)
    resp = client.embeddings.create(model=OPENAI_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return _normalize(vecs)


def embed_texts(texts: List[str]) -> np.ndarray:
    """Return NxD normalized embeddings."""
    if USE_OPENAI:
        return _embed_openai(texts)
    return _embed_local(texts)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# -----------------------------
# NLP helpers
# -----------------------------
def chunk(text: str, max_chars: int = 1200) -> List[str]:
    import re
    sentences = re.split(r'(?<=[\.\!\?])\s+', text.strip())
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 <= max_chars:
            cur += (" " if cur else "") + s
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks or [text]


def extract_skills(text: str) -> List[str]:
    # Minimal curated list; extend as needed
    candidates = [
        "python", "java", "c++", "typescript", "react", "next.js", "node",
        "fastapi", "docker", "kubernetes", "aws", "gcp", "azure", "terraform",
        "sql", "postgres", "mongodb", "firebase", "tailwind", "ci/cd",
        "pytest", "jest"
    ]
    lower = text.lower()
    return sorted({c for c in candidates if c in lower})


def coverage(jd_terms: List[str], resume_terms: List[str]) -> float:
    if not jd_terms:
        return 0.0
    hits = sum(1 for t in jd_terms if t in resume_terms)
    return hits / float(len(jd_terms))


def fit_score(resume: str, jd: str) -> Dict[str, Any]:
    r_chunks = chunk(resume)
    j_chunks = chunk(jd)

    r_vec = embed_texts(r_chunks).mean(axis=0)
    j_vec = embed_texts(j_chunks).mean(axis=0)
    sem = cosine(r_vec, j_vec)

    jd_terms = extract_skills(jd)
    r_terms = extract_skills(resume)
    cov = coverage(jd_terms, r_terms)

    score = 100.0 * (0.7 * sem + 0.3 * cov)

    gaps = [
        {"term": t, "suggestion": f"Add evidence of {t} (project, metric, impact)."}
        for t in jd_terms if t not in r_terms
    ]
    overlap = [{"term": t, "evidence": "found in resume"} for t in jd_terms if t in r_terms]

    return {
        "fit_score": round(score, 1),
        "top_overlap": overlap[:12],
        "gaps": gaps[:12],
        "explanations": {
            "methods": ["embedding cosine", "keyword coverage"],
            "weights": {"semantic": 0.7, "keywords": 0.3},
        },
    }


# -----------------------------
# Schemas
# -----------------------------
class AnalyzeIn(BaseModel):
    resume: str
    job_description: str
    # Optional request override; if "openai", flips USE_OPENAI for this process
    model: str | None = None


class BulletsIn(BaseModel):
    resume: str
    job_description: str
    tone: str = "concise"  # 'concise' | 'impact'


# -----------------------------
# Routes
# -----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(body: AnalyzeIn):
    global USE_OPENAI
    # Allow request-level override for convenience
    if body.model:
        USE_OPENAI = (body.model.lower() == "openai")

    result = fit_score(body.resume, body.job_description)

    # Placeholder bullets; you can replace with an LLM later
    result["tailored_bullets"] = [
        "Delivered feature X improving load time by 28% using React and Node.",
        "Automated CI pipeline with Docker and GitHub Actions, cutting deploy time by 60%.",
        "Optimized Postgres queries reducing P95 latency from 420ms to 180ms.",
    ]
    return result


@app.post("/bullets")
def bullets(body: BulletsIn):
    # Simple heuristic using overlaps/gaps; swap with LLM when ready
    analysis = fit_score(body.resume, body.job_description)
    matched = [o["term"] for o in analysis.get("top_overlap", [])][:3]
    gaps = [g["term"] for g in analysis.get("gaps", [])][:2]
    tone_prefix = "Delivered" if body.tone == "impact" else "Implemented"

    bullets_out: List[str] = []
    if matched:
        bullets_out.append(
            f"{tone_prefix} {', '.join(matched)} solutions aligned to role requirements; "
            "improved reliability and velocity while meeting stakeholder goals."
        )
    else:
        bullets_out.append(
            f"{tone_prefix} features mapped directly to JD responsibilities; "
            "aligned deliverables with acceptance criteria and timelines."
        )

    bullets_out.append(
        "Quantified impact with KPIs (latency, conversion, adoption); collaborated cross-functionally "
        "to de-risk launches and improve DX."
    )

    if gaps:
        bullets_out.append(
            f"Up-skilled in {', '.join(gaps)} via focused projects/labs; applied learnings to code quality "
            "and infrastructure readiness."
        )
    else:
        bullets_out.append(
            "Applied best practices in testing, CI/CD, and observability to maintain high code quality."
        )

    return {"bullets": bullets_out}
