# Resume ↔ JD Matcher (AI)

A FastAPI app that analyzes your resume against any job description using NLP and vector embeddings.  
It returns a 0–100 fit score, highlights missing skills, and drafts ATS-friendly, quant-driven resume bullets tailored to the role.

## Stack
- FastAPI (Python backend)
- sentence-transformers (embeddings)
- FAISS / pgvector (optional vector store)
- Next.js + Tailwind (frontend)

## Setup

### Backend
```bash
cd app
pip install fastapi uvicorn sentence-transformers numpy
uvicorn main:app --reload
```

### Frontend (Next.js)
```bash
cd web
npm install
npm run dev
```


## Environment

- Backend: copy `.env.example` to `.env` and set values as needed.
- Frontend: optionally set `NEXT_PUBLIC_API_BASE` (defaults to `http://localhost:8000`).

