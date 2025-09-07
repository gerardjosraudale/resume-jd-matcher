# 📄 Resume ↔ JD Matcher (AI)

AI-powered tool that matches a resume against a job description using **NLP + embeddings**.  
It provides a **fit score (0–100)**, highlights **skill gaps**, and generates **tailored resume bullets** to strengthen applications.  

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)  
![Next.js](https://img.shields.io/badge/Next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white)  
![TailwindCSS](https://img.shields.io/badge/Tailwind-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white)  
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)  

---

## ✨ Features
- **Resume ↔ JD Fit Score** — blends semantic similarity + keyword coverage.  
- **Gap Analysis** — highlights missing technical skills with suggestions.  
- **Tailored Bullets** — generates ATS-friendly, STAR-style bullet points.  
- **UI Dashboard** — clean Next.js + Tailwind interface with Fit Score card.  
- **Two Modes** — concise or impact-focused bullets.  
- **Extendable** — ready for OpenAI embeddings, pgvector, or LLM upgrades.  

---

## 🖥️ Demo

![screenshot](docs/demo-screenshot.png)  
*(Example Fit Score card + tailored bullets UI)*

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, sentence-transformers, NumPy, python-dotenv  
- **Frontend**: Next.js (TypeScript), TailwindCSS, React  
- **APIs**: REST endpoints `/analyze` and `/bullets`  
- **Optional**: OpenAI embeddings + pgvector for advanced use  

---

## 🚀 Getting Started

### 1. Clone
```bash
git clone https://github.com/gerardjosraudale/resume-jd-matcher.git
cd resume-jd-matcher


### 2. Backend

cd app
pip install -r ../requirements.txt
uvicorn main:app --reload

Visit: http://localhost:8000/docs for API docs.
Health check: http://localhost:8000/health

### 3. Frontend
cd web
npm install
npm run dev

Visit: http://localhost:3000
⚙️ Environment
Copy .env.example → .env and update:
USE_OPENAI=false
OPENAI_API_KEY=sk-...
FRONTEND_ORIGIN=http://localhost:3000
For frontend, you can set:
NEXT_PUBLIC_API_BASE=http://localhost:8000
📂 Project Structure
resume-jd-matcher/
├── app/                # FastAPI backend
│   └── main.py
├── web/                # Next.js frontend
│   ├── pages/
│   │   ├── index.tsx
│   │   └── _app.tsx
│   ├── components/
│   │   ├── FitScoreCard.tsx
│   │   └── Tag.tsx
│   ├── lib/api.ts
│   └── styles/globals.css
├── requirements.txt
├── .env.example
├── README.md
📈 Future Improvements
Cover letter generator
Multi-JD ranking
PDF export of analysis
GitHub Actions for CI/CD
pgvector / FAISS vector store for large-scale matching
👤 Author
Josue Raudales
LinkedIn • GitHub