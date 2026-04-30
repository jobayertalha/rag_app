# 🎯 CareerVector — AI Career Match

> A RAG-powered career advisor that tells you exactly how well your CV matches real AI/ML roles — built on FAISS vector search, HuggingFace embeddings, and Groq's LLaMA-3.3-70b.

**Live app →** [your-streamlit-link-here]  
**Built with** Python 3.10 · Streamlit · LangChain · FAISS · HuggingFace · Groq

---

## What it does

CareerVector runs on three paths from the home screen. Pick whichever fits where you are in your journey.

### 📄 CV Analyser
Upload your CV as a PDF and the app embeds it using `all-MiniLM-L6-v2` and runs FAISS cosine similarity search against a curated knowledge base built from the **Phitron AI/ML Job Market 2026** dataset — real job descriptions collected from LinkedIn, not hand-written summaries.

- Calibrated match % score — a student with 2–3 projects scores 35–55%, never an inflated 90%
- Salary grid across junior, mid, and senior levels sourced from real BD market data
- Skill gap breakdown — what's missing from your CV, why it matters, and how to learn it
- Resume recommendations in the format `Add X → unlocks Y role — reason`
- 3-step career path (0–12 months → 1–2 years → 3–5 years) calibrated to your actual CV level
- Multi-turn chat agent via LangChain + Groq for follow-up questions with bounded conversation history

### 🎯 JD Matching
Paste any job description and the entire analysis re-targets to that specific role.

- Semantic alignment scored between your CV and the target JD using FAISS vector search
- Personalised "why this fits you" explanation referencing your actual projects and certifications
- Runner-up role suggestion — the second-best match based on your profile
- Real salary ranges and market demand data from the Phitron 2026 LinkedIn dataset

### 🧩 Interest Quiz
No CV? No problem. For students and career explorers who want an honest signal before committing.

- 10 shuffled questions across ML interest, math comfort, coding confidence, and career preference
- Interest score out of 30 with three alignment bands: strong (21+), moderate (10–20), low (below 10)
- Category-level breakdown — ML interest, analytical thinking, hands-on interest, learning commitment
- 5 role suggestions tailored to your score band with salary ranges from real market data
- Tailored next steps for each alignment level

### ✨ Additional features
- Dark and light mode with a fully consistent blue theme across both
- Share your results directly to LinkedIn and Facebook in one click
- Sign out to reset the session and start fresh
- Sidebar navigation custom-implemented to work within Streamlit's native constraints

---

## Pipeline

```
User CV (PDF) + optional JD text
        │
        ▼
  PDF text extraction (pypdf)
        │
        ▼
  Embedding — HuggingFace all-MiniLM-L6-v2
        │
        ▼
  FAISS cosine similarity search
  against jd_knowledge_base.json embeddings
        │
        ▼
  Retrieve:
    ├── Top matched roles with calibrated % scores
    ├── Skill gaps (missing from CV)
    ├── Salary ranges (Junior / Mid / Senior)
    └── Market demand signal
        │
        ▼
  LLM — Groq API / LLaMA-3.3-70b-versatile
        │
        ▼
  Structured personalised advice
  (role match, career path, resume tips, chat)
```

---

## Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit (custom CSS, dark/light theme) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector search | FAISS (cosine similarity) |
| LLM | Groq API — LLaMA-3.3-70b-versatile |
| Agent framework | LangChain |
| PDF parsing | pypdf |
| Knowledge base | Custom JD dataset — Phitron AI/ML Job Market 2026 (LinkedIn) |
| Runtime | Python 3.10 |

---

## Project structure

```
careervector/
├── app.py                   # Streamlit UI — theme engine, all page renderers
├── agent.py                 # LangChain agent, Groq LLM, chat history management
├── rag.py                   # FAISS index loading, CV embedding, role retrieval, scoring
├── ingest.py                # One-time index builder from jd_knowledge_base.json
├── quiz.py                  # Interest quiz logic, scoring, alignment bands
├── jd_knowledge_base.json   # 12 real AI/ML roles — Phitron Job Market 2026
├── requirements.txt
├── runtime.txt              # python-3.10
└── .env.example
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/your-username/careervector.git
cd careervector
pip install -r requirements.txt
```

> First install downloads PyTorch CPU wheels (~500MB) and the HuggingFace embedding model (~80MB). Subsequent runs are fast.

### 2. Configure API keys

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here       # Required — free at console.groq.com
SERPAPI_API_KEY=your_serpapi_key_here     # Optional — enables live job search
```

Get your free Groq API key at https://console.groq.com

### 3. Build the FAISS index

```bash
python ingest.py
```

This embeds every JD text in `jd_knowledge_base.json` and saves the index to `./faiss_index/`. Only needs to run once — or again whenever you update the knowledge base.

### 4. Run the app

```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Customising the knowledge base

The JD knowledge base (`jd_knowledge_base.json`) contains 12 real AI/ML roles sourced from the Phitron AI/ML Job Market 2026 LinkedIn dataset. Each role has a `jd_text` field that gets embedded by FAISS.

To add your own job descriptions from LinkedIn or Indeed:

1. Open `jd_knowledge_base.json`
2. Find the relevant role or add a new entry
3. Paste the full JD text into the `jd_text` field
4. Run `python ingest.py` to rebuild the index

FAISS will now match CVs against your real JD language instead of the defaults.

Minimum role schema:

```json
{
  "title": "ML Engineer",
  "category": "Entry-level",
  "description": "Short description",
  "skills": ["Python", "PyTorch", "MLflow"],
  "salary_min": 30000,
  "salary_max": 60000,
  "market_demand": "High",
  "location": "Dhaka",
  "jd_text": "Full JD text pasted here..."
}
```

---

## Scoring calibration

Match percentages are deliberately calibrated — not inflated — to give honest signals:

| CV profile | Expected match range |
|---|---|
| Student, 2–3 projects, 1–2 certs, no work experience | 35–55% |
| Student with internship, 3+ projects, multiple certs | 55–72% |
| Professional with work experience and project portfolio | 70–88% |
| 90%+ | Only when CV explicitly matches advanced experience |

---

## Technical challenges solved

**1. FAISS retrieval latency on full CV text**  
Embedding entire CV documents caused slow retrieval. Fixed by isolating key CV signals — work experience, projects, and certifications — before embedding, significantly reducing noise and computation time.

**2. No top navbar in Streamlit**  
Streamlit does not support custom top navigation bars natively. Implemented a structured sidebar with section grouping and active-state indicators as the navigation layer.

**3. Default red focus outline in Streamlit**  
Streamlit's default input focus outline is red, conflicting with the blue theme. Overridden globally using injected CSS via `st.markdown()`.

**4. Dark/light mode consistency**  
All UI elements rendered correctly in dark mode but broke in light mode. Fixed by building a full theme dictionary (`get_theme()`) that injects all color values as CSS variables — both modes use explicit white/blue-family values rather than inheriting system defaults.

---

## Deployment

The app is deployed on Streamlit Community Cloud.

To deploy your own fork:

1. Push the repo to GitHub (ensure `jd_knowledge_base.json` is included)
2. Go to https://share.streamlit.io and connect your repo
3. Set `GROQ_API_KEY` in the Streamlit secrets manager
4. The FAISS index builds automatically on first run via `ingest.ensure_index()`

> Note: The first cold start takes ~2–3 minutes while the embedding model downloads. Subsequent loads are fast.


## Roadmap

- [ ] Multi-CV comparison mode
- [ ] Live job listings via SerpAPI integration
- [ ] Export results as PDF report
- [ ] Support for more regional job markets beyond BD

---

## Author

Built by  Talha Jobayer Zihan,final semester CSE student exploring the intersection of NLP, RAG pipelines, and practical career tooling.

LinkedIn: https://www.linkedin.com/in/talha-jobayer-696a74237/

GitHub: https://github.com/jobayertalha

---

```

---

Three things to replace before pasting: your Streamlit live link at the top, your name under Author, and the LinkedIn/GitHub URLs at the bottom. Everything else is ready to go as-is.
