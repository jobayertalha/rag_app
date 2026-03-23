# ⚡ AI Career Match

A Streamlit app that matches your CV to AI/data roles using FAISS vector search + LLM analysis.

## Pipeline

```
User CV + JD
    ↓
Embedding (HuggingFace all-MiniLM-L6-v2)
    ↓
Vector Search (FAISS cosine similarity)
    ↓
Retrieve:
   - Similar job roles (with % match scores)
   - Required skills
   - Career paths
    ↓
LLM (Groq / LLaMA-3.3-70b)
    ↓
Personalized Advice
```

## Setup

### 1. Clone & install
```bash
git clone <your-repo>
cd ai-career-match
pip install -r requirements.txt
```

### 2. Set up API keys
```bash
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=your_groq_api_key_here       # Required — get free at console.groq.com
SERPAPI_API_KEY=your_serpapi_key_here     # Optional — enables live job search
```

**Get Groq API key (free):** https://console.groq.com
**Get SerpAPI key (optional):** https://serpapi.com

### 3. Build the FAISS index
```bash
python ingest.py
```
This downloads the ~80MB embedding model on first run and builds `./faiss_index/`.

### 4. Run the app
```bash
streamlit run app.py
```

Open http://localhost:8501

---

## Add your own Job Descriptions

The knowledge base is `data/jd_knowledge_base.json`. Each role has a `jd_text` field.

To add a real JD from LinkedIn/Indeed:
1. Open `data/jd_knowledge_base.json`
2. Find the relevant role (e.g. `"AI / LLM Engineer"`)
3. Replace the `jd_text` value with the real JD text you copied
4. Run `python ingest.py` to rebuild the index

FAISS will now match CVs against real JD language.

---

## Project Structure

```
ai-career-match/
├── app.py                    # Streamlit UI (dark theme)
├── agent.py                  # LangChain agent + Groq LLM
├── rag.py                    # FAISS vector search pipeline
├── ingest.py                 # Build FAISS index from JDs
├── requirements.txt
├── .env.example
└── data/
    └── jd_knowledge_base.json   # JD knowledge base (8 roles)
```

## UI Features

- **Welcome screen** — asks for your name first
- **Hero card** — top matched role + big % match score
- **Full breakdown** — all 4 matched roles with % and skill chips
- **Why X is Right for You** — LLM-personalised explanation
- **Next Steps** — actionable learning path
- **Salary grid** — Junior / Mid / Senior ranges
- **Market Demand** — current demand indicator
- **Skill Gaps** — what's missing + how to learn it
- **Resume Recommendations** — specific additions to make
- **Career Path** — step-by-step progression
- **Runner-up** — second best match
- **Chat** — ask follow-up questions, search live jobs
