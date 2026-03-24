"""
rag.py
Architecture:
  User CV + Target JD
      ↓
  Embedding (HuggingFace all-MiniLM-L6-v2, runs locally)
      ↓
  Vector Search (FAISS cosine similarity)
      ↓
  Retrieve:
     - similar job roles  (with % match scores)
     - required skills
     - career paths
      ↓
  Structured dict → injected into LLM prompt in agent.py
"""

import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "all-MiniLM-L6-v2"
_index = None   # module-level cache
_embedder = None

def load_roles(path="jd_knowledge_base.json"):
    # Try file first, fall back to embedded data
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    # Embedded fallback — no file needed
    import urllib.request
    raise FileNotFoundError(
        f"Missing: {path}\n"
        "Fix: Add data/jd_knowledge_base.json to your GitHub repo.\n"
        "Download it from the files shared in this chat."
    )




def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embedder


# ── Build FAISS index from JD knowledge base ──────────────────────
def build_index(roles):
    """
    Each JD entry becomes one embedded document.
    Embeds the jd_text field so FAISS matches CVs against real JD language.
    """
    embedder = get_embedder()
    texts, metadatas = [], []
    for r in roles:
        # Use jd_text as the primary embedding text for semantic matching
        text = r.get("jd_text", "")
        if not text:
            # Fallback: build text from structured fields
            text = (
                f"Job title: {r.get('title', r.get('role', ''))}. "
                f"Category: {r.get('category', '')}. "
                f"{r.get('description', '')} "
                f"Required skills: {', '.join(r.get('skills', []))}. "
                f"Career path: {r.get('career_path', '')}"
            )
        texts.append(text)
        metadatas.append(r)

    index = FAISS.from_texts(texts, embedder, metadatas=metadatas)
    os.makedirs("faiss_index", exist_ok=True)
    index.save_local("faiss_index")
    return index


def load_index():
    global _index
    if _index is None:
        embedder = get_embedder()
        _index = FAISS.load_local(
            "faiss_index", embedder,
            allow_dangerous_deserialization=True
        )
    return _index


def _score_to_pct(score: float) -> int:
    """
    FAISS L2 distance → 0-100 match percentage.
    Lower distance = more similar = higher %.
    Typical range: 0.0 (identical) to ~2.0 (unrelated).
    """
    return max(0, min(100, int((1 - score / 2.0) * 100)))


# ── CORE PIPELINE: CV + JD → Embed → FAISS → Retrieve ────────────
def retrieve_context(cv_text: str, jd_text: str = "", k: int = 5) -> dict:
    """
    Steps:
      1. Combine CV + optional target JD into one query string
      2. HuggingFace embedding of the combined query
      3. FAISS similarity_search_with_score against JD knowledge base
      4. Convert raw scores to match percentages
      5. Compute skill gaps and resume recommendations
      6. Return structured dict for LLM + UI
    """
    # Auto-build index if it doesn't exist
    if not os.path.exists("faiss_index"):
        print("Building FAISS index on first run...")
        roles = load_roles()
        build_index(roles)

    # 1. Combine inputs — JD biases search toward that role type
    query = cv_text
    if jd_text:
        query += f"\n\nTarget job the candidate is applying for:\n{jd_text}"

    # 2+3. Embed + FAISS search
    index = load_index()
    results = index.similarity_search_with_score(query, k=k)

    # 4. Structure results with match percentages
    all_matches = []
    for doc, score in results:
        role = dict(doc.metadata)
        role["match_pct"] = _score_to_pct(score)
        role["raw_score"] = round(float(score), 4)
        # Ensure role name key exists
        if "role" not in role:
            role["role"] = role.get("title", "Unknown Role")
        all_matches.append(role)

    all_matches.sort(key=lambda x: x["match_pct"], reverse=True)
    top_match = all_matches[0] if all_matches else {}

    # 5a. Skill gaps — skills in matched roles not found in CV
    cv_lower = cv_text.lower()
    required_skills = list(dict.fromkeys(
        sk for r in all_matches for sk in r.get("skills", [])
    ))
    skill_gaps = [sk for sk in required_skills if sk.lower() not in cv_lower]

    # 5b. Resume skills — gap skills ranked by frequency across matched roles
    skill_freq = {}
    for r in all_matches:
        for sk in r.get("skills", []):
            if sk.lower() not in cv_lower:
                skill_freq[sk] = skill_freq.get(sk, 0) + 1
    resume_skills = sorted(skill_freq, key=skill_freq.get, reverse=True)[:8]

    # 6. Format context string for LLM prompt
    blocks = []
    for r in all_matches[:4]:
        sal_min = r.get("salary_min", 0)
        sal_max = r.get("salary_max", 0)
        blocks.append(
            f"Role: {r.get('title', r.get('role', 'Unknown'))} — {r.get('category', '')} ({r['match_pct']}% match)\n"
            f"Description: {r.get('description', '')}\n"
            f"Required skills: {', '.join(r.get('skills', []))}\n"
            f"Salary range: ₨{sal_min:,}–₨{sal_max:,}/year\n"
            f"Market demand: {r.get('market_demand', '')}\n"
            f"Career path: {r.get('career_path', '')}\n"
            f"Why good fit: {r.get('why_good_fit', '')}\n"
            f"Next steps: {r.get('next_steps', '')}"
        )

    return {
        "top_match":       top_match,
        "top_role":        top_match,
        "all_matches":     all_matches,
        "similar_roles":   all_matches,
        "required_skills": required_skills,
        "skill_gaps":      skill_gaps[:8],
        "resume_skills":   resume_skills,
        "raw_context":     "\n\n---\n\n".join(blocks),
        "jd_provided":     bool(jd_text),
    }
