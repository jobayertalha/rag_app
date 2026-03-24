"""
rag.py — Fast focused extraction.
Only extracts skills, projects, interests from CV.
Only extracts requirements from JD.
This reduces noise and speeds up matching significantly.
"""

import json
import os
import re
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "all-MiniLM-L6-v2"
_index = None
_embedder = None


def load_roles(path="jd_knowledge_base.json"):
    # Try root first, then data/ subfolder
    for p in [path, f"data/{path}", "data/jd_knowledge_base.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Cannot find jd_knowledge_base.json. "
        f"Make sure it is committed to the root of your GitHub repo."
    )


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embedder


def build_index(roles):
    embedder = get_embedder()
    texts, metadatas = [], []
    for r in roles:
        text = r.get("jd_text", "")
        if not text:
            text = (
                f"Job title: {r.get('title', r.get('role', ''))}. "
                f"Category: {r.get('category', '')}. "
                f"{r.get('description', '')} "
                f"Required skills: {', '.join(r.get('skills', []))}."
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
    return max(0, min(100, int((1 - score / 2.0) * 100)))


# ── Fast CV extraction: focus only on skills, projects, interests ──
def extract_cv_focus(cv_text: str) -> str:
    """
    Extract only the high-signal parts of a CV:
    skills, technologies, projects, interests, tools.
    Ignores address, phone, dates, references — noise for matching.
    """
    lines = cv_text.split('\n')
    focus_lines = []
    capture = False
    focus_sections = [
        'skill', 'project', 'interest', 'technolog', 'tool',
        'experience', 'education', 'certification', 'achievement',
        'publication', 'research', 'award', 'language'
    ]
    skip_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # dates
        r'\b\+?\d[\d\s\-]{8,}\b',                # phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
    ]

    for line in lines:
        line_lower = line.lower().strip()
        # Check if this is a section header
        if any(kw in line_lower for kw in focus_sections):
            capture = True
        # Skip noise lines
        skip = any(re.search(p, line) for p in skip_patterns)
        if not skip and len(line.strip()) > 2:
            focus_lines.append(line)

    result = '\n'.join(focus_lines)
    # Cap at 2000 chars for speed
    return result[:2000] if len(result) > 2000 else result


# ── Fast JD extraction: focus only on requirements ────────────────
def extract_jd_focus(jd_text: str) -> str:
    """
    Extract only requirements, skills, qualifications from JD.
    Ignores company culture, benefits, office hours — noise.
    """
    if not jd_text:
        return ""
    lines = jd_text.split('\n')
    focus_lines = []
    capture = False
    focus_keywords = [
        'requirement', 'skill', 'qualif', 'experience', 'must',
        'knowledge', 'proficien', 'familiar', 'expert', 'ability',
        'responsible', 'role', 'technolog', 'tool', 'framework'
    ]
    skip_keywords = [
        'offer', 'benefit', 'salary', 'culture', 'about us',
        'office hour', 'work arrangement', 'what we provide'
    ]

    for line in lines:
        line_lower = line.lower().strip()
        if any(kw in line_lower for kw in skip_keywords):
            capture = False
            continue
        if any(kw in line_lower for kw in focus_keywords):
            capture = True
        if capture or any(kw in line_lower for kw in focus_keywords):
            if len(line.strip()) > 3:
                focus_lines.append(line)

    result = '\n'.join(focus_lines)
    return result[:1500] if len(result) > 1500 else result


# ── CORE PIPELINE ─────────────────────────────────────────────────
def retrieve_context(cv_text: str, jd_text: str = "", k: int = 5) -> dict:
    # Auto-build index if missing
    if not os.path.exists("faiss_index"):
        print("Building FAISS index on first run...")
        roles = load_roles()
        build_index(roles)

    # Extract focused signals only
    cv_focused = extract_cv_focus(cv_text)
    jd_focused = extract_jd_focus(jd_text)

    # Build query — JD requirements weighted heavily
    if jd_focused:
        query = f"Required skills and qualifications:\n{jd_focused}\n\nCandidate skills and projects:\n{cv_focused}\n\nMatch candidate to job requirements:\n{jd_focused}"
    else:
        query = cv_focused

    # FAISS search
    index = load_index()
    results = index.similarity_search_with_score(query, k=k)

    all_matches = []
    for doc, score in results:
        role = dict(doc.metadata)
        role["match_pct"] = _score_to_pct(score)
        role["raw_score"] = round(float(score), 4)
        if "role" not in role:
            role["role"] = role.get("title", "Unknown Role")
        all_matches.append(role)

    all_matches.sort(key=lambda x: x["match_pct"], reverse=True)
    top_match = all_matches[0] if all_matches else {}

    # Skill gaps — compare against focused CV
    cv_lower = cv_focused.lower()
    required_skills = list(dict.fromkeys(
        sk for r in all_matches for sk in r.get("skills", [])
    ))
    skill_gaps = [sk for sk in required_skills if sk.lower() not in cv_lower]

    skill_freq = {}
    for r in all_matches:
        for sk in r.get("skills", []):
            if sk.lower() not in cv_lower:
                skill_freq[sk] = skill_freq.get(sk, 0) + 1
    resume_skills = sorted(skill_freq, key=skill_freq.get, reverse=True)[:8]

    blocks = []
    for r in all_matches[:4]:
        sal_min = r.get("salary_min", 0)
        sal_max = r.get("salary_max", 0)
        blocks.append(
            f"Role: {r.get('title', r.get('role', 'Unknown'))} — {r.get('category', '')} ({r['match_pct']}% match)\n"
            f"Description: {r.get('description', '')}\n"
            f"Required skills: {', '.join(r.get('skills', []))}\n"
            f"Salary: ~৳{sal_min:,}–৳{sal_max:,}/year (approx)\n"
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
        "cv_focused":      cv_focused,
        "jd_focused":      jd_focused,
    }
