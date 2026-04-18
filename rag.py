"""
rag.py — Fast CV-JD Matching with AI/ML Focused Scoring
Checks: Experience, Projects, Certificates, Skills, Responsibilities
Computationally efficient with keyword-based + semantic hybrid
"""

import json
import os
import re
import threading
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBED_MODEL = "all-MiniLM-L6-v2"
_index = None
_embedder = None
_index_lock = threading.Lock()

# ============================================================
# AI/ML KEYWORDS FOR FAST SCORING
# ============================================================

# Core AI/ML skills (high weight)
CORE_AI_ML_SKILLS = {
    'python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch',
    'langchain', 'rag', 'llm', 'gpt', 'bert', 'transformer', 'nlp',
    'computer vision', 'cv', 'yolo', 'opencv', 'scikit-learn', 'keras',
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter', 'colab'
}

# Project keywords that indicate AI/ML work
PROJECT_KEYWORDS = {
    'chatbot', 'llm', 'rag', 'recommendation', 'prediction', 'classification',
    'detection', 'segmentation', 'generative', 'sentiment', 'nlp', 'cv',
    'computer vision', 'object detection', 'face recognition', 'ocr',
    'time series', 'forecasting', 'clustering', 'anomaly detection'
}

# Certificate keywords
CERT_KEYWORDS = {
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'data science',
    'ai', 'artificial intelligence', 'nlp', 'computer vision', 'llm',
    'langchain', 'rag', 'aws machine learning', 'azure ai', 'gcp ai'
}

# Experience keywords (roles that indicate AI/ML work)
EXPERIENCE_KEYWORDS = {
    'data scientist', 'machine learning engineer', 'ml engineer', 'ai engineer',
    'deep learning engineer', 'nlp engineer', 'computer vision engineer',
    'ai researcher', 'ml researcher', 'data science intern', 'ai intern',
    'llm engineer', 'prompt engineer', 'ai developer'
}

# Job responsibility keywords (from JD)
RESPONSIBILITY_KEYWORDS = {
    'build machine learning', 'train model', 'deploy model', 'llm',
    'rag pipeline', 'langchain', 'vector database', 'fine-tune',
    'feature engineering', 'eda', 'model evaluation', 'hyperparameter',
    'neural network', 'cnn', 'rnn', 'lstm', 'transformer', 'attention'
}


def load_roles(path="jd_knowledge_base.json"):
    """Load roles from JSON - checks multiple locations."""
    possible_paths = [
        path,
        f"data/{path}",
        "data/jd_knowledge_base.json",
        "../data/jd_knowledge_base.json",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Cannot find jd_knowledge_base.json. Tried: {possible_paths}"
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
        with _index_lock:
            if _index is None:
                embedder = get_embedder()
                _index = FAISS.load_local(
                    "faiss_index", embedder,
                    allow_dangerous_deserialization=True
                )
    return _index


# ============================================================
# FAST CV SECTION EXTRACTION
# ============================================================

def extract_section(cv_text: str, section_name: str) -> str:
    """Extract a specific section from CV (experience, projects, certificates)."""
    patterns = [
        rf'(?i){section_name}[\s:]*\n(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|\Z)',
        rf'(?i){section_name}[\s:]*\n(.*?)(?=\n\w+[\s:]*\n|\Z)',
    ]
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.DOTALL)
        if match:
            return match.group(1).strip()[:2000]
    return ""


def extract_experience(cv_text: str) -> str:
    """Extract work experience section."""
    return extract_section(cv_text, "experience|work experience|employment|work history")


def extract_projects(cv_text: str) -> str:
    """Extract projects section."""
    return extract_section(cv_text, "projects|personal projects|academic projects")


def extract_certificates(cv_text: str) -> str:
    """Extract certificates section."""
    return extract_section(cv_text, "certificates|certifications|courses|training")


def extract_skills(cv_text: str) -> str:
    """Extract skills section."""
    return extract_section(cv_text, "skills|technical skills|core competencies")
    def match_companies(cv_text: str, role_match: dict) -> list:
    """Match CV to real companies from JD dataset."""
    cv_lower = cv_text.lower()
    companies = []
    
    # Load real companies from JD knowledge base
    roles = load_roles()
    
    for role in roles:
        if "company" in role and role.get("company"):
            company_data = {
                "name": role["company"],
                "role": role.get("title", role.get("role")),
                "match_score": 0,
                "salary_range": role.get("salary", {}),
                "location": role.get("location", "Dhaka"),
                "requirements": role.get("requirements", "")[:200]
            }
            
            # Calculate match score based on skills overlap
            required_skills = role.get("skills", [])
            skills_found = sum(1 for s in required_skills if s.lower() in cv_lower)
            match_score = (skills_found / max(len(required_skills), 1)) * 100
            
            company_data["match_score"] = round(match_score, 1)
            companies.append(company_data)
    
    # Sort by match score and return top 5
    companies.sort(key=lambda x: x["match_score"], reverse=True)
    return companies[:5]


# ============================================================
# SCORING FUNCTIONS
# ============================================================

def score_ai_ml_readiness(cv_text: str) -> dict:
    """
    Score CV for AI/ML readiness based on:
    - Experience (30%)
    - Projects (25%)
    - Certificates (20%)
    - Skills (25%)
    """
    experience = extract_experience(cv_text).lower()
    projects = extract_projects(cv_text).lower()
    certificates = extract_certificates(cv_text).lower()
    skills = extract_skills(cv_text).lower()
    full_cv = cv_text.lower()
    
    # Experience Score (30%)
    exp_score = 0
    exp_matches = 0
    for keyword in EXPERIENCE_KEYWORDS:
        if keyword in experience or keyword in full_cv:
            exp_matches += 1
    exp_score = min(1.0, exp_matches / 4) * 30  # 4+ matches = 30%
    
    # Projects Score (25%)
    proj_score = 0
    proj_matches = 0
    for keyword in PROJECT_KEYWORDS:
        if keyword in projects or keyword in full_cv:
            proj_matches += 1
    proj_score = min(1.0, proj_matches / 5) * 25  # 5+ matches = 25%
    
    # Certificates Score (20%)
    cert_score = 0
    cert_matches = 0
    for keyword in CERT_KEYWORDS:
        if keyword in certificates or keyword in full_cv:
            cert_matches += 1
    cert_score = min(1.0, cert_matches / 4) * 20  # 4+ matches = 20%
    
    # Skills Score (25%)
    skill_score = 0
    skill_matches = 0
    for keyword in CORE_AI_ML_SKILLS:
        if keyword in skills or keyword in full_cv:
            skill_matches += 1
    skill_score = min(1.0, skill_matches / 10) * 25  # 10+ matches = 25%
    
    total_score = exp_score + proj_score + cert_score + skill_score
    
    # Determine readiness level
    if total_score < 30:
        level = "Not Ready"
        recommendation = "Focus on building foundational AI/ML skills. Consider other roles like Data Analyst or Business Analyst first."
    elif total_score < 60:
        level = "Building"
        recommendation = "You have good fundamentals. Build more projects, get certifications, and gain practical experience."
    else:
        level = "Ready"
        recommendation = "You're ready to apply for AI/ML roles! Your CV shows strong alignment with the field."
    
    return {
        "total_score": round(total_score, 1),
        "level": level,
        "recommendation": recommendation,
        "breakdown": {
            "experience": round(exp_score, 1),
            "projects": round(proj_score, 1),
            "certificates": round(cert_score, 1),
            "skills": round(skill_score, 1)
        },
        "stats": {
            "ai_skills_found": skill_matches,
            "projects_found": proj_matches,
            "certificates_found": cert_matches,
            "experience_matches": exp_matches
        }
    }


def score_jd_alignment(cv_text: str, jd_text: str) -> float:
    """
    Score how well CV matches JD responsibilities.
    Focuses ONLY on responsibilities/skills from JD.
    """
    if not jd_text:
        return 0.5  # Neutral if no JD provided
    
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    # Extract responsibilities section from JD
    resp_text = extract_section(jd_text, "responsibilities|what you|key responsibilities|role")
    if not resp_text:
        resp_text = jd_text[:1500]
    resp_lower = resp_text.lower()
    
    # Count keyword matches
    matches = 0
    total_keywords = 0
    
    for keyword in RESPONSIBILITY_KEYWORDS:
        if keyword in resp_lower:
            total_keywords += 1
            if keyword in cv_lower:
                matches += 1
    
    # Also check for general skill matches
    for skill in CORE_AI_ML_SKILLS:
        if skill in jd_lower:
            total_keywords += 0.5
            if skill in cv_lower:
                matches += 0.5
    
    if total_keywords == 0:
        return 0.5
    
    alignment_score = min(1.0, matches / total_keywords)
    return alignment_score


def calculate_final_match(cv_text: str, jd_text: str, role_skills: list) -> dict:
    """
    Calculate final match percentage with practical scoring.
    - Base score from AI/ML readiness (60%)
    - JD alignment boost (40% if JD provided)
    """
    # Get AI/ML readiness score
    readiness = score_ai_ml_readiness(cv_text)
    base_score = readiness["total_score"]
    
    # Get JD alignment if JD provided
    jd_alignment = score_jd_alignment(cv_text, jd_text) if jd_text else 0.5
    
    if jd_text:
        # Weighted: 60% readiness, 40% JD alignment
        final_score = (base_score * 0.6) + (jd_alignment * 40)
    else:
        # No JD: just use readiness score with small adjustment
        final_score = base_score
    
    # Adjust based on skill overlap with role
    cv_lower = cv_text.lower()
    skill_overlap = 0
    for skill in role_skills[:8]:
        if skill.lower() in cv_lower:
            skill_overlap += 1
    skill_boost = (skill_overlap / max(len(role_skills[:8]), 1)) * 10
    final_score = min(92, final_score + skill_boost)
    
    return {
        "match_pct": round(final_score),
        "readiness": readiness,
        "jd_alignment": round(jd_alignment * 100, 1) if jd_text else None
    }


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve_context(cv_text: str, jd_text: str = "", k: int = 5) -> dict:
    """Fast retrieval with practical scoring."""
    
    # Auto-build index if missing
    if not os.path.exists("faiss_index"):
        print("Building FAISS index on first run...")
        roles = load_roles()
        build_index(roles)

    # Get AI/ML readiness score
    readiness = score_ai_ml_readiness(cv_text)
    
    # Load index for role matching
    index = load_index()
    
    # Build query for FAISS
    if jd_text:
        # Extract JD responsibilities for query
        jd_focus = extract_section(jd_text, "responsibilities|requirements|qualifications")
        if not jd_focus:
            jd_focus = jd_text[:1000]
        query = f"Candidate AI/ML readiness: {readiness['total_score']}%\n\nJob requirements: {jd_focus}"
    else:
        cv_focus = extract_section(cv_text, "skills|experience|projects")
        if not cv_focus:
            cv_focus = cv_text[:1500]
        query = f"Candidate skills and experience: {cv_focus}"
    
    # FAISS search
    results = index.similarity_search_with_score(query, k=k)
    
    all_matches = []
    for doc, faiss_score in results:
        role = dict(doc.metadata)
        
        # Calculate final match percentage
        match_result = calculate_final_match(cv_text, jd_text, role.get("skills", []))
        
        role["match_pct"] = match_result["match_pct"]
        role["readiness_score"] = match_result["readiness"]["total_score"]
        role["readiness_level"] = match_result["readiness"]["level"]
        
        if "role" not in role:
            role["role"] = role.get("title", "Unknown Role")
        all_matches.append(role)
    
    # Sort by match percentage
    all_matches.sort(key=lambda x: x["match_pct"], reverse=True)
    top_match = all_matches[0] if all_matches else {}
    
    # Calculate skill gaps
    cv_lower = cv_text.lower()
    required_skills = list(dict.fromkeys(
        sk for r in all_matches[:3] for sk in r.get("skills", [])
    ))
    
    skill_gaps = []
    for sk in required_skills:
        sk_lower = sk.lower()
        if sk_lower not in cv_lower and sk_lower.replace(" ", "") not in cv_lower:
            if not any(variant in cv_lower for variant in [sk_lower, sk_lower.replace("-", ""), sk_lower.replace(" ", "")]):
                skill_gaps.append(sk)
    
    # Resume skills to add
    skill_freq = {}
    for r in all_matches[:3]:
        for sk in r.get("skills", []):
            if sk.lower() not in cv_lower:
                skill_freq[sk] = skill_freq.get(sk, 0) + (r["match_pct"] / 100)
    resume_skills = sorted(skill_freq, key=skill_freq.get, reverse=True)[:6]
    
    # Build context for LLM
    blocks = []
    for r in all_matches[:4]:
        sal_min = r.get("salary_min", 0)
        sal_max = r.get("salary_max", 0)
        blocks.append(
            f"Role: {r.get('title', r.get('role', 'Unknown'))} — {r.get('category', '')} ({r['match_pct']}% match)\n"
            f"Required skills: {', '.join(r.get('skills', []))}\n"
            f"Salary: ~৳{sal_min:,}–৳{sal_max:,}/year\n"
            f"Market demand: {r.get('market_demand', '')}\n"
            f"Career path: {r.get('career_path', '')}\n"
        )
    
    return {
        "top_match": top_match,
        "all_matches": all_matches,
        "similar_roles": all_matches,
        "skill_gaps": skill_gaps[:6],
        "resume_skills": resume_skills,
        "raw_context": "\n\n---\n\n".join(blocks),
        "readiness": readiness,
        "cv_focused": cv_text[:1500],
        "jd_provided": bool(jd_text),
    }
