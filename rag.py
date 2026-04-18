"""
rag.py — CV-JD Matching with Real BD Job Market Data (2026)
Optimized: No torchvision issues, fast keyword-based scoring
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
# AI/ML KEYWORDS FOR FAST SCORING (from real BD job market)
# ============================================================

CORE_AI_ML_SKILLS = {
    'python', 'machine learning', 'deep learning', 'tensorflow', 'pytorch',
    'langchain', 'rag', 'llm', 'gpt', 'bert', 'transformer', 'nlp',
    'computer vision', 'cv', 'yolo', 'opencv', 'scikit-learn', 'keras',
    'pandas', 'numpy', 'matplotlib', 'sql', 'git', 'docker', 'kubernetes',
    'aws', 'gcp', 'azure', 'mlflow', 'huggingface', 'fastapi', 'flask'
}

PROJECT_KEYWORDS = {
    'chatbot', 'llm', 'rag', 'recommendation', 'prediction', 'classification',
    'detection', 'segmentation', 'generative', 'sentiment', 'nlp', 'cv',
    'computer vision', 'object detection', 'face recognition', 'ocr',
    'time series', 'forecasting', 'clustering', 'anomaly detection'
}

CERT_KEYWORDS = {
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'data science',
    'ai', 'artificial intelligence', 'nlp', 'computer vision', 'llm',
    'langchain', 'rag', 'aws', 'azure', 'gcp', 'huggingface'
}

EXPERIENCE_KEYWORDS = {
    'data scientist', 'machine learning engineer', 'ml engineer', 'ai engineer',
    'deep learning engineer', 'nlp engineer', 'computer vision engineer',
    'ai researcher', 'data science intern', 'ai intern', 'llm engineer',
    'prompt engineer', 'ai developer', 'data analyst'
}


def load_roles(path="jd_knowledge_base.json"):
    """Load roles from JSON - checks multiple locations."""
    possible_paths = [
        path,
        f"data/{path}",
        "data/jd_knowledge_base.json",
        "../data/jd_knowledge_base.json",
        "/mount/src/rag_app/jd_knowledge_base.json",
        "/mount/src/rag_app/data/jd_knowledge_base.json",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    # Return default roles if file not found
    return get_default_roles()


def get_default_roles():
    """Fallback default roles if JSON file is missing."""
    return [
        {
            "role": "AI/ML Engineer",
            "title": "AI/ML Engineer",
            "company": "Tech Company",
            "category": "Entry-level",
            "description": "Build and deploy machine learning models",
            "skills": ["Python", "Machine Learning", "TensorFlow", "PyTorch"],
            "salary_min": 30000,
            "salary_max": 60000,
            "location": "Dhaka",
            "jd_text": "Looking for an AI/ML Engineer with Python and ML skills."
        }
    ]


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
                if os.path.exists("faiss_index"):
                    _index = FAISS.load_local(
                        "faiss_index", embedder,
                        allow_dangerous_deserialization=True
                    )
                else:
                    # Build index if it doesn't exist
                    roles = load_roles()
                    _index = build_index(roles)
    return _index


def extract_section(cv_text: str, section_name: str) -> str:
    """Extract a specific section from CV. Returns empty string if not found."""
    patterns = [
        rf'(?i){section_name}[\s:]*\n(.*?)(?=\n[A-Z][A-Z\s]+:|\n\n|\Z)',
        rf'(?i){section_name}[\s:]*\n(.*?)(?=\n\w+[\s:]*\n|\Z)',
    ]
    for pattern in patterns:
        match = re.search(pattern, cv_text, re.DOTALL)
        if match and match.group(1):
            return match.group(1).strip()[:2000]
    return ""  # Return empty string if no match


def extract_experience(cv_text: str) -> str:
    result = extract_section(cv_text, "experience|work experience|employment|work history")
    return result if result else ""


def extract_projects(cv_text: str) -> str:
    result = extract_section(cv_text, "projects|personal projects|academic projects")
    return result if result else ""


def extract_certificates(cv_text: str) -> str:
    result = extract_section(cv_text, "certificates|certifications|courses|training")
    return result if result else ""


def extract_skills(cv_text: str) -> str:
    result = extract_section(cv_text, "skills|technical skills|core competencies")
    return result if result else ""


def extract_cv_focus(cv_text: str) -> str:
    """Extract high-signal content from CV."""
    if not cv_text:
        return ""
    
    lines = cv_text.split('\n')
    focus_lines = []
    
    keep_patterns = [
        r'(?i)(?:skill|technical|experience|project|certification|education|achievement)',
        r'(?i)(?:python|sql|tensorflow|pytorch|langchain|rag|llm|ml|ai|data|analytics)',
    ]
    
    skip_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
        r'\b\+?\d[\d\s\-]{8,}\b',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    ]
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 3:
            continue
        
        should_keep = any(re.search(p, line_stripped) for p in keep_patterns)
        is_noise = any(re.search(p, line_stripped) for p in skip_patterns)
        
        if should_keep and not is_noise:
            focus_lines.append(line_stripped[:200])
    
    result = '\n'.join(focus_lines)
    return result[:3000] if len(result) > 3000 else result


def score_ai_ml_readiness(cv_text: str) -> dict:
    """Score CV for AI/ML readiness based on real BD job market."""
    if not cv_text:
        return {
            "total_score": 0,
            "level": "Not Ready",
            "recommendation": "Please upload your CV to get analysis.",
            "breakdown": {"experience": 0, "projects": 0, "certificates": 0, "skills": 0},
            "stats": {"ai_skills_found": 0, "projects_found": 0, "certificates_found": 0, "experience_matches": 0}
        }
    
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
    exp_score = min(1.0, exp_matches / 3) * 30
    
    # Projects Score (25%)
    proj_score = 0
    proj_matches = 0
    for keyword in PROJECT_KEYWORDS:
        if keyword in projects or keyword in full_cv:
            proj_matches += 1
    proj_score = min(1.0, proj_matches / 4) * 25
    
    # Certificates Score (20%)
    cert_score = 0
    cert_matches = 0
    for keyword in CERT_KEYWORDS:
        if keyword in certificates or keyword in full_cv:
            cert_matches += 1
    cert_score = min(1.0, cert_matches / 3) * 20
    
    # Skills Score (25%)
    skill_score = 0
    skill_matches = 0
    for keyword in CORE_AI_ML_SKILLS:
        if keyword in skills or keyword in full_cv:
            skill_matches += 1
    skill_score = min(1.0, skill_matches / 8) * 25
    
    total_score = exp_score + proj_score + cert_score + skill_score
    
    if total_score < 30:
        level = "Not Ready"
        recommendation = "Focus on building foundational AI/ML skills. Consider Data Analyst roles first."
    elif total_score < 60:
        level = "Building"
        recommendation = "You have good fundamentals. Build more projects and get certifications."
    else:
        level = "Ready"
        recommendation = "You're ready to apply for AI/ML roles in Bangladesh!"
    
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


def calculate_match_with_role(cv_text: str, role: dict) -> int:
    """Calculate match percentage with a specific role."""
    if not cv_text:
        return 0
    
    cv_lower = cv_text.lower()
    role_skills = role.get("skills", [])
    
    if not role_skills:
        return 50
    
    skills_found = 0
    for skill in role_skills:
        skill_lower = skill.lower()
        if skill_lower in cv_lower:
            skills_found += 1
        elif skill_lower.replace(" ", "") in cv_lower:
            skills_found += 0.5
    
    match_pct = int((skills_found / len(role_skills)) * 100)
    
    # Cap at 92% (no perfect match)
    return min(92, max(5, match_pct))


def retrieve_context(cv_text: str, jd_text: str = "", k: int = 5) -> dict:
    """Fast retrieval with real BD job market matching."""
    
    if not cv_text:
        return {
            "top_match": {},
            "all_matches": [],
            "similar_roles": [],
            "skill_gaps": [],
            "resume_skills": [],
            "raw_context": "",
            "readiness": score_ai_ml_readiness(""),
            "cv_focused": "",
            "jd_provided": False,
        }
    
    # Load index
    index = load_index()
    
    # Get AI/ML readiness score
    readiness = score_ai_ml_readiness(cv_text)
    
    # Prepare query for FAISS
    cv_focused = extract_cv_focus(cv_text)
    if jd_text:
        query = f"Candidate: {cv_focused}\nJob: {jd_text[:1000]}"
    else:
        query = cv_focused
    
    # FAISS search
    results = index.similarity_search_with_score(query, k=k)
    
    all_matches = []
    for doc, _ in results:
        role = dict(doc.metadata)
        
        # Calculate match percentage
        match_pct = calculate_match_with_role(cv_text, role)
        
        role["match_pct"] = match_pct
        role["readiness_score"] = readiness["total_score"]
        
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
        company = r.get("company", "Various")
        blocks.append(
            f"Company: {company}\n"
            f"Role: {r.get('title', r.get('role', 'Unknown'))} ({r['match_pct']}% match)\n"
            f"Required skills: {', '.join(r.get('skills', []))}\n"
            f"Salary: ৳{sal_min:,}–৳{sal_max:,}/month\n"
            f"Location: {r.get('location', 'Dhaka')}\n"
        )
    
    return {
        "top_match": top_match,
        "all_matches": all_matches,
        "similar_roles": all_matches,
        "skill_gaps": skill_gaps[:6],
        "resume_skills": resume_skills,
        "raw_context": "\n\n---\n\n".join(blocks),
        "readiness": readiness,
        "cv_focused": cv_focused[:1500],
        "jd_provided": bool(jd_text),
    }
