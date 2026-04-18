"""
rag.py — CV-JD Matching with Real BD Job Market Data (2026)
Optimized: Realistic scoring, filters irrelevant experience
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
    'time series', 'forecasting', 'clustering', 'anomaly detection', 'kaggle'
}

CERT_KEYWORDS = {
    'machine learning', 'deep learning', 'tensorflow', 'pytorch', 'data science',
    'ai', 'artificial intelligence', 'nlp', 'computer vision', 'llm',
    'langchain', 'rag', 'aws', 'azure', 'gcp', 'huggingface', 'coursera'
}

EXPERIENCE_KEYWORDS = {
    'data scientist', 'machine learning engineer', 'ml engineer', 'ai engineer',
    'deep learning engineer', 'nlp engineer', 'computer vision engineer',
    'ai researcher', 'data science intern', 'ai intern', 'llm engineer',
    'prompt engineer', 'ai developer', 'data analyst', 'data engineer'
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
    return ""



def match_cv_with_jd(cv_text: str, jd_text: str) -> dict:
    """
    Match CV with specific JD using FAISS semantic search.
    Returns match score and similar roles from knowledge base.
    """
    if not cv_text or not jd_text:
        return {"match_pct": 0, "similar_roles": []}
    
    # Load index
    index = load_index()
    
    # Search for JDs similar to the provided one
    results = index.similarity_search_with_score(jd_text, k=5)
    
    all_matches = []
    for doc, score in results:
        role = dict(doc.metadata)
        # Calculate match with CV
        cv_lower = cv_text.lower()
        role_skills = role.get("skills", [])
        skills_found = sum(1 for s in role_skills if s.lower() in cv_lower)
        match_pct = int((skills_found / max(len(role_skills), 1)) * 100)
        
        role["match_pct"] = min(95, match_pct)
        all_matches.append(role)
    
    all_matches.sort(key=lambda x: x["match_pct"], reverse=True)
    
    return {
        "match_pct": all_matches[0]["match_pct"] if all_matches else 0,
        "similar_roles": all_matches[:4],
        "recommended_companies": [{"name": r.get("company"), "role": r.get("title"), "salary": r.get("salary")} 
                                   for r in all_matches[:3] if r.get("company")]
    }
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


def filter_relevant_experience(cv_text: str) -> tuple:
    """
    Extract ONLY AI/ML related experience from CV.
    Returns (relevant_experience_text, has_relevant_experience, matched_keywords)
    """
    experience = extract_experience(cv_text)
    if not experience:
        return "", False, []
    
    lines = experience.split('\n')
    relevant_lines = []
    matched_keywords = set()
    
    ai_keywords = [
        'machine learning', 'data science', 'ai', 'artificial intelligence',
        'data analyst', 'ml', 'deep learning', 'nlp', 'computer vision',
        'llm', 'rag', 'langchain', 'tensorflow', 'pytorch', 'keras',
        'scikit-learn', 'pandas', 'data analysis', 'data visualization',
        'sql', 'python', 'analytics', 'business intelligence', 'data engineer',
        'etl', 'data pipeline', 'model deployment', 'mlops'
    ]
    
    for line in lines:
        line_lower = line.lower()
        for kw in ai_keywords:
            if kw in line_lower:
                relevant_lines.append(line)
                matched_keywords.add(kw)
                break
    
    relevant_text = '\n'.join(relevant_lines) if relevant_lines else ""
    has_relevant = len(relevant_text) > 50  # At least some meaningful content
    
    return relevant_text, has_relevant, list(matched_keywords)[:5]


def score_ai_ml_readiness(cv_text: str) -> dict:
    """Score CV for AI/ML readiness - ONLY counts relevant experience."""
    if not cv_text:
        return {
            "total_score": 0,
            "level": "Not Ready",
            "recommendation": "Please upload your CV to get analysis.",
            "breakdown": {"experience": 0, "projects": 0, "certificates": 0, "skills": 0},
            "stats": {"ai_skills_found": 0, "projects_found": 0, "certificates_found": 0, "has_ai_experience": False}
        }
    
    experience = extract_experience(cv_text).lower()
    projects = extract_projects(cv_text).lower()
    certificates = extract_certificates(cv_text).lower()
    skills = extract_skills(cv_text).lower()
    full_cv = cv_text.lower()
    
    # Check if experience contains AI/ML keywords
    has_ai_experience = any(kw in experience for kw in [
        'machine learning', 'data science', 'ai', 'artificial intelligence',
        'data analyst', 'ml engineer', 'data scientist', 'deep learning',
        'nlp', 'computer vision', 'llm', 'rag', 'langchain', 'tensorflow',
        'pytorch', 'keras', 'scikit-learn', 'pandas', 'data analysis',
        'data engineer', 'business intelligence', 'analytics'
    ])
    
    # Experience Score (30%) - Only counts if relevant
    if has_ai_experience:
        exp_matches = 0
        for keyword in EXPERIENCE_KEYWORDS:
            if keyword in experience:
                exp_matches += 1
        exp_score = min(1.0, exp_matches / 3) * 30
    else:
        exp_score = 0  # No AI/ML experience = 0 points
    
    # Projects Score (25%)
    proj_matches = 0
    for keyword in PROJECT_KEYWORDS:
        if keyword in projects or keyword in full_cv:
            proj_matches += 1
    proj_score = min(1.0, proj_matches / 4) * 25
    
    # Certificates Score (20%)
    cert_matches = 0
    for keyword in CERT_KEYWORDS:
        if keyword in certificates or keyword in full_cv:
            cert_matches += 1
    cert_score = min(1.0, cert_matches / 3) * 20
    
    # Skills Score (25%)
    skill_matches = 0
    for keyword in CORE_AI_ML_SKILLS:
        if keyword in skills or keyword in full_cv:
            skill_matches += 1
    skill_score = min(1.0, skill_matches / 8) * 25
    
    total_score = exp_score + proj_score + cert_score + skill_score
    
    # Reality check: If no AI experience AND low skill score, cap at 25%
    if not has_ai_experience and skill_matches < 4:
        total_score = min(total_score, 25)
    
    # Determine level
    if total_score < 30:
        level = "Not Ready"
        if not has_ai_experience:
            recommendation = "Your CV lacks AI/ML work experience. Build projects or get relevant internships first."
        else:
            recommendation = "Focus on building foundational AI/ML skills and more projects."
    elif total_score < 60:
        level = "Building"
        recommendation = "You have foundational skills. Build more AI projects and gain practical experience."
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
            "has_ai_experience": has_ai_experience
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
    
    # Get relevant experience check
    _, has_ai_exp, _ = filter_relevant_experience(cv_text)
    
    # If no AI experience, reduce match percentage significantly
    if not has_ai_exp:
        match_pct = int(match_pct * 0.6)  # 40% reduction
    
    # Cap at 92% (no perfect match)
    return min(92, max(5, match_pct))


def retrieve_context(cv_text: str, jd_text: str = "", k: int = 5) -> dict:
    """Fast retrieval with real BD job market matching and realistic scoring."""
    
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
    
    # Filter relevant experience first
    relevant_exp, has_ai_experience, matched_keywords = filter_relevant_experience(cv_text)
    
    # Load index
    index = load_index()
    
    # Get AI/ML readiness score
    readiness = score_ai_ml_readiness(cv_text)
    
    # Override readiness if no AI experience
    if not has_ai_experience and readiness["total_score"] > 25:
        readiness["total_score"] = min(readiness["total_score"], 25)
        readiness["level"] = "Not Ready"
        readiness["recommendation"] = "Your CV lacks AI/ML work experience. Focus on building AI projects or getting relevant internships first."
    
    # Prepare query for FAISS - prioritize relevant experience
    cv_focused = extract_cv_focus(cv_text)
    
    # Build better query that emphasizes relevant experience
    if relevant_exp:
        enhanced_query = f"Relevant AI/ML Experience: {relevant_exp[:500]}\n\nFull CV: {cv_focused}"
    else:
        enhanced_query = cv_focused
    
    if jd_text:
        query = f"Candidate: {enhanced_query}\nJob Requirements: {jd_text[:1000]}"
    else:
        query = enhanced_query
    
    # FAISS search
    results = index.similarity_search_with_score(query, k=k)
    
    all_matches = []
    for doc, _ in results:
        role = dict(doc.metadata)
        
        # Calculate match percentage (now accounts for AI experience)
        match_pct = calculate_match_with_role(cv_text, role)
        
        # Additional penalty if role requires AI experience but CV has none
        role_requires_ai = any(kw in role.get("jd_text", "").lower() for kw in 
                               ['machine learning', 'deep learning', 'ai', 'llm', 'nlp', 'computer vision'])
        
        if role_requires_ai and not has_ai_experience:
            match_pct = int(match_pct * 0.5)  # 50% penalty for AI roles with no experience
        
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
    
    # Build context for LLM with emphasis on relevant experience
    blocks = []
    for r in all_matches[:4]:
        sal_min = r.get("salary_min", 0)
        sal_max = r.get("salary_max", 0)
        company = r.get("company", "Various")
        
        # Add warning if CV lacks AI experience for this role
        warning = ""
        if not has_ai_experience and "AI" in r.get("title", ""):
            warning = "\n⚠️ NOTE: This role requires AI/ML experience which is not present in candidate's CV."
        
        blocks.append(
            f"Company: {company}{warning}\n"
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
        "has_ai_experience": has_ai_experience,  # Add this for UI to show warnings
        "matched_experience_keywords": matched_keywords
    }
