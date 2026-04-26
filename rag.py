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
    """
    Score CV for AI/ML readiness — calibrated for meaningful differentiation.
    Beginner (no projects/certs): 10-29
    Developing (some skills/certs/1-2 projects): 30-49
    Intermediate (solid projects + certs, no formal exp): 50-67
    Advanced (strong certs + projects + some exp): 68-81
    Job Ready (industry exp + research/publications + strong skills): 82-97
    Work experience is a strong bonus; projects + certs carry most weight for students.
    """
    if not cv_text:
        return {
            "total_score": 0,
            "level": "Beginner",
            "recommendation": "Please upload your CV to get analysis.",
            "breakdown": {"experience": 0, "projects": 0, "certificates": 0, "skills": 0, "achievements": 0},
            "stats": {"ai_skills_found": 0, "projects_found": 0, "certificates_found": 0, "has_ai_experience": False}
        }

    experience    = extract_experience(cv_text).lower()
    projects      = extract_projects(cv_text).lower()
    certificates  = extract_certificates(cv_text).lower()
    skills        = extract_skills(cv_text).lower()
    full_cv       = cv_text.lower()

    # ── Has AI/ML experience? ──
    exp_keywords = [
        'machine learning', 'data science', 'ai', 'artificial intelligence',
        'data analyst', 'ml engineer', 'data scientist', 'deep learning',
        'nlp', 'computer vision', 'llm', 'rag', 'langchain', 'tensorflow',
        'pytorch', 'keras', 'scikit-learn', 'pandas', 'data analysis',
        'data engineer', 'business intelligence', 'analytics', 'research intern',
        'ai intern', 'ml intern', 'research assistant', 'undergraduate research'
    ]
    has_ai_experience = any(kw in experience for kw in exp_keywords)

    # ── Experience Score (max 25) ──
    # Formal AI/ML role titles in experience = strong signal
    exp_matches = sum(1 for kw in EXPERIENCE_KEYWORDS if kw in experience)
    research_signals = sum(1 for kw in [
        'research intern', 'ml intern', 'ai intern', 'research assistant',
        'undergraduate research', 'graduate research', 'internship'
    ] if kw in full_cv)
    if exp_matches >= 2:
        exp_score = 22.0     # Multiple formal AI/ML roles
    elif exp_matches == 1:
        exp_score = 15.0     # One formal AI/ML role
    elif research_signals >= 2:
        exp_score = 10.0     # Research internship or similar
    elif research_signals == 1 or has_ai_experience:
        exp_score = 6.0      # Some relevant context
    else:
        exp_score = 0.0
    exp_score = min(25.0, exp_score)

    # ── Projects Score (max 28) — key differentiator for students ──
    proj_matches = sum(1 for kw in PROJECT_KEYWORDS if kw in projects or kw in full_cv)
    project_count = len(re.findall(
        r'(?i)(?:^|\n)\s*(?:\d+[\.\)]\s+|[-•*]\s+)?(?:project|built|developed|implemented|created|designed)\b',
        cv_text
    ))
    github_signal = 1 if 'github' in full_cv else 0
    kaggle_signal = 1 if 'kaggle' in full_cv else 0
    research_pub = sum(1 for kw in ['arxiv', 'paper', 'publication', 'published', 'ieee', 'conference', 'journal']
                       if kw in full_cv)
    # Base: project keywords (max 3 = 12pts)
    proj_base = min(12.0, proj_matches * 4.0)
    proj_bonus = (github_signal * 3) + (kaggle_signal * 3) + min(10, research_pub * 5)
    proj_score = min(28.0, proj_base + proj_bonus)
    if proj_matches == 0 and project_count == 0 and not github_signal:
        proj_score = 0.0

    # ── Certificates Score (max 25) ──
    cert_matches = sum(1 for kw in CERT_KEYWORDS if kw in certificates or kw in full_cv)
    platform_certs = sum(1 for kw in [
        'coursera', 'udemy', 'deeplearning.ai', 'fast.ai',
        'google certificate', 'microsoft certified', 'aws certified',
        'tensorflow developer', 'pytorch', 'nvidia deep learning',
        'datacamp', 'edx', 'linkedin learning'
    ] if kw in full_cv)
    cert_count = len(re.findall(
        r'(?i)(?:certificate|certification|certified|course completed|nanodegree)', cv_text
    ))
    cert_base = min(12.0, cert_matches * 4.0)
    cert_platform_bonus = min(8.0, platform_certs * 4.0)
    cert_count_bonus = min(5.0, max(0.0, (cert_count - 1) * 1.5))
    cert_score = min(25.0, cert_base + cert_platform_bonus + cert_count_bonus)
    if cert_matches == 0 and platform_certs == 0:
        cert_score = 0.0

    # ── Skills Score (max 17) ──
    skill_matches = sum(1 for kw in CORE_AI_ML_SKILLS if kw in skills or kw in full_cv)
    if skill_matches >= 10:
        skill_score = 17.0
    elif skill_matches >= 7:
        skill_score = 13.0
    elif skill_matches >= 5:
        skill_score = 10.0
    elif skill_matches >= 3:
        skill_score = 7.0
    elif skill_matches >= 1:
        skill_score = 3.0
    else:
        skill_score = 0.0

    # ── Achievements bonus (max 5) ──
    achievement_signals = sum(1 for kw in [
        'award', 'winner', 'champion', 'hackathon', 'competition', 'rank',
        'scholarship', 'honor', 'distinction', 'cum laude', 'dean', 'merit',
        'first place', 'second place', 'top', 'finalist', 'selected'
    ] if kw in full_cv)
    achievement_score = min(5.0, achievement_signals * 1.5)

    total_score = exp_score + proj_score + cert_score + skill_score + achievement_score

    # ── Floors: prevent wildly low scores for candidates with real content ──
    if skill_matches >= 3 and cert_matches >= 1 and proj_matches >= 1:
        total_score = max(total_score, 35.0)   # solid beginner floor
    elif skill_matches >= 2 or cert_matches >= 1:
        total_score = max(total_score, 20.0)   # basic floor

    total_score = round(min(97, total_score), 1)

    # ── Levels ──
    if total_score < 30:
        level = "Beginner"
        rec = "Start with Python + ML fundamentals. Build 1-2 small AI projects and earn a free Coursera certificate."
    elif total_score < 50:
        level = "Developing"
        rec = "Good foundation! Deepen skills with real AI projects, contribute to GitHub, and target internships."
    elif total_score < 68:
        level = "Intermediate"
        rec = "Strong profile! Target junior AI/ML roles and research internships in Bangladesh's growing tech scene."
    elif total_score < 82:
        level = "Advanced"
        rec = "Very strong AI/ML profile! Apply confidently to mid-level AI Engineer and Data Scientist roles."
    else:
        level = "Job Ready"
        rec = "Excellent profile! Apply for senior AI/ML Engineer, Research Scientist, and team lead roles."

    return {
        "total_score": total_score,
        "level": level,
        "recommendation": rec,
        "breakdown": {
            "experience":   round(exp_score, 1),
            "projects":     round(proj_score, 1),
            "certificates": round(cert_score, 1),
            "skills":       round(skill_score, 1),
            "achievements": round(achievement_score, 1),
        },
        "stats": {
            "ai_skills_found":    skill_matches,
            "projects_found":     proj_matches,
            "certificates_found": cert_matches,
            "has_ai_experience":  has_ai_experience
        }
    }


def calculate_match_with_role(cv_text: str, role: dict) -> int:
    """
    Calculate match % with a role.
    Rewards skills found anywhere in CV (projects, certs, education).
    No harsh penalty for lacking formal work experience.
    """
    if not cv_text:
        return 0

    cv_lower = cv_text.lower()
    role_skills = role.get("skills", [])

    if not role_skills:
        return 50

    skills_found = 0.0
    for skill in role_skills:
        skill_lower = skill.lower()
        if skill_lower in cv_lower:
            skills_found += 1.0
        elif skill_lower.replace(" ", "") in cv_lower:
            skills_found += 0.8
        elif any(part in cv_lower for part in skill_lower.split() if len(part) > 3):
            skills_found += 0.4   # partial keyword match

    match_pct = int((skills_found / len(role_skills)) * 100)

    # Bonus for AI-related projects and certificates
    project_boost = sum(1 for kw in ['project', 'github', 'kaggle', 'research', 'thesis', 'paper']
                        if kw in cv_lower)
    cert_boost    = sum(1 for kw in ['coursera', 'certificate', 'certification', 'deeplearning',
                                      'udemy', 'google', 'microsoft', 'nvidia']
                        if kw in cv_lower)
    match_pct += min(15, (project_boost + cert_boost) * 3)

    # Soft penalty (not harsh) if no AI experience — max 15% reduction
    _, has_ai_exp, _ = filter_relevant_experience(cv_text)
    if not has_ai_exp:
        match_pct = int(match_pct * 0.85)

    return min(92, max(8, match_pct))


def retrieve_context(cv_text: str, jd_text: str = "", k: int = 5) -> dict:
    """
    Retrieve best-matching roles from FAISS knowledge base.
    Query is enriched with skills, projects, certificates for better semantic match.
    Scoring is fair for beginners without formal work experience.
    """
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

    relevant_exp, has_ai_experience, matched_keywords = filter_relevant_experience(cv_text)
    index    = load_index()
    readiness = score_ai_ml_readiness(cv_text)  # uses improved scorer

    # ── Build enriched FAISS query: skills + projects + certs are all signals ──
    skills_text  = extract_skills(cv_text)
    projects_text = extract_projects(cv_text)
    certs_text   = extract_certificates(cv_text)
    cv_focused   = extract_cv_focus(cv_text)

    query_parts = []
    if skills_text:
        query_parts.append(f"Technical Skills: {skills_text[:400]}")
    if projects_text:
        query_parts.append(f"Projects: {projects_text[:400]}")
    if certs_text:
        query_parts.append(f"Certifications: {certs_text[:300]}")
    if relevant_exp:
        query_parts.append(f"Experience: {relevant_exp[:300]}")
    if not query_parts:
        query_parts.append(cv_focused[:800])

    enhanced_query = "\n\n".join(query_parts)

    if jd_text:
        query = f"{enhanced_query}\n\nTarget JD: {jd_text[:800]}"
    else:
        query = enhanced_query

    results = index.similarity_search_with_score(query, k=k)

    all_matches = []
    for doc, _ in results:
        role = dict(doc.metadata)
        match_pct = calculate_match_with_role(cv_text, role)
        # No additional penalty beyond what calculate_match_with_role already applies
        role["match_pct"] = match_pct
        role["readiness_score"] = readiness["total_score"]
        if "role" not in role:
            role["role"] = role.get("title", "Unknown Role")
        all_matches.append(role)

    all_matches.sort(key=lambda x: x["match_pct"], reverse=True)
    top_match = all_matches[0] if all_matches else {}

    # ── Skill gaps: skills required by top roles but missing from CV ──
    cv_lower = cv_text.lower()
    required_skills = list(dict.fromkeys(
        sk for r in all_matches[:3] for sk in r.get("skills", [])
    ))
    skill_gaps = [
        sk for sk in required_skills
        if not any(v in cv_lower for v in [
            sk.lower(), sk.lower().replace(" ", ""), sk.lower().replace("-", "")
        ])
    ]

    # ── Resume skills to add (sorted by frequency across top matches) ──
    skill_freq = {}
    for r in all_matches[:3]:
        for sk in r.get("skills", []):
            if sk.lower() not in cv_lower:
                skill_freq[sk] = skill_freq.get(sk, 0) + (r["match_pct"] / 100)
    resume_skills = sorted(skill_freq, key=skill_freq.get, reverse=True)[:6]

    # ── Context blocks for LLM ──
    blocks = []
    for r in all_matches[:4]:
        sal_min = r.get("salary_min", 0)
        sal_max = r.get("salary_max", 0)
        blocks.append(
            f"Company: {r.get('company', 'Various')}\n"
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
        "cv_focused": enhanced_query[:1500],
        "jd_provided": bool(jd_text),
        "has_ai_experience": has_ai_experience,
        "matched_experience_keywords": matched_keywords
    }
