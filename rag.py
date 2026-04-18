"""
rag.py — Fast focused extraction with REALISTIC match percentages.
Now uses semantic overlap + skill matching for accurate CV-JD comparison.
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


def calculate_semantic_overlap(cv_text: str, jd_text: str, role_skills: list) -> float:
    """
    Calculate TRUE semantic overlap between CV and job requirements.
    Uses skill matching, keyword extraction, and contextual analysis.
    """
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    # Skill matching (60% of score)
    skills_found = 0
    for skill in role_skills:
        skill_lower = skill.lower()
        if skill_lower in cv_lower:
            skills_found += 1
        # Check for variations
        elif " " in skill_lower and skill_lower.replace(" ", "") in cv_lower:
            skills_found += 0.5
    
    skill_score = min(1.0, skills_found / max(len(role_skills), 1))
    
    # Keyword extraction from JD (requirements focus)
    jd_keywords = set()
    requirement_patterns = [
        r'(?:experience with|knowledge of|proficiency in|familiarity with|expertise in)\s+([a-z][a-z\s]+?)(?=\.|,|\n|and)',
        r'(?:must have|required|essential|preferred)\s+([a-z][a-z\s]+?)(?=\.|,|\n)'
    ]
    for pattern in requirement_patterns:
        matches = re.findall(pattern, jd_lower, re.IGNORECASE)
        for match in matches:
            words = match.strip().split()[:3]
            jd_keywords.add(' '.join(words))
    
    # CV keyword presence
    if jd_keywords:
        keyword_matches = sum(1 for kw in jd_keywords if kw in cv_lower)
        keyword_score = keyword_matches / max(len(jd_keywords), 1)
    else:
        keyword_score = 0
    
    # Weighted score: 60% skills, 30% keywords, 10% length normalization
    final_score = (skill_score * 0.6) + (keyword_score * 0.3)
    
    # Small bonus for longer CV (more experienced candidates)
    cv_length_bonus = min(0.05, len(cv_text) / 20000 * 0.05)
    final_score = min(0.95, final_score + cv_length_bonus)
    
    return final_score


def extract_cv_focus(cv_text: str) -> str:
    """
    Extract high-signal content from CV: skills, experience, projects, certifications.
    Removes noise like dates, emails, phone numbers.
    """
    lines = cv_text.split('\n')
    focus_lines = []
    
    # Important sections to keep
    keep_patterns = [
        r'(?i)(?:skill|technical|experience|project|certification|education|achievement)',
        r'(?i)(?:python|sql|tensorflow|pytorch|langchain|rag|llm|ml|ai|data|analytics)',
        r'(?i)(?:docker|kubernetes|aws|gcp|azure|mlflow|huggingface|fastapi)',
    ]
    
    # Noise patterns to skip
    skip_patterns = [
        r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # dates
        r'\b\+?\d[\d\s\-]{8,}\b',                # phone numbers
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # emails
        r'(?i)^(?:page|ref|reference|available upon request)',
    ]
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped or len(line_stripped) < 3:
            continue
        
        # Check if line contains any keep pattern
        should_keep = any(re.search(p, line_stripped) for p in keep_patterns)
        
        # Skip noise lines
        is_noise = any(re.search(p, line_stripped) for p in skip_patterns)
        
        if should_keep and not is_noise:
            focus_lines.append(line_stripped[:200])  # Cap line length
    
    result = '\n'.join(focus_lines)
    
    # Cap total length for performance
    if len(result) > 3000:
        result = result[:3000]
    
    return result


def extract_jd_focus(jd_text: str) -> str:
    """Extract only requirements and responsibilities from JD."""
    if not jd_text:
        return ""
    
    lines = jd_text.split('\n')
    focus_lines = []
    capture = False
    
    # Section headers that indicate requirements
    capture_keywords = [
        'requirement', 'qualification', 'responsibilities', 'what you',
        'must have', 'required', 'essential', 'skills', 'experience',
        'need', 'looking for', 'ideal candidate', 'about you'
    ]
    
    # Skip these sections
    skip_keywords = [
        'benefit', 'perk', 'salary', 'compensation', 'equal opportunity',
        'diversity', 'culture', 'about us', 'company overview'
    ]
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Check for skip sections
        if any(kw in line_lower for kw in skip_keywords):
            capture = False
            continue
        
        # Check for capture sections
        if any(kw in line_lower for kw in capture_keywords):
            capture = True
            focus_lines.append(line)
        elif capture and len(line.strip()) > 5:
            focus_lines.append(line)
    
    result = '\n'.join(focus_lines)
    return result[:2000] if len(result) > 2000 else result


def retrieve_context(cv_text: str, jd_text: str = "", k: int = 5) -> dict:
    """Enhanced retrieval with realistic match percentages."""
    
    # Auto-build index if missing
    if not os.path.exists("faiss_index"):
        print("Building FAISS index on first run...")
        roles = load_roles()
        build_index(roles)

    # Extract focused signals
    cv_focused = extract_cv_focus(cv_text)
    jd_focused = extract_jd_focus(jd_text)

    # Build query
    if jd_focused:
        query = f"Candidate has: {cv_focused}\n\nJob requires: {jd_focused}"
    else:
        query = cv_focused

    # FAISS search
    index = load_index()
    results = index.similarity_search_with_score(query, k=k)

    all_matches = []
    for doc, faiss_score in results:
        role = dict(doc.metadata)
        
        # Calculate true semantic overlap
        semantic_score = calculate_semantic_overlap(
            cv_focused, 
            jd_focused if jd_focused else role.get("jd_text", ""), 
            role.get("skills", [])
        )
        
        # Convert FAISS distance to similarity (0-1)
        faiss_similarity = 1 / (1 + faiss_score)
        
        # Blend: 30% FAISS, 70% semantic (semantic is more reliable)
        blended_score = (faiss_similarity * 0.3) + (semantic_score * 0.7)
        
        # Convert to percentage (0-100)
        match_pct = int(blended_score * 100)
        
        # Ensure realistic ranges
        if match_pct > 92:
            match_pct = 92  # Cap at 92% - no perfect matches
        elif match_pct < 5:
            match_pct = max(3, match_pct)
        
        role["match_pct"] = match_pct
        role["semantic_score"] = round(semantic_score, 4)
        
        if "role" not in role:
            role["role"] = role.get("title", "Unknown Role")
        all_matches.append(role)

    # Sort by match percentage
    all_matches.sort(key=lambda x: x["match_pct"], reverse=True)
    top_match = all_matches[0] if all_matches else {}

    # Calculate skill gaps (only significant ones)
    cv_lower = cv_focused.lower()
    required_skills = list(dict.fromkeys(
        sk for r in all_matches[:3] for sk in r.get("skills", [])
    ))
    
    # Only show skills that are actually missing and important
    skill_gaps = []
    for sk in required_skills:
        sk_lower = sk.lower()
        if sk_lower not in cv_lower and sk_lower.replace(" ", "") not in cv_lower:
            # Check if skill appears in any form
            if not any(variant in cv_lower for variant in [sk_lower, sk_lower.replace("-", ""), sk_lower.replace(" ", "")]):
                skill_gaps.append(sk)
    
    # Resume skills to add (top missing skills from best matches)
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
        "cv_focused": cv_focused[:1500],
        "jd_provided": bool(jd_text),
    }
