"""
app.py — CV Analyzer with Smart Tabs & Company Matching
"""

import streamlit as st
import tempfile
import os
import re

# Import at the top (not inside functions) - OPTIMIZED
from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context

st.set_page_config(
    page_title="CV Analyzer | AI Career Match",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS STYLES
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

* {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #0a0a14 0%, #0f0f20 100%);
}

/* Navigation Bar */
.nav-bar {
    background: rgba(15, 15, 32, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #2d2d5a;
    padding: 0.75rem 2rem;
    position: sticky;
    top: 0;
    z-index: 1000;
    margin-bottom: 1.5rem;
}
.nav-container {
    max-width: 1200px;
    margin: 0 auto;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.nav-links {
    display: flex;
    gap: 0.5rem;
    align-items: center;
    flex-wrap: wrap;
}
.nav-btn {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.4rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}
.nav-btn:hover {
    background: rgba(168, 85, 247, 0.1);
    color: #a855f7;
}
.nav-btn.active {
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: white;
}
.user-name {
    font-size: 0.8rem;
    color: #64748b;
    padding-left: 1rem;
    border-left: 1px solid #2d2d5a;
}

/* Cards */
.card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
}
.skill-chip {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 6px;
    margin: 0.2rem;
}
.gap-chip {
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.3);
    color: #fca5a5;
}
.add-chip {
    background: rgba(34, 197, 94, 0.15);
    border-color: rgba(34, 197, 94, 0.3);
    color: #86efac;
}
.hero-match {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Analysis Tabs */
.analysis-tabs {
    display: flex;
    gap: 0.5rem;
    margin: 1rem 0 1.5rem 0;
    padding: 0.5rem;
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    flex-wrap: wrap;
}
.tab-btn {
    background: transparent;
    border: none;
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}
.tab-btn:hover {
    background: rgba(168, 85, 247, 0.1);
    color: #a855f7;
}
.tab-btn.active {
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: white;
}
.company-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s;
}
.company-card:hover {
    border-color: #7c3aed;
    transform: translateX(4px);
}
.salary-badge {
    display: inline-block;
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
    font-size: 0.7rem;
    padding: 0.2rem 0.5rem;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_tab" not in st.session_state:
    st.session_state.analysis_tab = "overview"
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "cv_text" not in st.session_state:
    st.session_state.cv_text = None
if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""
if "analysis_raw" not in st.session_state:
    st.session_state.analysis_raw = None
if "retrieved" not in st.session_state:
    st.session_state.retrieved = None
if "candidate_name" not in st.session_state:
    st.session_state.candidate_name = ""
if "name_entered" not in st.session_state:
    st.session_state.name_entered = False
if "matched_companies" not in st.session_state:
    st.session_state.matched_companies = []


def parse_analysis(text: str) -> dict:
    """Parse LLM response for structured data."""
    def get(tag):
        m = re.search(rf"{tag}:\s*(.+?)(?=\n[A-Z_]+:|$)", text, re.DOTALL)
        return m.group(1).strip() if m else ""

    def get_list(tag):
        m = re.search(rf"{tag}:\s*((?:\n- .+)+)", text)
        if not m:
            return []
        return [l.lstrip("- ").strip() for l in m.group(1).strip().split("\n") if l.strip().startswith("-")]

    return {
        "top_role": get("TOP_ROLE"),
        "match_pct": get("MATCH_PCT"),
        "why_right": get("WHY_RIGHT"),
        "next_steps": get_list("NEXT_STEPS"),
        "skill_gaps": get_list("SKILL_GAPS"),
        "resume_add": get_list("RESUME_ADD"),
        "career_path": get_list("CAREER_PATH"),
        "runner_up": get("RUNNER_UP"),
        "runner_up_why": get("RUNNER_UP_WHY"),
    }


def match_companies(cv_text: str, all_matches: list) -> list:
    """Match CV to real companies from JD dataset."""
    from rag import load_roles
    
    cv_lower = cv_text.lower()
    companies = []
    
    try:
        roles = load_roles()
        
        for role in roles:
            if "company" in role and role.get("company"):
                # Calculate match score
                required_skills = role.get("skills", [])
                skills_found = sum(1 for s in required_skills if s.lower() in cv_lower)
                match_score = (skills_found / max(len(required_skills), 1)) * 100
                
                # Get salary info
                salary = role.get("salary", {})
                salary_range = ""
                if salary.get("junior") and salary.get("junior") != "0":
                    salary_range = f"৳{salary['junior']}"
                    if salary.get("mid"):
                        salary_range += f" - ৳{salary['mid']}"
                
                companies.append({
                    "name": role["company"],
                    "role": role.get("title", role.get("role")),
                    "match_score": round(match_score, 1),
                    "salary_range": salary_range,
                    "location": role.get("location", "Dhaka"),
                    "category": role.get("category", "Entry-level"),
                    "skills": required_skills[:5],
                    "description": role.get("description", "")[:150]
                })
        
        companies.sort(key=lambda x: x["match_score"], reverse=True)
        return companies[:6]
    except:
        return []


def render_navbar():
    """Render navigation bar."""
    name = st.session_state.candidate_name or "Guest"
    first_name = name.split()[0] if name else "Guest"
    
    col1, col2, col3 = st.columns([2, 3, 1])
    
    with col1:
        st.markdown('<div class="nav-logo">📄 CV Analyzer</div>', unsafe_allow_html=True)
    
    with col2:
        cols = st.columns([1, 1, 1, 1, 1])
        pages = ["🏠 Home", "📊 Analysis", "💬 Chat"]
        page_keys = ["home", "analyze", "chat"]
        
        for idx, (label, key) in enumerate(zip(pages, page_keys)):
            with cols[idx]:
                if st.button(label, key=f"nav_{key}", use_container_width=True):
                    st.session_state.page = key
                    if key == "analyze":
                        st.session_state.analysis_tab = "overview"
                    st.rerun()
    
    with col3:
        st.markdown(f'<div class="user-name">👋 {first_name}</div>', unsafe_allow_html=True)
    
    st.markdown("<hr style='margin: 0.5rem 0 1rem 0; border-color: #1a1a30;'>", unsafe_allow_html=True)


def render_home():
    """Home page with upload section."""
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 2rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 2.5rem; font-weight: 800;'>
            Match Your CV to <span style='background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>AI/Data Roles</span>
        </div>
        <p style='color: #64748b; margin-top: 0.5rem;'>
            Upload your CV and get personalized career matches using AI + vector search
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown('<div class="card"><div class="card-title">📄 Upload Your CV</div>', unsafe_allow_html=True)
        uploaded_cv = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed", key="cv_uploader")
        if uploaded_cv:
            st.success(f"✅ {uploaded_cv.name}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">📝 Job Description (Optional)</div>', unsafe_allow_html=True)
        jd_input = st.text_area("Paste JD here", height=150, placeholder="Paste a job description to bias the match toward a specific role...", label_visibility="collapsed", key="jd_input")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analyze button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        if st.button("⚡ Analyze My CV", use_container_width=True, type="primary"):
            if uploaded_cv:
                with st.spinner("📖 Reading CV..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_cv.read())
                        tmp_path = tmp.name
                    
                    cv_text = extract_cv_text(tmp_path)
                    os.unlink(tmp_path)
                
                st.session_state.cv_text = cv_text
                st.session_state.jd_text = jd_input.strip() if jd_input else ""
                st.session_state.messages = []
                
                with st.spinner("🔍 Matching with FAISS..."):
                    st.session_state.retrieved = retrieve_context(cv_text, jd_input.strip() if jd_input else "", k=5)
                
                with st.spinner("🏢 Finding matching companies..."):
                    st.session_state.matched_companies = match_companies(cv_text, [])
                
                with st.spinner("🤖 Generating AI analysis..."):
                    st.session_state.agent = build_agent(cv_text, jd_input.strip() if jd_input else "", st.session_state.candidate_name)
                    
                    raw = run_agent(st.session_state.agent,
                        "Analyse this candidate's CV and give a full career match. "
                        "Follow EXACTLY these tags:\n"
                        "TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
                        "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY\n"
                        "Be specific — reference actual CV skills throughout."
                    )
                    st.session_state.analysis_raw = raw
                
                st.session_state.page = "analyze"
                st.session_state.analysis_tab = "overview"
                st.success("✅ Analysis complete!")
                st.rerun()
            else:
                st.warning("⚠️ Please upload your CV first")
    
    # Features section
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-top: 2rem;'>
        <div class='card' style='text-align: center;'>
            <div style='font-size: 2rem;'>🎯</div>
            <div style='font-weight: 600; color: #cbd5e1;'>Smart Matching</div>
            <div style='font-size: 0.75rem; color: #64748b;'>FAISS vector search</div>
        </div>
        <div class='card' style='text-align: center;'>
            <div style='font-size: 2rem;'>🏢</div>
            <div style='font-weight: 600; color: #cbd5e1;'>Company Match</div>
            <div style='font-size: 0.75rem; color: #64748b;'>Real BD companies</div>
        </div>
        <div class='card' style='text-align: center;'>
            <div style='font-size: 2rem;'>💰</div>
            <div style='font-weight: 600; color: #cbd5e1;'>Salary Insights</div>
            <div style='font-size: 0.75rem; color: #64748b;'>Market benchmarks</div>
        </div>
        <div class='card' style='text-align: center;'>
            <div style='font-size: 2rem;'>💬</div>
            <div style='font-weight: 600; color: #cbd5e1;'>AI Chat</div>
            <div style='font-size: 0.75rem; color: #64748b;'>Ask anything</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_score_card(score: float, level: str, recommendation: str):
    """Render a beautiful score card with range-based feedback."""
    
    if score < 30:
        color = "#ef4444"
        bg_color = "rgba(239, 68, 68, 0.1)"
        icon = "🔴"
        status = "Not Ready for AI/ML Roles"
        next_action = "Focus on building foundational skills"
    elif score < 60:
        color = "#f59e0b"
        bg_color = "rgba(245, 158, 11, 0.1)"
        icon = "🟡"
        status = "Building Foundation"
        next_action = "Keep learning and building projects"
    elif score < 75:
        color = "#10b981"
        bg_color = "rgba(16, 185, 129, 0.1)"
        icon = "🟢"
        status = "Getting Ready"
        next_action = "Start applying to junior roles"
    else:
        color = "#06b6d4"
        bg_color = "rgba(6, 182, 212, 0.1)"
        icon = "🌟"
        status = "Ready to Apply!"
        next_action = "You're qualified — start applying!"
    
    st.markdown(f"""
    <div style='background: {bg_color}; border: 1px solid {color}; border-radius: 20px; padding: 1.5rem; margin: 1rem 0;'>
        <div style='display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap;'>
            <div>
                <div style='font-size: 0.7rem; color: {color}; text-transform: uppercase; letter-spacing: 0.1em;'>
                    {icon} AI/ML READINESS SCORE
                </div>
                <div style='font-size: 3rem; font-weight: 800; color: {color};'>
                    {score}%
                </div>
                <div style='font-weight: 600; color: #f1f5f9;'>{status}</div>
            </div>
            <div style='max-width: 300px;'>
                <div style='color: #94a3b8; font-size: 0.85rem;'>{recommendation}</div>
                <div style='margin-top: 0.5rem;'>
                    <span style='background: {color}; color: white; padding: 0.2rem 0.8rem; border-radius: 20px; font-size: 0.7rem;'>
                        🎯 {next_action}
                    </span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_analysis_tabs():
    """Render smart analysis tabs."""
    tabs = [
        ("📊 Overview", "overview"),
        ("🏢 Companies", "companies"),
        ("💰 Salary", "salary"),
        ("🔧 Skills", "skills"),
        ("🗺️ Career Path", "path")
    ]
    
    cols = st.columns(len(tabs))
    for idx, (label, key) in enumerate(tabs):
        with cols[idx]:
            is_active = st.session_state.analysis_tab == key
            btn_style = "active" if is_active else ""
            if st.button(label, key=f"tab_{key}", use_container_width=True):
                st.session_state.analysis_tab = key
                st.rerun()


def render_overview_tab(retrieved, all_matches, top_match, parsed, readiness):
    """Overview tab - hero section + role breakdown."""
    # Hero Section
    top_role = parsed.get("top_role") or top_match.get("title", top_match.get("role", "AI Professional"))
    match_pct = parsed.get("match_pct") or str(top_match.get("match_pct", 0))
    
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0 1.5rem 0;'>
        <div class='hero-match'>{match_pct}% Match</div>
        <div style='font-family: Syne, sans-serif; font-size: 1.5rem; font-weight: 700; color: #f1f5f9;'>
            {top_role}
        </div>
        <p style='color: #94a3b8; max-width: 600px; margin: 0.5rem auto;'>
            {parsed.get("why_right", top_match.get("why_good_fit", "Great alignment with your skills and experience"))}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Score card
    if readiness:
        render_score_card(
            readiness.get("total_score", 0),
            readiness.get("level", "Not Ready"),
            readiness.get("recommendation", "")
        )
    
    # All matches grid
    if all_matches:
        st.markdown("### 📊 Role Breakdown")
        cols = st.columns(min(len(all_matches), 4))
        for i, role in enumerate(all_matches[:4]):
            with cols[i]:
                skills = role.get("skills", [])[:4]
                skills_html = "".join(f"<span class='skill-chip'>{s}</span>" for s in skills)
                
                match = role['match_pct']
                if match < 30:
                    badge_color = "#ef4444"
                elif match < 60:
                    badge_color = "#f59e0b"
                else:
                    badge_color = "#10b981"
                    
                st.markdown(f"""
                <div class='card'>
                    <div style='font-weight: 700; color: #cbd5e1;'>{role.get('title', role.get('role', ''))}</div>
                    <div style='font-size: 1.5rem; font-weight: 800; color: {badge_color};'>{match}%</div>
                    <div style='margin-top: 0.5rem;'>{skills_html}</div>
                </div>
                """, unsafe_allow_html=True)
    
    # Runner up
    runner_up = parsed.get("runner_up") or (all_matches[1].get("title") if len(all_matches) > 1 else "")
    if runner_up:
        runner_up_pct = all_matches[1].get("match_pct", 0) if len(all_matches) > 1 else 0
        st.info(f"🥈 **Runner-up:** {runner_up} ({runner_up_pct}%)")


def render_companies_tab():
    """Companies tab - matching companies from real JD dataset."""
    companies = st.session_state.matched_companies
    
    if not companies:
        st.info("No company matches found. Try uploading a CV with more AI/ML skills.")
        return
    
    st.markdown("### 🏢 Top Matching Companies in Bangladesh")
    st.markdown("*Based on real job postings from LinkedIn (2026)*")
    
    for comp in companies[:5]:
        match_color = "#10b981" if comp["match_score"] > 60 else "#f59e0b" if comp["match_score"] > 30 else "#ef4444"
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"""
            <div class='company-card'>
                <div style='display: flex; justify-content: space-between; align-items: start;'>
                    <div>
                        <div style='font-weight: 800; color: #f1f5f9; font-size: 1rem;'>{comp['name']}</div>
                        <div style='font-size: 0.75rem; color: #a855f7;'>{comp['role']} • {comp['category']}</div>
                        <div style='font-size: 0.7rem; color: #64748b; margin-top: 0.25rem;'>📍 {comp['location']}</div>
                    </div>
                    <div style='text-align: right;'>
                        <div style='font-size: 1.3rem; font-weight: 800; color: {match_color};'>{comp['match_score']}%</div>
                        <div style='font-size: 0.65rem; color: #64748b;'>Match</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            # Skills
            if comp.get("skills"):
                skills_html = "".join(f"<span class='skill-chip'>{s[:20]}</span>" for s in comp["skills"][:4])
                st.markdown(f"<div style='margin-top: 0.5rem;'>{skills_html}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if comp.get("salary_range") and comp["salary_range"]:
                st.markdown(f"""
                <div style='background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 12px; padding: 0.8rem; text-align: center; height: 100%;'>
                    <div style='font-size: 0.65rem; color: #fbbf24;'>💰 Est. Salary</div>
                    <div style='font-size: 0.8rem; font-weight: 700; color: #fbbf24;'>{comp['salary_range']}</div>
                    <div style='font-size: 0.6rem; color: #64748b;'>per month</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 12px; padding: 0.8rem; text-align: center; height: 100%;'>
                    <div style='font-size: 0.65rem; color: #64748b;'>🎓 Internship</div>
                    <div style='font-size: 0.7rem; color: #a855f7;'>Growth opportunity</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("""
    <div style='background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1rem; margin-top: 1rem;'>
        <div style='font-size: 0.75rem; color: #64748b;'>
            💡 <strong>Pro Tip:</strong> These matches are based on your skills alignment with real job postings from LinkedIn Bangladesh (2026). 
            Click on the job links in the chat to apply directly!
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_salary_tab(retrieved, all_matches):
    """Salary tab - salary insights and benchmarks."""
    st.markdown("### 💰 Salary Insights")
    
    # Extract salary data from matches
    salaries = []
    for role in all_matches[:4]:
        if role.get("salary_min") and role.get("salary_max"):
            salaries.append({
                "role": role.get("title", role.get("role")),
                "min": role.get("salary_min", 0),
                "max": role.get("salary_max", 0)
            })
    
    if salaries:
        st.markdown("#### 📊 Market Salary Ranges (BDT/month)")
        
        for sal in salaries:
            st.markdown(f"""
            <div style='margin-bottom: 1rem;'>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #cbd5e1; font-weight: 500;'>{sal['role']}</span>
                    <span style='color: #fbbf24;'>৳{sal['min']:,} - ৳{sal['max']:,}</span>
                </div>
                <div style='background: #1a1a30; border-radius: 10px; height: 8px; margin-top: 0.3rem;'>
                    <div style='width: {min(100, (sal['max']/100000)*100)}%; background: linear-gradient(90deg, #f59e0b, #fbbf24); height: 100%; border-radius: 10px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("#### 📈 Salary by Experience Level")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <div style='font-size: 0.7rem; color: #64748b;'>🎓 Intern/Junior</div>
            <div style='font-size: 1.2rem; font-weight: 800; color: #f59e0b;'>15k - 35k</div>
            <div style='font-size: 0.65rem; color: #64748b;'>BDT/month</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <div style='font-size: 0.7rem; color: #64748b;'>👨‍💻 Mid-Level</div>
            <div style='font-size: 1.2rem; font-weight: 800; color: #10b981;'>40k - 70k</div>
            <div style='font-size: 0.65rem; color: #64748b;'>BDT/month</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <div style='font-size: 0.7rem; color: #64748b;'>🚀 Senior</div>
            <div style='font-size: 1.2rem; font-weight: 800; color: #06b6d4;'>70k - 150k+</div>
            <div style='font-size: 0.65rem; color: #64748b;'>BDT/month</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("💡 **Note:** Salaries are based on real job postings from Bangladesh market (2026). International remote roles may pay higher.")


def render_skills_tab(parsed, retrieved):
    """Skills tab - gaps and recommendations."""
    st.markdown("### 🔧 Skills Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        skill_gaps = parsed.get("skill_gaps") or retrieved.get("skill_gaps", [])
        if skill_gaps:
            st.markdown("#### 🔴 Skill Gaps to Fill")
            for gap in skill_gaps[:6]:
                st.markdown(f"""
                <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;'>
                    <span style='color: #fca5a5; font-weight: 500;'>{gap}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("✅ No major skill gaps! Your CV is well-aligned.")
    
    with col2:
        resume_skills = parsed.get("resume_add") or retrieved.get("resume_skills", [])
        if resume_skills:
            st.markdown("#### ✅ Resume Recommendations")
            for skill in resume_skills[:6]:
                st.markdown(f"""
                <div style='background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 8px; padding: 0.5rem; margin-bottom: 0.5rem;'>
                    <span style='color: #86efac;'>+ {skill}</span>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("#### 📚 Learning Resources")
    
    resource_cols = st.columns(3)
    resources = [
        ("🐍 Python", "https://www.python.org/about/gettingstarted/"),
        ("🤗 Hugging Face", "https://huggingface.co/learn"),
        ("📊 Kaggle", "https://www.kaggle.com/learn"),
        ("🔷 PyTorch", "https://pytorch.org/tutorials/"),
        ("🧠 Fast.ai", "https://www.fast.ai/"),
        ("🎓 DeepLearning.AI", "https://www.deeplearning.ai/")
    ]
    
    for idx, (name, url) in enumerate(resources):
        with resource_cols[idx % 3]:
            st.markdown(f"[{name}]({url})", unsafe_allow_html=True)


def render_path_tab(parsed):
    """Career path tab."""
    st.markdown("### 🗺️ Your Career Path")
    
    career_path = parsed.get("career_path", [])
    
    if career_path:
        for i, step in enumerate(career_path[:5]):
            st.markdown(f"""
            <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                <div style='width: 32px; height: 32px; background: linear-gradient(135deg, #7c3aed, #db2777); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 700; margin-right: 1rem;'>
                    {i+1}
                </div>
                <div style='flex: 1; background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 12px; padding: 0.8rem 1rem;'>
                    <span style='color: #cbd5e1;'>{step}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Complete analysis to see your personalized career path.")
    
    st.markdown("---")
    st.markdown("#### 🎯 Quick Action Items")
    
    action_cols = st.columns(2)
    with action_cols[0]:
        st.markdown("""
        - 📝 Update LinkedIn profile
        - 🔗 Build portfolio projects
        - 📚 Take relevant certifications
        """)
    with action_cols[1]:
        st.markdown("""
        - 🤝 Network with industry professionals
        - 📊 Contribute to open source
        - 🎯 Apply to matching companies
        """)


def render_analysis():
    """Display analysis results with smart tabs."""
    if not st.session_state.cv_text:
        st.info("👈 Please upload your CV on the Home page first")
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    retrieved = st.session_state.retrieved or {}
    all_matches = retrieved.get("all_matches", [])
    top_match = retrieved.get("top_match", {})
    readiness = retrieved.get("readiness", {})
    parsed = parse_analysis(st.session_state.analysis_raw or "") if st.session_state.analysis_raw else {}
    
    # Render smart tabs
    render_analysis_tabs()
    
    # Render selected tab
    if st.session_state.analysis_tab == "overview":
        render_overview_tab(retrieved, all_matches, top_match, parsed, readiness)
    elif st.session_state.analysis_tab == "companies":
        render_companies_tab()
    elif st.session_state.analysis_tab == "salary":
        render_salary_tab(retrieved, all_matches)
    elif st.session_state.analysis_tab == "skills":
        render_skills_tab(parsed, retrieved)
    elif st.session_state.analysis_tab == "path":
        render_path_tab(parsed)


def render_chat():
    """Chat interface."""
    if not st.session_state.agent:
        st.info("👈 Please analyze your CV first on the Home page")
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <div style='font-size: 1.2rem; font-weight: 600; color: #f1f5f9;'>💬 Ask Your Career Advisor</div>
        <p style='color: #64748b;'>Ask about skills, salaries, job search, or career path</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Quick questions
    st.markdown("#### Quick Questions")
    q_cols = st.columns(3)
    questions = [
        "What roles am I best suited for?",
        "What skills am I missing?",
        "What should I add to my resume?",
        "Which companies should I apply to?",
        "What's the salary range for me?",
        "What is my career path?"
    ]
    for i, q in enumerate(questions):
        with q_cols[i % 3]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        resp = run_agent(st.session_state.agent, q)
                        st.markdown(resp)
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                st.rerun()
    
    st.markdown("---")
    
    # Chat input
    if prompt := st.chat_input("Ask anything about your career..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = run_agent(st.session_state.agent, prompt)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
        st.rerun()


# ============================================================
# MAIN APP
# ============================================================
def main():
    # Name entry screen
    if not st.session_state.name_entered:
        st.markdown("""
        <div style='text-align: center; padding: 3rem 0;'>
            <div style='font-family: Syne, sans-serif; font-size: 3rem; font-weight: 800;'>
                CV <span style='background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Analyzer</span>
            </div>
            <p style='color: #64748b; margin: 1rem 0;'>AI-powered career matching for data & AI roles</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
            
            with st.form(key="name_form"):
                name_val = st.text_input(
                    "name", 
                    placeholder="e.g. Talha Jobayer", 
                    label_visibility="collapsed", 
                    key="name_field"
                )
                st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
                submitted = st.form_submit_button("Get Started →", use_container_width=True, type="primary")
                
                if submitted and name_val.strip():
                    st.session_state.candidate_name = name_val.strip()
                    st.session_state.name_entered = True
                    st.rerun()
                elif submitted:
                    st.warning("Please enter your name")
            
            st.markdown('</div>', unsafe_allow_html=True)
        st.stop()
    
    # Render navbar and page content
    render_navbar()
    
    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "analyze":
        render_analysis()
    elif st.session_state.page == "chat":
        render_chat()


if __name__ == "__main__":
    main()
