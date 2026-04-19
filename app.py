"""
app.py — CV Analyzer with Dual Mode (CV Analysis + JD Matching)
"""

import streamlit as st
import tempfile
import os
import re

# Import at the top
from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context, load_roles

st.set_page_config(
    page_title="CV Analyzer | AI Career Match",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS STYLES - Clean minimal design
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

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
footer {visibility: hidden;}

/* Navigation Bar - Single, clean, at top */
.nav-bar {
    background: rgba(15, 15, 32, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #2d2d5a;
    padding: 0.6rem 2rem;
    margin-bottom: 1.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.nav-left {
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    cursor: pointer;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
}
.nav-logo:hover {
    background: rgba(168, 85, 247, 0.1);
}
.nav-btn {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.3rem 0.8rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}
.nav-btn:hover {
    background: rgba(168, 85, 247, 0.1);
    color: #a855f7;
}
.nav-user {
    background: transparent;
    border: 1px solid #2d2d5a;
    color: #94a3b8;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    cursor: pointer;
    transition: all 0.2s;
    display: flex;
    align-items: center;
    gap: 0.3rem;
}
.nav-user:hover {
    background: rgba(168, 85, 247, 0.1);
    border-color: #a855f7;
    color: #a855f7;
}

/* Dropdown menu - small, positioned properly */
.dropdown {
    position: absolute;
    right: 2rem;
    background: #0f0f20;
    border: 1px solid #2d2d5a;
    border-radius: 12px;
    padding: 0.5rem;
    min-width: 160px;
    z-index: 1000;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}
.dropdown-item {
    padding: 0.5rem 0.8rem;
    font-size: 0.8rem;
    color: #cbd5e1;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}
.dropdown-item:hover {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
}
.dropdown-divider {
    height: 1px;
    background: #2d2d5a;
    margin: 0.3rem 0;
}

/* Cards */
.card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 1.2rem;
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
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Mode Cards */
.mode-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin: 1.5rem 0;
}
.mode-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s;
}
.mode-card:hover {
    border-color: #7c3aed;
    transform: translateY(-2px);
}
.mode-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.mode-title { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.3rem; }
.mode-desc { font-size: 0.7rem; color: #64748b; }

/* JD Match Card */
.jd-match-card {
    background: linear-gradient(135deg, #0f0f20 0%, #1a0f35 100%);
    border: 1px solid #2d2060;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.jd-match-score { font-size: 3rem; font-weight: 800; }

/* Welcome Card */
.welcome-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    max-width: 400px;
    margin: 2rem auto;
}

/* Company Card */
.company-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 0.8rem;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_submode" not in st.session_state:
    st.session_state.analysis_submode = None
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
if "jd_match_result" not in st.session_state:
    st.session_state.jd_match_result = None
if "show_dropdown" not in st.session_state:
    st.session_state.show_dropdown = False


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
    cv_lower = cv_text.lower()
    companies = []
    
    try:
        roles = load_roles()
        for role in roles:
            if "company" in role and role.get("company"):
                required_skills = role.get("skills", [])
                skills_found = sum(1 for s in required_skills if s.lower() in cv_lower)
                match_score = (skills_found / max(len(required_skills), 1)) * 100
                
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
                })
        
        companies.sort(key=lambda x: x["match_score"], reverse=True)
        return companies[:6]
    except:
        return []


def calculate_jd_match_score(cv_text: str, jd_text: str) -> dict:
    """Calculate how well CV matches a specific JD."""
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    jd_keywords = set()
    keyword_patterns = [
        r'(?:experience with|knowledge of|proficiency in|familiarity with)\s+([a-z][a-z\s]+?)(?=\.|,|\n)',
        r'(?:must have|required|essential|preferred)\s+([a-z][a-z\s]+?)(?=\.|,|\n)',
        r'\b(python|sql|tensorflow|pytorch|langchain|rag|llm|nlp|computer vision|docker|kubernetes|aws|gcp|azure|mlflow|scikit-learn|pandas|numpy|git)\b'
    ]
    
    for pattern in keyword_patterns:
        matches = re.findall(pattern, jd_lower, re.IGNORECASE)
        for match in matches:
            words = match.strip().split()[:3]
            jd_keywords.add(' '.join(words))
    
    skill_list = ['python', 'sql', 'tensorflow', 'pytorch', 'langchain', 'rag', 'llm', 'nlp', 'docker', 'kubernetes']
    for skill in skill_list:
        if skill in jd_lower:
            jd_keywords.add(skill)
    
    found_keywords = []
    missing_keywords = []
    
    for kw in jd_keywords:
        if kw in cv_lower or kw.replace(" ", "") in cv_lower:
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)
    
    match_pct = int((len(found_keywords) / max(len(jd_keywords), 1)) * 100)
    
    matching_companies = []
    try:
        roles = load_roles()
        for role in roles:
            role_jd = role.get("jd_text", "").lower()
            if any(kw in role_jd for kw in list(jd_keywords)[:10]):
                salary = role.get("salary", {})
                salary_str = f"৳{salary['junior']}" if salary.get("junior") else ""
                matching_companies.append({
                    "name": role.get("company", "Unknown"),
                    "role": role.get("title", role.get("role")),
                    "salary": salary_str,
                    "location": role.get("location", "Dhaka")
                })
    except:
        pass
    
    return {
        "match_pct": min(95, match_pct),
        "matched_keywords": found_keywords[:15],
        "missing_keywords": missing_keywords[:15],
        "found_count": len(found_keywords),
        "total_keywords": len(jd_keywords),
        "matching_companies": matching_companies[:4]
    }


def render_navbar():
    """Single clean navigation bar - no duplication"""
    name = st.session_state.candidate_name or "Guest"
    first_name = name.split()[0] if name else "Guest"
    
    # Use HTML for clean nav bar
    st.markdown(f'''
    <div class="nav-bar">
        <div class="nav-left">
            <span class="nav-logo" onclick="location.reload()">📄 CV Analyzer</span>
            <button class="nav-btn" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'home'}}, '*')">🏠 Home</button>
            <button class="nav-btn" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'analyze'}}, '*')">📊 Analysis</button>
            <button class="nav-btn" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'chat'}}, '*')">💬 Chat</button>
            <button class="nav-btn" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'career'}}, '*')">🎯 Career Rec</button>
        </div>
        <div class="nav-right">
            <button class="nav-user" id="userBtn" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'toggle_dropdown'}}, '*')">👋 {first_name} ▼</button>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Handle navigation via Streamlit buttons (cleaner)
    col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
    
    with col1:
        if st.button("📄 CV Analyzer", key="nav_logo", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.analysis_submode = None
            st.rerun()
    
    with col2:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.analysis_submode = None
            st.rerun()
    
    with col3:
        if st.button("📊 Analysis", key="nav_analysis", use_container_width=True):
            if st.session_state.cv_text:
                st.session_state.page = "analyze"
                st.session_state.analysis_tab = "overview"
            else:
                st.warning("Please analyze your CV first")
            st.rerun()
    
    with col4:
        if st.button("💬 Chat", key="nav_chat", use_container_width=True):
            if st.session_state.agent:
                st.session_state.page = "chat"
            else:
                st.warning("Please analyze your CV first")
            st.rerun()
    
    with col5:
        if st.button("🎯 Career Rec", key="nav_career", use_container_width=True):
            if st.session_state.cv_text:
                st.session_state.page = "career"
            else:
                st.warning("Please analyze your CV first")
            st.rerun()
    
    with col6:
        if st.button(f"👋 {first_name} ▼", key="nav_user", use_container_width=True):
            st.session_state.show_dropdown = not st.session_state.show_dropdown
            st.rerun()
    
    # Small dropdown menu when user clicks name
    if st.session_state.show_dropdown:
        st.markdown(f'''
        <div class="dropdown">
            <div style="padding: 0.3rem 0.8rem; font-size: 0.7rem; color: #64748b;">Signed in as</div>
            <div style="padding: 0.3rem 0.8rem; font-size: 0.8rem; font-weight: 600; color: #cbd5e1;">{name}</div>
            <div class="dropdown-divider"></div>
            <div class="dropdown-item" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'signout'}}, '*')">🚪 Sign Out / Change Name</div>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚪 Sign Out", key="signout_btn", use_container_width=True):
            # Reset all session state
            st.session_state.name_entered = False
            st.session_state.cv_text = None
            st.session_state.analysis_raw = None
            st.session_state.retrieved = None
            st.session_state.agent = None
            st.session_state.messages = []
            st.session_state.show_dropdown = False
            st.rerun()
    
    st.markdown("<hr style='margin: 0.5rem 0 1rem 0; border-color: #1a1a30;'>", unsafe_allow_html=True)


def render_welcome_screen():
    """Clean welcome screen"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 3rem; font-weight: 800;'>
            CV <span style='background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Analyzer</span>
        </div>
        <p style='color: #64748b; margin: 0.5rem 0;'>AI-powered career matching for data & AI roles</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.2rem; font-weight: 600; color: #cbd5e1; margin-bottom: 1rem;">👋 Welcome!</div>', unsafe_allow_html=True)
        
        with st.form(key="name_form"):
            name_val = st.text_input("Your name", placeholder="Talha Jobayer", label_visibility="collapsed")
            submitted = st.form_submit_button("✨ Start Analysis →", use_container_width=True, type="primary")
            
            if submitted and name_val and name_val.strip():
                st.session_state.candidate_name = name_val.strip()
                st.session_state.name_entered = True
                st.rerun()
            elif submitted:
                st.error("Please enter your name")
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_home():
    """Home page with two options"""
    name = st.session_state.candidate_name
    first_name = name.split()[0] if name else "there"
    
    st.markdown(f"""
    <div style='text-align: center; margin: 1rem 0 1.5rem 0;'>
        <div style='font-size: 1.5rem; font-weight: 700; color: #f1f5f9;'>Hello, {first_name}! 👋</div>
        <p style='color: #64748b;'>Ready to analyze your CV?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>📄</div>
            <div class='mode-title'>Analyze My CV</div>
            <div class='mode-desc'>Get matched with AI/ML roles from our knowledge base</div>
            <div class='mode-desc' style='margin-top: 0.5rem; color: #a855f7;'>✨ Skills | Roles | Salary | Career Path</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📄 Analyze My CV", key="btn_cv", use_container_width=True):
            st.session_state.analysis_submode = "cv_analysis"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>🎯</div>
            <div class='mode-title'>Match with Job Description</div>
            <div class='mode-desc'>Paste a JD and see how well your CV matches</div>
            <div class='mode-desc' style='margin-top: 0.5rem; color: #10b981;'>✨ Match % | Missing Skills | Companies</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Match with JD", key="btn_jd", use_container_width=True):
            st.session_state.analysis_submode = "jd_match"
            st.rerun()
    
    # Show selected mode UI
    if st.session_state.analysis_submode == "cv_analysis":
        st.markdown("---")
        uploaded_cv = st.file_uploader("Upload your CV (PDF)", type=["pdf"], key="cv_upload")
        if uploaded_cv and st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            process_cv_analysis(uploaded_cv, "")
    
    elif st.session_state.analysis_submode == "jd_match":
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_cv = st.file_uploader("Your CV (PDF)", type=["pdf"], key="jd_cv")
        with col2:
            jd_text = st.text_area("Job Description", height=150, placeholder="Paste JD here...")
        if uploaded_cv and jd_text and st.button("🎯 Calculate Match", type="primary", use_container_width=True):
            process_jd_match(uploaded_cv, jd_text)


def process_cv_analysis(uploaded_cv, jd_text):
    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_cv.read())
            cv_text = extract_cv_text(tmp.name)
            os.unlink(tmp.name)
        
        st.session_state.cv_text = cv_text
        st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
        st.session_state.matched_companies = match_companies(cv_text, [])
        st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
        
        raw = run_agent(st.session_state.agent,
            "Analyse this candidate's CV and give a full career match. "
            "Follow EXACTLY these tags: TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
            "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY")
        st.session_state.analysis_raw = raw
    
    st.session_state.page = "analyze"
    st.rerun()


def process_jd_match(uploaded_cv, jd_text):
    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_cv.read())
            cv_text = extract_cv_text(tmp.name)
            os.unlink(tmp.name)
        
        st.session_state.cv_text = cv_text
        st.session_state.jd_match_result = calculate_jd_match_score(cv_text, jd_text)
        st.session_state.matched_companies = match_companies(cv_text, [])
    
    st.session_state.page = "jd_result"
    st.rerun()


def render_jd_result():
    result = st.session_state.jd_match_result
    if not result:
        st.warning("No result found")
        return
    
    match_pct = result["match_pct"]
    color = "#ef4444" if match_pct < 30 else "#f59e0b" if match_pct < 60 else "#10b981" if match_pct < 80 else "#06b6d4"
    status = ["Low Match", "Partial Match", "Good Match", "Excellent Match!"][[30,60,80,100].index(min([x for x in [30,60,80,100] if x > match_pct]))] if match_pct < 80 else "Excellent Match!"
    
    st.markdown(f"""
    <div class='jd-match-card'>
        <div style='color:{color};font-size:0.8rem;'>JD Match Score</div>
        <div class='jd-match-score' style='color:{color};'>{match_pct}%</div>
        <div style='font-size:1.2rem;color:#f1f5f9;'>{status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Keywords Found", f"{result['found_count']}/{result['total_keywords']}")
    with col2:
        if result.get("matching_companies"):
            st.markdown("**🏢 Similar Companies**")
            for comp in result["matching_companies"][:3]:
                st.markdown(f"- **{comp['name']}** - {comp['role']}")
    
    if result.get("missing_keywords"):
        st.markdown("**❌ Missing Keywords to Add**")
        st.markdown(" ".join(f"<span class='skill-chip gap-chip'>{kw}</span>" for kw in result["missing_keywords"][:8]), unsafe_allow_html=True)
    
    if st.button("← New Match", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.analysis_submode = "jd_match"
        st.rerun()


def render_analysis():
    if not st.session_state.cv_text:
        st.warning("Please analyze your CV first")
        return
    
    retrieved = st.session_state.retrieved or {}
    all_matches = retrieved.get("all_matches", [])
    top_match = retrieved.get("top_match", {})
    readiness = retrieved.get("readiness", {})
    parsed = parse_analysis(st.session_state.analysis_raw or "") if st.session_state.analysis_raw else {}
    
    # Simple tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "🏢 Companies", "💰 Salary", "🔧 Skills", "🗺️ Path"])
    
    with tab1:
        top_role = parsed.get("top_role") or top_match.get("title", "AI Professional")
        match_pct = parsed.get("match_pct") or str(top_match.get("match_pct", 0))
        st.markdown(f"<div class='hero-match' style='text-align:center;'>{match_pct}% Match</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{top_role}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:#94a3b8;'>{parsed.get('why_right', 'Great alignment!')}</p>", unsafe_allow_html=True)
        
        score = readiness.get("total_score", 0)
        if score < 30:
            st.warning(f"🔴 Readiness Score: {score}% - Focus on fundamentals")
        elif score < 60:
            st.warning(f"🟡 Readiness Score: {score}% - Keep building")
        else:
            st.success(f"🟢 Readiness Score: {score}% - Ready to apply!")
    
    with tab2:
        companies = st.session_state.matched_companies
        for comp in companies[:5]:
            st.markdown(f"**{comp['name']}** - {comp['role']} ({comp['match_score']}% match)\n📍 {comp['location']}")
            if comp.get("salary_range"):
                st.caption(f"💰 {comp['salary_range']}/month")
    
    with tab3:
        for role in all_matches[:4]:
            if role.get("salary_min"):
                st.metric(role.get("title"), f"৳{role['salary_min']:,} - ৳{role['salary_max']:,}")
    
    with tab4:
        gaps = parsed.get("skill_gaps") or retrieved.get("skill_gaps", [])
        for g in gaps[:6]:
            st.markdown(f"<span class='skill-chip gap-chip'>{g}</span>", unsafe_allow_html=True)
    
    with tab5:
        path = parsed.get("career_path", [])
        for i, p in enumerate(path[:5]):
            st.markdown(f"{i+1}. {p}")


def render_chat():
    if not st.session_state.agent:
        st.warning("Please analyze your CV first")
        return
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    if prompt := st.chat_input("Ask about your career..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = run_agent(st.session_state.agent, prompt)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
        st.rerun()


def render_career_rec():
    if not st.session_state.cv_text:
        st.warning("Please analyze your CV first")
        return
    
    retrieved = st.session_state.retrieved or {}
    readiness = retrieved.get("readiness", {})
    score = readiness.get("total_score", 0)
    
    if score < 30:
        st.warning("🔴 **Focus on Fundamentals**\n\n- Complete Python basics\n- Take ML courses\n- Build small projects")
    elif score < 60:
        st.warning("🟡 **Building Momentum**\n\n- Take advanced courses\n- Build portfolio projects\n- Get certifications")
    else:
        st.success("🟢 **Ready for Job Search!**\n\n- Update LinkedIn\n- Apply to matching companies\n- Prepare for interviews")


# ============================================================
# MAIN
# ============================================================
def main():
    if not st.session_state.name_entered:
        render_welcome_screen()
        return
    
    render_navbar()
    
    if st.session_state.page == "home":
        render_home()
    elif st.session_state.page == "analyze":
        render_analysis()
    elif st.session_state.page == "chat":
        render_chat()
    elif st.session_state.page == "jd_result":
        render_jd_result()
    elif st.session_state.page == "career":
        render_career_rec()


if __name__ == "__main__":
    main()
