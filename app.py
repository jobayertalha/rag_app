"""
app.py — CV Analyzer with Dual Mode (CV Analysis + JD Matching)
"""

import streamlit as st
import tempfile
import os
import re

# Import at the top (not inside functions) - OPTIMIZED
from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context, load_roles

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

/* Hide default Streamlit elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Navigation Bar */
.top-nav {
    background: rgba(15, 15, 32, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #2d2d5a;
    padding: 0.75rem 2rem;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    cursor: pointer;
}
.nav-links {
    display: flex;
    gap: 0.25rem;
    align-items: center;
}
.nav-link {
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
.nav-link:hover {
    background: rgba(168, 85, 247, 0.1);
    color: #a855f7;
}
.nav-link.active {
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: white;
}
.user-name {
    font-size: 0.8rem;
    color: #64748b;
    padding: 0.4rem 1rem;
    cursor: pointer;
    border-left: 1px solid #2d2d5a;
    margin-left: 0.5rem;
}
.user-name:hover {
    color: #a855f7;
}

/* Main content padding for fixed nav */
.main-content {
    padding-top: 70px;
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

/* Mode Cards */
.mode-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-top: 1rem;
}
.mode-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 20px;
    padding: 1.8rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
}
.mode-card:hover {
    border-color: #7c3aed;
    transform: translateY(-4px);
    background: #13132a;
}
.mode-icon {
    font-size: 2.5rem;
    margin-bottom: 0.75rem;
}
.mode-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}
.mode-desc {
    font-size: 0.75rem;
    color: #64748b;
}

/* JD Match Card */
.jd-match-card {
    background: linear-gradient(135deg, #0f0f20 0%, #1a0f35 100%);
    border: 1px solid #2d2060;
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.jd-match-score {
    font-size: 4rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
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

/* Welcome Card - No empty box */
.welcome-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    max-width: 450px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analysis_submode" not in st.session_state:
    st.session_state.analysis_submode = None  # None, "cv_analysis", or "jd_match"
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
    
    # Extract keywords from JD
    jd_keywords = set()
    keyword_patterns = [
        r'(?:experience with|knowledge of|proficiency in|familiarity with)\s+([a-z][a-z\s]+?)(?=\.|,|\n)',
        r'(?:must have|required|essential|preferred)\s+([a-z][a-z\s]+?)(?=\.|,|\n)',
        r'\b(python|sql|tensorflow|pytorch|langchain|rag|llm|nlp|computer vision|docker|kubernetes|aws|gcp|azure|mlflow|scikit-learn|pandas|numpy|git|linux|keras|opencv|flask|fastapi|django)\b'
    ]
    
    for pattern in keyword_patterns:
        matches = re.findall(pattern, jd_lower, re.IGNORECASE)
        for match in matches:
            words = match.strip().split()[:3]
            jd_keywords.add(' '.join(words))
    
    # Add individual skills
    skill_list = ['python', 'sql', 'tensorflow', 'pytorch', 'langchain', 'rag', 'llm', 'nlp', 'docker', 'kubernetes', 'aws', 'gcp', 'azure']
    for skill in skill_list:
        if skill in jd_lower:
            jd_keywords.add(skill)
    
    # Calculate match
    found_keywords = []
    missing_keywords = []
    
    for kw in jd_keywords:
        if kw in cv_lower or kw.replace(" ", "") in cv_lower:
            found_keywords.append(kw)
        else:
            missing_keywords.append(kw)
    
    match_pct = int((len(found_keywords) / max(len(jd_keywords), 1)) * 100)
    
    # Find matching companies for this JD
    matching_companies = []
    try:
        roles = load_roles()
        for role in roles:
            role_jd = role.get("jd_text", "").lower()
            if any(kw in role_jd for kw in list(jd_keywords)[:10]):
                salary = role.get("salary", {})
                salary_str = ""
                if salary.get("junior"):
                    salary_str = f"৳{salary['junior']}"
                    if salary.get("mid"):
                        salary_str += f" - ৳{salary['mid']}"
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
    """Render navigation bar at top."""
    name = st.session_state.candidate_name or "Guest"
    first_name = name.split()[0] if name else "Guest"
    
    # Use HTML for fixed navbar
    st.markdown(f"""
    <div class="top-nav">
        <div class="nav-logo" onclick="location.reload()">📄 CV Analyzer</div>
        <div class="nav-links">
            <button class="nav-link" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'home'}}, '*')">🏠 Home</button>
            <button class="nav-link" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'analyze'}}, '*')">📊 Analysis</button>
            <button class="nav-link" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'chat'}}, '*')">💬 Chat</button>
            <button class="nav-link" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'career'}}, '*')">🎯 Career Rec</button>
            <span class="user-name" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'reset'}}, '*')">👋 {first_name}</span>
        </div>
    </div>
    <div class="main-content"></div>
    """, unsafe_allow_html=True)
    
    # Buttons for navigation (Streamlit way)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.analysis_submode = None
            st.rerun()
    
    with col2:
        if st.button("📊 Analysis", key="nav_analyze", use_container_width=True):
            if st.session_state.cv_text:
                st.session_state.page = "analyze"
                st.session_state.analysis_tab = "overview"
            else:
                st.warning("Please analyze your CV first on Home page")
            st.rerun()
    
    with col3:
        if st.button("💬 Chat", key="nav_chat", use_container_width=True):
            if st.session_state.agent:
                st.session_state.page = "chat"
            else:
                st.warning("Please analyze your CV first")
            st.rerun()
    
    with col4:
        if st.button("🎯 Career Rec", key="nav_career", use_container_width=True):
            if st.session_state.cv_text:
                st.session_state.page = "career"
            else:
                st.warning("Please analyze your CV first")
            st.rerun()
    
    with col5:
        if st.button(f"👋 {first_name}", key="nav_reset", use_container_width=True):
            st.session_state.name_entered = False
            st.session_state.cv_text = None
            st.session_state.analysis_raw = None
            st.session_state.retrieved = None
            st.session_state.agent = None
            st.session_state.messages = []
            st.rerun()
    
    st.markdown("<hr style='margin: 0.5rem 0 1rem 0; border-color: #1a1a30;'>", unsafe_allow_html=True)


def render_welcome_screen():
    """Clean welcome screen - no empty box"""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 3rem; font-weight: 800;'>
            CV <span style='background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Analyzer</span>
        </div>
        <p style='color: #64748b; margin: 0.5rem 0;'>AI-powered career matching for data & AI roles</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-family: Syne, sans-serif; font-size: 1.2rem; font-weight: 600; color: #cbd5e1; margin-bottom: 1rem;">👋 Welcome! What\'s your name?</div>', unsafe_allow_html=True)
        
        with st.form(key="name_entry_form"):
            name_val = st.text_input(
                "Name",
                placeholder="Talha Jobayer",
                label_visibility="collapsed",
                key="welcome_name_input"
            )
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("✨ Start Your Career Analysis →", use_container_width=True, type="primary")
            
            if submitted and name_val and name_val.strip():
                st.session_state.candidate_name = name_val.strip()
                st.session_state.name_entered = True
                st.rerun()
            elif submitted:
                st.error("Please enter your name to continue")
        
        st.markdown('</div>', unsafe_allow_html=True)


def render_home():
    """Home page with two options - CV Analysis or JD Match"""
    name = st.session_state.candidate_name
    first_name = name.split()[0] if name else "there"
    
    st.markdown(f"""
    <div style='text-align: center; padding: 1rem 0 1rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 1.8rem; font-weight: 800; color: #f1f5f9;'>
            Hello, {first_name}! 👋
        </div>
        <p style='color: #64748b; margin-top: 0.25rem;'>Ready to analyze your CV?</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Two options as clickable cards
    st.markdown('<div class="mode-grid">', unsafe_allow_html=True)
    
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
        if st.button("📄 Analyze My CV", key="btn_cv_analysis", use_container_width=True):
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
        if st.button("🎯 Match with JD", key="btn_jd_match", use_container_width=True):
            st.session_state.analysis_submode = "jd_match"
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show the selected mode UI
    if st.session_state.analysis_submode == "cv_analysis":
        render_cv_analysis_ui()
    elif st.session_state.analysis_submode == "jd_match":
        render_jd_match_ui()


def render_cv_analysis_ui():
    """CV Analysis UI - upload CV and analyze"""
    st.markdown("---")
    st.markdown("### 📄 Upload Your CV")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_cv = st.file_uploader("Choose PDF file", type=["pdf"], label_visibility="collapsed", key="cv_analysis_upload")
        if uploaded_cv:
            st.success(f"✅ {uploaded_cv.name}")
    
    with col2:
        st.markdown("""
        <div style='background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1rem;'>
            <div style='font-size: 0.75rem; color: #64748b;'>📊 What we analyze:</div>
            <div style='font-size: 0.7rem; color: #94a3b8;'>• Technical skills & tools<br>• Project experience<br>• Certifications<br>• AI/ML work experience</div>
        </div>
        """, unsafe_allow_html=True)
    
    if uploaded_cv:
        if st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
            process_cv_analysis(uploaded_cv, "")


def render_jd_match_ui():
    """JD Match UI - upload CV and paste JD"""
    st.markdown("---")
    st.markdown("### 🎯 Match Your CV with a Job Description")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("**📄 Your CV**")
        uploaded_cv = st.file_uploader("Upload CV (PDF)", type=["pdf"], label_visibility="collapsed", key="jd_match_cv")
        if uploaded_cv:
            st.success(f"✅ {uploaded_cv.name}")
    
    with col2:
        st.markdown("**📝 Job Description**")
        jd_input = st.text_area(
            "Paste JD here",
            height=200,
            placeholder="Paste the job description from LinkedIn, Indeed, etc...",
            label_visibility="collapsed",
            key="jd_match_text"
        )
    
    if uploaded_cv and jd_input:
        if st.button("🎯 Calculate Match Score", use_container_width=True, type="primary"):
            process_jd_match(uploaded_cv, jd_input)
    elif uploaded_cv and not jd_input:
        st.info("📝 Please paste a Job Description to continue")
    elif not uploaded_cv and jd_input:
        st.info("📄 Please upload your CV to continue")


def process_cv_analysis(uploaded_cv, jd_text):
    """Process CV analysis against knowledge base."""
    with st.spinner("📖 Reading CV..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_cv.read())
            tmp_path = tmp.name
        
        cv_text = extract_cv_text(tmp_path)
        os.unlink(tmp_path)
    
    st.session_state.cv_text = cv_text
    st.session_state.jd_text = ""
    st.session_state.messages = []
    
    with st.spinner("🔍 Matching with FAISS..."):
        st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
    
    with st.spinner("🏢 Finding matching companies..."):
        st.session_state.matched_companies = match_companies(cv_text, [])
    
    with st.spinner("🤖 Generating AI analysis..."):
        st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
        
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


def process_jd_match(uploaded_cv, jd_text):
    """Process CV vs JD matching."""
    with st.spinner("📖 Reading CV..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_cv.read())
            tmp_path = tmp.name
        
        cv_text = extract_cv_text(tmp_path)
        os.unlink(tmp_path)
    
    st.session_state.cv_text = cv_text
    st.session_state.jd_text = jd_text
    st.session_state.messages = []
    
    with st.spinner("🎯 Calculating JD match score..."):
        jd_match_result = calculate_jd_match_score(cv_text, jd_text)
        st.session_state.jd_match_result = jd_match_result
    
    with st.spinner("🏢 Finding matching companies..."):
        st.session_state.matched_companies = match_companies(cv_text, [])
    
    st.session_state.page = "jd_result"
    st.success("✅ JD Match complete!")
    st.rerun()


def render_jd_result():
    """Display JD matching results."""
    if not st.session_state.jd_match_result:
        st.info("No JD match result found. Please go back and try again.")
        if st.button("← Back to Home"):
            st.session_state.page = "home"
            st.session_state.analysis_submode = None
            st.rerun()
        return
    
    result = st.session_state.jd_match_result
    match_pct = result["match_pct"]
    
    if match_pct < 30:
        color = "#ef4444"
        status = "Low Match"
        advice = "Your CV needs significant updates to match this role."
    elif match_pct < 60:
        color = "#f59e0b"
        status = "Partial Match"
        advice = "You have some relevant skills. Focus on filling the gaps."
    elif match_pct < 80:
        color = "#10b981"
        status = "Good Match"
        advice = "You're a strong candidate for this role!"
    else:
        color = "#06b6d4"
        status = "Excellent Match!"
        advice = "You're highly qualified for this position. Apply now!"
    
    st.markdown(f"""
    <div class='jd-match-card'>
        <div style='font-size: 0.8rem; color: {color}; text-transform: uppercase; letter-spacing: 0.1em;'>JD Match Score</div>
        <div class='jd-match-score' style='color: {color};'>{match_pct}%</div>
        <div style='font-size: 1.2rem; font-weight: 600; color: #f1f5f9;'>{status}</div>
        <p style='color: #94a3b8; margin-top: 0.5rem;'>{advice}</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='card'>
            <div class='card-title'>📊 Match Details</div>
            <div>✅ Keywords matched: <strong style='color: #10b981;'>{result['found_count']}</strong> / {result['total_keywords']}</div>
            <div style='margin-top: 0.5rem;'>🎯 Match rate: <strong style='color: {color};'>{match_pct}%</strong></div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if result.get("matching_companies"):
            st.markdown("### 🏢 Similar Companies Hiring")
            for comp in result["matching_companies"][:3]:
                st.markdown(f"""
                <div class='company-card'>
                    <div style='font-weight: 700;'>{comp['name']}</div>
                    <div style='font-size: 0.75rem; color: #a855f7;'>{comp['role']}</div>
                    <div style='font-size: 0.7rem; color: #64748b;'>{comp['salary']} 📍 {comp['location']}</div>
                </div>
                """, unsafe_allow_html=True)
    
    if result.get("matched_keywords"):
        st.markdown("### ✅ Keywords Found in Your CV")
        matched_html = "".join(f"<span class='skill-chip add-chip'>{kw}</span>" for kw in result["matched_keywords"][:10])
        st.markdown(matched_html, unsafe_allow_html=True)
    
    if result.get("missing_keywords"):
        st.markdown("### ❌ Missing Keywords")
        missing_html = "".join(f"<span class='skill-chip gap-chip'>{kw}</span>" for kw in result["missing_keywords"][:10])
        st.markdown(missing_html, unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("← New JD Match", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.analysis_submode = "jd_match"
        st.rerun()


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
            if st.button(label, key=f"tab_{key}", use_container_width=True):
                st.session_state.analysis_tab = key
                st.rerun()


def render_overview_tab(retrieved, all_matches, top_match, parsed, readiness):
    """Overview tab - hero section + role breakdown."""
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
    
    if readiness:
        render_score_card(
            readiness.get("total_score", 0),
            readiness.get("level", "Not Ready"),
            readiness.get("recommendation", "")
        )
    
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
                        <div style='font-size: 0.7rem; color: #64748b;'>📍 {comp['location']}</div>
                    </div>
                    <div style='text-align: right;'>
                        <div style='font-size: 1.3rem; font-weight: 800; color: {match_color};'>{comp['match_score']}%</div>
                        <div style='font-size: 0.65rem; color: #64748b;'>Match</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if comp.get("skills"):
                skills_html = "".join(f"<span class='skill-chip'>{s[:20]}</span>" for s in comp["skills"][:4])
                st.markdown(f"<div style='margin-top: 0.5rem;'>{skills_html}</div>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if comp.get("salary_range") and comp["salary_range"]:
                st.markdown(f"""
                <div style='background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 12px; padding: 0.8rem; text-align: center; height: 100%;'>
                    <div style='font-size: 0.65rem; color: #fbbf24;'>💰 Salary</div>
                    <div style='font-size: 0.75rem; font-weight: 700; color: #fbbf24;'>{comp['salary_range']}</div>
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


def render_salary_tab(retrieved, all_matches):
    """Salary tab - salary insights and benchmarks."""
    st.markdown("### 💰 Salary Insights")
    
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
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <div style='font-size: 0.7rem; color: #64748b;'>👨‍💻 Mid-Level</div>
            <div style='font-size: 1.2rem; font-weight: 800; color: #10b981;'>40k - 70k</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='card' style='text-align: center;'>
            <div style='font-size: 0.7rem; color: #64748b;'>🚀 Senior</div>
            <div style='font-size: 1.2rem; font-weight: 800; color: #06b6d4;'>70k - 150k+</div>
        </div>
        """, unsafe_allow_html=True)


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
            st.success("✅ No major skill gaps!")
    
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
        ("🐍 Python", "https://www.python.org/"),
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
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - 📝 Update LinkedIn profile
        - 🔗 Build portfolio projects
        - 📚 Take relevant certifications
        """)
    with col2:
        st.markdown("""
        - 🤝 Network with industry professionals
        - 📊 Contribute to open source
        - 🎯 Apply to matching companies
        """)


def render_analysis():
    """Display analysis results with smart tabs."""
    if not st.session_state.cv_text:
        st.info("👈 Please analyze your CV first on the Home page")
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.session_state.analysis_submode = None
            st.rerun()
        return
    
    retrieved = st.session_state.retrieved or {}
    all_matches = retrieved.get("all_matches", [])
    top_match = retrieved.get("top_match", {})
    readiness = retrieved.get("readiness", {})
    parsed = parse_analysis(st.session_state.analysis_raw or "") if st.session_state.analysis_raw else {}
    
    render_analysis_tabs()
    
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


def render_career_rec():
    """Career Recommendations page."""
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <div style='font-size: 1.5rem; font-weight: 700; color: #f1f5f9;'>🎯 Career Recommendations</div>
        <p style='color: #64748b;'>Personalized advice based on your CV analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.cv_text:
        st.info("👈 Please analyze your CV first on the Home page")
        if st.button("Go to Home →"):
            st.session_state.page = "home"
            st.session_state.analysis_submode = None
            st.rerun()
        return
    
    retrieved = st.session_state.retrieved or {}
    readiness = retrieved.get("readiness", {})
    all_matches = retrieved.get("all_matches", [])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='card'>
            <div class='card-title'>📊 Your Readiness Score</div>
        """, unsafe_allow_html=True)
        
        score = readiness.get("total_score", 0)
        if score < 30:
            st.warning(f"🔴 **Score: {score}%** - Focus on foundational skills")
            st.markdown("""
            **Recommended next steps:**
            - Complete Python basics
            - Take ML fundamentals courses
            - Build 2-3 small projects
            """)
        elif score < 60:
            st.warning(f"🟡 **Score: {score}%** - Building momentum")
            st.markdown("""
            **Recommended next steps:**
            - Take advanced ML courses
            - Build portfolio projects
            - Get relevant certifications
            """)
        else:
            st.success(f"🟢 **Score: {score}%** - Ready for job search!")
            st.markdown("""
            **Recommended next steps:**
            - Update LinkedIn profile
            - Start applying to matching companies
            - Prepare for technical interviews
            """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if all_matches:
            st.markdown("""
            <div class='card'>
                <div class='card-title'>🎯 Best Matched Roles</div>
            """, unsafe_allow_html=True)
            for role in all_matches[:3]:
                st.markdown(f"""
                <div style='margin-bottom: 0.75rem;'>
                    <div style='font-weight: 600; color: #cbd5e1;'>{role.get('title', role.get('role'))}</div>
                    <div style='font-size: 0.8rem; color: #a855f7;'>{role.get('match_pct', 0)}% match</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    if st.button("📊 View Full Analysis →", use_container_width=True, type="primary"):
        st.session_state.page = "analyze"
        st.rerun()


def render_chat():
    """Chat interface."""
    if not st.session_state.agent:
        st.info("👈 Please analyze your CV first on the Home page")
        if st.button("Go to Home"):
            st.session_state.page = "home"
            st.session_state.analysis_submode = None
            st.rerun()
        return
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <div style='font-size: 1.2rem; font-weight: 600; color: #f1f5f9;'>💬 Ask Your Career Advisor</div>
        <p style='color: #64748b;'>Ask about skills, salaries, job search, or career path</p>
    </div>
    """, unsafe_allow_html=True)
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
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
    # Show welcome screen if name not entered
    if not st.session_state.name_entered:
        render_welcome_screen()
        return
    
    # Render navbar and page content
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
