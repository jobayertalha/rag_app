"""
app.py — AI Career Platform
Clean modular architecture with proper page routing and fixed navbar
"""

import streamlit as st
import tempfile
import os
import re

from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context, match_cv_with_jd, score_ai_ml_readiness
from quiz import QUESTIONS, calculate_interest_score

st.set_page_config(
    page_title="AI Career Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS — Fixed navbar, no duplicate, no lag
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Core resets */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
footer {visibility: hidden;}
.stApp { background: linear-gradient(135deg, #0a0a14 0%, #0f0f20 100%); min-height: 100vh; }
* { font-family: 'DM Sans', sans-serif; }

/* FIXED NAVBAR - Critical fix - NO DUPLICATE */
.navbar-fixed {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 99999;
    background: rgba(15, 15, 32, 0.98);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid #2d2d5a;
    padding: 0.75rem 2rem;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
}
.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1400px;
    margin: 0 auto;
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
    gap: 0.5rem;
}
.nav-btn {
    background: transparent;
    border: none;
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 500;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.2s;
}
.nav-btn:hover {
    background: rgba(168, 85, 247, 0.1);
    color: #a855f7;
}
.nav-btn-active {
    background: rgba(168, 85, 247, 0.15);
    color: #a855f7;
    border: 1px solid rgba(168, 85, 247, 0.3);
}
.user-info {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #cbd5e1;
    font-size: 0.85rem;
}
.user-name {
    background: rgba(168, 85, 247, 0.1);
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    border: 1px solid rgba(168, 85, 247, 0.2);
}

/* Main content padding to account for fixed navbar */
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
.feature-card {
    background: linear-gradient(135deg, #0f0f20 0%, #1a0f35 100%);
    border: 1px solid #2d2060;
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
    height: 100%;
}
.feature-card:hover {
    transform: translateY(-5px);
    border-color: #a855f7;
    box-shadow: 0 12px 40px rgba(168, 85, 247, 0.2);
}
.feature-icon {
    font-size: 3rem;
    margin-bottom: 1rem;
}
.feature-title {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}
.feature-desc {
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.5;
}
.hero-match {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
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
.result-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 1.5rem;
    margin-top: 1rem;
}

/* About page specific */
.about-container {
    max-width: 700px;
    margin: 0 auto;
}
.about-hero {
    text-align: center;
    margin-bottom: 2rem;
}
.about-hero-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
}
.about-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.about-subtitle {
    color: #64748b;
    font-size: 0.9rem;
}
.contact-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid #1e1e3a;
}
.contact-icon {
    font-size: 1.2rem;
    min-width: 40px;
    text-align: center;
}
.contact-label {
    font-weight: 600;
    color: #cbd5e1;
    min-width: 100px;
}
.contact-value {
    color: #94a3b8;
    flex: 1;
}
.contact-link {
    color: #a855f7;
    text-decoration: none;
}
.contact-link:hover {
    text-decoration: underline;
}
.tech-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}
.tech-pill {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    font-size: 0.72rem;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================
_defaults = {
    "page": "home",
    "candidate_name": "",
    "name_entered": False,
    "cv_text": None,
    "agent": None,
    "analysis_raw": None,
    "retrieved": None,
    "jd_match_result": None,
    "quiz_responses": {},
    "quiz_result": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# NAVIGATION - Smooth, no lag
# ============================================================
def nav_goto(page):
    """Smooth navigation without page rebuild flicker"""
    if st.session_state.page != page:
        st.session_state.page = page
        st.rerun()


def render_navbar():
    """SINGLE navbar - fixed, no duplicate"""
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    current_page = st.session_state.page
    
    # Use columns for click handling without JavaScript
    st.markdown(f"""
    <div class="navbar-fixed">
        <div class="nav-container">
            <div class="nav-logo" style="cursor:pointer;" onclick="window.location.reload()">🚀 AI Career Platform</div>
            <div class="nav-links">
                <span class="nav-btn {'nav-btn-active' if current_page == 'home' else ''}" id="nav-home">🏠 Home</span>
                <span class="nav-btn {'nav-btn-active' if current_page == 'analyze' else ''}" id="nav-analyze">📄 Analyze CV</span>
                <span class="nav-btn {'nav-btn-active' if current_page == 'jd_match' else ''}" id="nav-jd">🎯 JD Match</span>
                <span class="nav-btn {'nav-btn-active' if current_page == 'quiz' else ''}" id="nav-quiz">🧠 Quiz</span>
                <span class="nav-btn {'nav-btn-active' if current_page == 'about' else ''}" id="nav-about">ℹ️ About</span>
            </div>
            <div class="user-info">
                <span class="user-name">👤 {first}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Hidden buttons for navigation (invisible, only for functionality)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if st.button("", key="nav_home_hidden", help="Home"):
            nav_goto("home")
    with col2:
        if st.button("", key="nav_analyze_hidden", help="Analyze CV"):
            nav_goto("analyze")
    with col3:
        if st.button("", key="nav_jd_hidden", help="JD Match"):
            nav_goto("jd_match")
    with col4:
        if st.button("", key="nav_quiz_hidden", help="Quiz"):
            nav_goto("quiz")
    with col5:
        if st.button("", key="nav_about_hidden", help="About"):
            nav_goto("about")
    
    # Hide the column buttons completely
    st.markdown("""
    <style>
    div[data-testid="column"]:has(button[key*="hidden"]) {
        display: none;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================
# WELCOME SCREEN
# ============================================================
def render_welcome():
    st.markdown("""
    <div style="max-width: 500px; margin: 80px auto; text-align: center;">
        <div style="font-family: 'Syne', sans-serif; font-size: 3rem; font-weight: 800; margin-bottom: 1rem;">
            AI <span style="background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Career</span> Platform
        </div>
        <p style="color: #64748b; margin-bottom: 2rem;">Your AI-powered career companion for data science & AI/ML roles</p>
        <div class="card">
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 1rem;">👋 Welcome! What's your name?</div>
    """, unsafe_allow_html=True)
    
    name = st.text_input("Name", placeholder="e.g. Talha Jobayer", label_visibility="collapsed")
    if st.button("✨ Get Started →", type="primary", use_container_width=True):
        if name and name.strip():
            st.session_state.candidate_name = name.strip()
            st.session_state.name_entered = True
            st.rerun()
        else:
            st.error("Please enter your name")
    
    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# HOME PAGE - 3 cards only, no expansion
# ============================================================
def render_home():
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    
    st.markdown(f"""
    <div class="main-content">
        <h2 style="text-align:center; margin-bottom: 0.5rem;">Hello, {first}! 👋</h2>
        <p style="text-align:center; color:#64748b; margin-bottom: 3rem;">What would you like to do today?</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Analyze My CV</div>
            <div class="feature-desc">Upload your CV and get matched with the best AI/ML roles from our knowledge base.</div>
            <div class="feature-desc" style="color:#a855f7; margin-top: 1rem;">✨ Role Match · Skills · Salary · Career Path</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📄 Analyze CV", key="home_cv", use_container_width=True, type="primary"):
            nav_goto("analyze")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Match with JD</div>
            <div class="feature-desc">Paste a job description and see exactly how well your CV aligns.</div>
            <div class="feature-desc" style="color:#10b981; margin-top: 1rem;">✨ Match % · Missing Skills · Target Companies</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 JD Match", key="home_jd", use_container_width=True, type="primary"):
            nav_goto("jd_match")
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Interest Quiz</div>
            <div class="feature-desc">Take a quiz to discover which AI/ML roles match your interests and thinking style.</div>
            <div class="feature-desc" style="color:#f59e0b; margin-top: 1rem;">✨ Interest Score · Role Suggestions · Career Fit</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧠 Take Quiz", key="home_quiz", use_container_width=True, type="primary"):
            nav_goto("quiz")


# ============================================================
# ANALYZE PAGE
# ============================================================
def render_analyze():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin-bottom: 1rem;'>📄 CV Analysis</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b; margin-bottom: 2rem;'>Upload your CV to get personalized career recommendations</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"], label_visibility="collapsed")
        if uploaded and st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing your CV..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    cv_text = extract_cv_text(tmp.name)
                    os.unlink(tmp.name)
                
                st.session_state.cv_text = cv_text
                st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
                st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
                st.session_state.analysis_raw = run_agent(
                    st.session_state.agent,
                    "Analyse this CV. Follow tags: TOP_ROLE, MATCH_PCT, WHY_RIGHT, SKILL_GAPS, RESUME_ADD, CAREER_PATH"
                )
            st.rerun()
    
    if st.session_state.retrieved:
        render_analysis_results()
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_results():
    retrieved = st.session_state.retrieved
    top_match = retrieved.get("top_match", {})
    readiness = retrieved.get("readiness", {})
    
    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"<div class='hero-match'>{top_match.get('match_pct', 0)}% Match</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{top_match.get('title', top_match.get('role', 'AI Professional'))}</h3>", unsafe_allow_html=True)
    
    score = readiness.get("total_score", 0)
    if score < 30:
        st.warning(f"🔴 Readiness Score: {score}% — Focus on building fundamentals")
    elif score < 60:
        st.warning(f"🟡 Readiness Score: {score}% — Keep building your skills")
    else:
        st.success(f"🟢 Readiness Score: {score}% — You're ready to apply!")
    
    col1, col2 = st.columns(2)
    with col1:
        gaps = retrieved.get("skill_gaps", [])
        if gaps:
            st.markdown("**❌ Skill Gaps**")
            st.markdown(" ".join(f"<span class='skill-chip gap-chip'>{g}</span>" for g in gaps[:6]), unsafe_allow_html=True)
    
    with col2:
        recs = retrieved.get("resume_skills", [])
        if recs:
            st.markdown("**➕ Recommended Additions**")
            st.markdown(" ".join(f"<span class='skill-chip'>+ {sk}</span>" for sk in recs[:6]), unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Career Path")
    for r in retrieved.get("all_matches", [])[:3]:
        st.markdown(f"**{r.get('title', r.get('role', 'Role'))}** — {r.get('company', 'Various')} ({r.get('match_pct', 0)}% match)")
        if r.get("salary_min"):
            st.caption(f"💰 ৳{r['salary_min']:,} – ৳{r['salary_max']:,}/month")
        st.markdown("---")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("← Back to Home", use_container_width=True):
        nav_goto("home")


# ============================================================
# JD MATCH PAGE
# ============================================================
def render_jd_match():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin-bottom: 1rem;'>🎯 Job Description Matching</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b; margin-bottom: 2rem;'>See how well your CV matches a specific job description</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader("Your CV (PDF)", type=["pdf"], key="jd_cv_upload")
    
    with col2:
        jd_text = st.text_area("Job Description", height=200, placeholder="Paste the full job description here...")
    
    if uploaded and jd_text and st.button("🎯 Calculate Match", type="primary", use_container_width=True):
        with st.spinner("Calculating match..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded.read())
                cv_text = extract_cv_text(tmp.name)
                os.unlink(tmp.name)
            
            st.session_state.cv_text = cv_text
            st.session_state.jd_match_result = match_cv_with_jd(cv_text, jd_text)
        st.rerun()
    
    if st.session_state.jd_match_result:
        render_jd_match_results()
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_jd_match_results():
    result = st.session_state.jd_match_result
    pct = result.get("match_pct", 0)
    
    if pct < 30:
        color, status = "#ef4444", "Low Match"
    elif pct < 60:
        color, status = "#f59e0b", "Partial Match"
    elif pct < 80:
        color, status = "#10b981", "Good Match"
    else:
        color, status = "#06b6d4", "Excellent Match!"
    
    st.markdown(f"""
    <div class='result-card' style='text-align:center;'>
        <div style='color:{color}; font-size:0.9rem; font-weight:600;'>MATCH SCORE</div>
        <div style='font-size:3rem; font-weight:800; color:{color};'>{pct}%</div>
        <div style='font-size:1.1rem; font-weight:600;'>{status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    if result.get("similar_roles"):
        st.markdown("### 📊 Similar Roles from Database")
        for role in result["similar_roles"][:3]:
            st.markdown(f"**{role.get('title', role.get('role', 'Role'))}** at {role.get('company', 'Various')}")
    
    if st.button("← Back to Home", use_container_width=True):
        st.session_state.jd_match_result = None
        nav_goto("home")


# ============================================================
# QUIZ PAGE
# ============================================================
def render_quiz():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin-bottom: 0.5rem;'>🧠 Career Interest Quiz</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#64748b; margin-bottom: 2rem;'>Discover which AI/ML roles match your thinking style and interests</p>", unsafe_allow_html=True)
    
    if st.session_state.quiz_result:
        render_quiz_results()
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    if not st.session_state.quiz_responses:
        st.markdown("""
        <div class='card' style='text-align:center; max-width: 500px; margin: 0 auto;'>
            <div style='font-size:2rem; margin-bottom: 1rem;'>📋</div>
            <div style='font-size:1.1rem; font-weight:600; margin-bottom: 0.5rem;'>Ready to discover your career fit?</div>
            <div style='color:#64748b; font-size:0.9rem;'>Answer 10 questions about your preferences and thinking style.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
            st.session_state.quiz_responses = {q["id"]: None for q in QUESTIONS}
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    with st.form("quiz_form"):
        for q in QUESTIONS:
            qid = q["id"]
            st.markdown(f"""
            <div class='card' style='margin-bottom: 1rem;'>
                <div style='font-weight:600; margin-bottom: 0.5rem;'>{q["question"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
            response = st.radio(
                f"q{qid}",
                options=q["options"],
                key=f"quiz_{qid}",
                label_visibility="collapsed",
                index=0 if st.session_state.quiz_responses.get(qid) is None else st.session_state.quiz_responses[qid]
            )
            st.session_state.quiz_responses[qid] = q["options"].index(response)
        
        if st.form_submit_button("📊 Get Results", type="primary", use_container_width=True):
            result = calculate_interest_score(st.session_state.quiz_responses)
            st.session_state.quiz_result = result
            st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)


def render_quiz_results():
    result = st.session_state.quiz_result
    pct = result["pct"]
    level = result["level"]
    color = result["color"]
    
    st.markdown(f"""
    <div class='result-card' style='text-align:center;'>
        <div style='font-size:3rem; font-weight:800; color:{color};'>{pct}%</div>
        <div style='font-size:1.3rem; font-weight:700; margin: 0.5rem 0;'>Interest Level: {level}</div>
        <div style='color:#94a3b8;'>{result["message"]}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Recommended Roles")
        for role in result["recommended_roles"]:
            st.markdown(f"- {role}")
    
    with col2:
        st.markdown("### 💡 Why This Fit?")
        st.markdown(result["explanation"])
    
    if st.button("🔄 Take Quiz Again", use_container_width=True):
        st.session_state.quiz_responses = {}
        st.session_state.quiz_result = None
        st.rerun()
    
    if st.button("← Back to Home", use_container_width=True):
        st.session_state.quiz_responses = {}
        st.session_state.quiz_result = None
        nav_goto("home")


# ============================================================
# ABOUT PAGE - Clean, no raw HTML, clickable contact
# ============================================================
def render_about():
    st.markdown("<div class='main-content'>", unsafe_allow_html=True)
    st.markdown("<div class='about-container'>", unsafe_allow_html=True)
    
    # Hero Section
    st.markdown("""
    <div class="about-hero">
        <div class="about-hero-icon">🚀</div>
        <div class="about-title">AI Career Platform</div>
        <div class="about-subtitle">AI-powered career matching for data science & AI/ML roles</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main Card
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    
    # Developer Info
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Talha Jobayer Zihan**  \n*Researcher & AI/ML Engineer*")
    st.markdown("---")
    
    # Contact Section - Structured rows with clickable links
    st.markdown("### 📞 Contact Information")
    
    # Department
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("🏛️")
    with col2:
        st.markdown("**Department of Computer Science & Engineering, RUET**")
    
    # Phone - Clickable
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("📞")
    with col2:
        st.markdown('<a href="tel:01721577792" style="color:#a855f7; text-decoration:none;">01721577792</a>', unsafe_allow_html=True)
    
    # Email - Clickable
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("✉️")
    with col2:
        st.markdown('<a href="mailto:jobayertalha2020@gmail.com" style="color:#a855f7; text-decoration:none;">jobayertalha2020@gmail.com</a>', unsafe_allow_html=True)
    
    # LinkedIn
    col1, col2 = st.columns([1, 5])
    with col1:
        st.markdown("🔗")
    with col2:
        st.markdown('<a href="https://linkedin.com" target="_blank" style="color:#a855f7; text-decoration:none;">LinkedIn Profile</a>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tech Stack
    st.markdown("### 🛠️ Technology Stack")
    tech_stack = ["Streamlit", "LangChain", "Groq LLaMA-3.3-70b", "FAISS", "HuggingFace", "Python"]
    cols = st.columns(len(tech_stack))
    for i, tech in enumerate(tech_stack):
        with cols[i]:
            st.markdown(f"<div class='tech-pill' style='text-align:center;'>{tech}</div>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)  # Close card
    
    # Back button
    if st.button("← Back to Home", use_container_width=True):
        nav_goto("home")
    
    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# MAIN ROUTER
# ============================================================
def main():
    # Name entry required first
    if not st.session_state.name_entered:
        render_welcome()
        return
    
    # Show SINGLE navbar (fixed)
    render_navbar()
    
    # Route to correct page
    page = st.session_state.page
    
    if page == "home":
        render_home()
    elif page == "analyze":
        render_analyze()
    elif page == "jd_match":
        render_jd_match()
    elif page == "quiz":
        render_quiz()
    elif page == "about":
        render_about()
    else:
        render_home()


if __name__ == "__main__":
    main()
