"""
app.py — AI Career Platform
Professional Production-Ready Design - Clean & Modern
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
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PROFESSIONAL CSS - Clean Modern Design
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Global Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.stApp {
    background: #0a0e1a;
    min-height: 100vh;
}

/* Hide Streamlit Default Elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
footer {visibility: hidden;}
[data-testid="stHeader"] {display: none;}
[data-testid="stToolbar"] {display: none;}

/* Make Sidebar Fixed - Never Hidden */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0e1a 100%);
    border-right: 1px solid #1f2937;
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    height: 100vh !important;
    z-index: 100 !important;
    overflow-y: auto !important;
}

/* Main content margin to accommodate fixed sidebar */
.main-content {
    margin-left: 0;
}

/* Sidebar Brand */
.sidebar-brand {
    text-align: center;
    padding: 2rem 1rem;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 1.5rem;
}

.sidebar-brand-icon {
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.sidebar-brand-text {
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}

/* User Profile Button */
button[key="user_menu_btn"] {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 12px !important;
    padding: 0.75rem !important;
    color: #e5e7eb !important;
    font-weight: 600 !important;
    text-align: left !important;
    transition: all 0.3s ease !important;
}

button[key="user_menu_btn"]:hover {
    background: #374151 !important;
    border-color: #4b5563 !important;
}

/* Sign Out Button */
button[key="signout_option"] {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    color: #f87171 !important;
}

button[key="signout_option"]:hover {
    background: #374151 !important;
    color: #ef4444 !important;
}

/* Navigation Buttons */
button[key^="sidebar_"] {
    transition: all 0.3s ease !important;
    border-radius: 10px !important;
    margin-bottom: 0.5rem !important;
    font-weight: 500 !important;
}

button[key^="sidebar_"]:hover {
    transform: translateX(5px);
}

/* Main Content Area */
.main-header {
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 2px solid #1f2937;
}

.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.main-header p {
    color: #9ca3af;
    font-size: 0.95rem;
}

/* Feature Cards */
.feature-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 2rem;
    transition: all 0.3s ease;
    margin-bottom: 1rem;
}

.feature-card:hover {
    transform: translateY(-4px);
    border-color: #3b82f6;
    box-shadow: 0 20px 40px rgba(59, 130, 246, 0.1);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.75rem;
}

.feature-desc {
    color: #9ca3af;
    font-size: 0.85rem;
    line-height: 1.5;
}

.feature-tags {
    margin-top: 1rem;
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
}

.feature-tag {
    background: #1f2937;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.7rem;
    color: #60a5fa;
}

/* Result Cards */
.result-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.match-score {
    text-align: center;
    padding: 1.5rem;
}

.match-percentage {
    font-size: 3rem;
    font-weight: 800;
    color: #3b82f6;
}

/* Skill Chips */
.skill-chip {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.75rem;
    color: #9ca3af;
    margin: 0.25rem;
}

.gap-chip {
    background: #1f2937;
    border-color: #7f1d1d;
    color: #fca5a5;
}

/* Form Elements */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #111827 !important;
    border: 1px solid #1f2937 !important;
    border-radius: 12px !important;
    color: #ffffff !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

.stButton > button {
    background: #3b82f6 !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    background: #2563eb !important;
    transform: translateY(-2px);
}

/* Contact & About Pages */
.contact-card, .about-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 2rem;
}

.contact-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 0;
    border-bottom: 1px solid #1f2937;
}

.contact-item:last-child {
    border-bottom: none;
}

.contact-icon {
    font-size: 1.3rem;
    min-width: 45px;
    color: #60a5fa;
}

.contact-label {
    font-weight: 600;
    color: #e5e7eb;
    min-width: 100px;
}

.contact-value {
    color: #9ca3af;
}

.contact-link {
    color: #60a5fa;
    text-decoration: none;
}

.contact-link:hover {
    color: #93c5fd;
    text-decoration: underline;
}

.social-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1.5rem;
}

.social-card {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    transition: all 0.3s ease;
    text-decoration: none;
}

.social-card:hover {
    background: #374151;
    transform: translateY(-3px);
}

.social-icon {
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}

.social-name {
    color: #e5e7eb;
    font-size: 0.8rem;
    font-weight: 500;
}

.interest-tag {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 0.3rem 1rem;
    font-size: 0.75rem;
    color: #60a5fa;
    margin: 0.25rem;
}

.tech-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-top: 1rem;
}

.tech-pill {
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    font-size: 0.7rem;
    color: #9ca3af;
}

/* Profile Header */
.profile-header {
    text-align: center;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1f2937;
}

.profile-icon {
    font-size: 3.5rem;
    margin-bottom: 0.5rem;
}

.profile-name {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 0.25rem;
}

.profile-title {
    color: #60a5fa;
    font-size: 0.9rem;
}

/* Section Headers */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1f2937;
}

/* Welcome Screen - FIXED: Smaller name entry block */
.welcome-container {
    max-width: 450px;
    margin: 80px auto;
    text-align: center;
}

.welcome-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
    color: #ffffff;
}

.welcome-gradient {
    color: #3b82f6;
}

.welcome-subtitle {
    color: #9ca3af;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
}

.welcome-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 20px;
    padding: 1.5rem;
}

/* Quiz Styles */
.quiz-question {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.quiz-question-text {
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 1rem;
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
    "show_user_menu": False,
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
# NAVIGATION
# ============================================================
def nav_goto(page):
    if st.session_state.page != page:
        st.session_state.page = page
        st.rerun()


def sign_out():
    st.session_state.candidate_name = ""
    st.session_state.name_entered = False
    st.session_state.page = "home"
    st.session_state.show_user_menu = False
    st.session_state.cv_text = None
    st.session_state.agent = None
    st.session_state.analysis_raw = None
    st.session_state.retrieved = None
    st.session_state.jd_match_result = None
    st.session_state.quiz_responses = {}
    st.session_state.quiz_result = None
    st.rerun()


def render_sidebar():
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    current_page = st.session_state.page
    
    st.sidebar.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-icon">🎯</div>
        <div class="sidebar-brand-text">AI Career Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.sidebar.columns([1, 4])
    with col1:
        st.markdown("👤")
    with col2:
        if st.button(f"{first} ▼", key="user_menu_btn", use_container_width=True):
            st.session_state.show_user_menu = not st.session_state.show_user_menu
            st.rerun()
    
    if st.session_state.show_user_menu:
        col1, col2 = st.sidebar.columns([1, 4])
        with col1:
            st.markdown("⏻")
        with col2:
            if st.button("Sign Out", key="signout_option", use_container_width=True):
                sign_out()
    
    st.sidebar.markdown("---")
    
    nav_items = [
        ("🏠 Home", "home"),
        ("📄 Analyze CV", "analyze"),
        ("🎯 JD Match", "jd_match"),
        ("🧠 Quiz", "quiz"),
        ("ℹ️ About", "about"),
        ("📞 Contact", "contact")
    ]
    
    for label, page_key in nav_items:
        is_active = (current_page == page_key)
        button_type = "primary" if is_active else "secondary"
        
        if st.sidebar.button(label, key=f"sidebar_{page_key}", use_container_width=True, type=button_type):
            st.session_state.show_user_menu = False
            nav_goto(page_key)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2025 AI Career Platform")


# ============================================================
# WELCOME SCREEN - FIXED: Smaller and centered
# ============================================================
def render_welcome():
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-title">
            AI <span class="welcome-gradient">Career</span> Platform
        </div>
        <p class="welcome-subtitle">Your AI-powered career companion for data science & AI/ML roles</p>
        <div class="welcome-card">
            <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 1rem; color: #e5e7eb;">👋 Welcome! What's your name?</div>
    """, unsafe_allow_html=True)
    
    # Smaller name input with form
    with st.form(key="welcome_form"):
        name = st.text_input("Name", placeholder="e.g. Talha Jobayer", label_visibility="collapsed")
        submit_button = st.form_submit_button("✨ Get Started →", use_container_width=True, type="primary")
        
        if submit_button:
            if name and name.strip():
                st.session_state.candidate_name = name.strip()
                st.session_state.name_entered = True
                st.session_state.show_user_menu = False
                st.rerun()
            else:
                st.error("Please enter your name")
    
    st.markdown("</div></div>", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================
def render_home():
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    
    st.markdown(f"""
    <div class="main-header">
        <h1>Hello, {first}! 👋</h1>
        <p>Welcome to your AI-powered career companion. Let's find your perfect role in AI/ML.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📄</div>
            <div class="feature-title">Analyze My CV</div>
            <div class="feature-desc">Upload your CV and get matched with the best AI/ML roles from our knowledge base.</div>
            <div class="feature-tags">
                <span class="feature-tag">Role Match</span>
                <span class="feature-tag">Skills</span>
                <span class="feature-tag">Salary</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📄 Analyze CV", key="home_cv", use_container_width=True):
            nav_goto("analyze")
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Match with JD</div>
            <div class="feature-desc">Paste a job description and see exactly how well your CV aligns.</div>
            <div class="feature-tags">
                <span class="feature-tag">Match %</span>
                <span class="feature-tag">Missing Skills</span>
                <span class="feature-tag">Target Companies</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 JD Match", key="home_jd", use_container_width=True):
            nav_goto("jd_match")
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🧠</div>
            <div class="feature-title">Interest Quiz</div>
            <div class="feature-desc">Take a quiz to discover which AI/ML roles match your interests.</div>
            <div class="feature-tags">
                <span class="feature-tag">Interest Score</span>
                <span class="feature-tag">Role Suggestions</span>
                <span class="feature-tag">Career Fit</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧠 Take Quiz", key="home_quiz", use_container_width=True):
            nav_goto("quiz")


# ============================================================
# ANALYZE PAGE
# ============================================================
def render_analyze():
    st.markdown("""
    <div class="main-header">
        <h1>📄 CV Analysis</h1>
        <p>Upload your CV to get personalized career recommendations based on real job market data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"], label_visibility="collapsed")
        if uploaded and st.button("🚀 Start Analysis", use_container_width=True):
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


def render_analysis_results():
    retrieved = st.session_state.retrieved
    top_match = retrieved.get("top_match", {})
    readiness = retrieved.get("readiness", {})
    
    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div class="match-score">
            <div style="font-size: 0.85rem; color: #9ca3af; margin-bottom: 0.5rem;">MATCH SCORE</div>
            <div class="match-percentage">{top_match.get('match_pct', 0)}%</div>
            <h3 style="color: #ffffff; margin-top: 0.5rem; font-size: 1.1rem;">{top_match.get('title', top_match.get('role', 'AI Professional'))}</h3>
        </div>
        """, unsafe_allow_html=True)
    
    score = readiness.get("total_score", 0)
    if score < 30:
        st.warning("🔴 **Readiness Score:** {}% — Focus on building fundamentals".format(score))
    elif score < 60:
        st.warning("🟡 **Readiness Score:** {}% — Keep building your skills".format(score))
    else:
        st.success("🟢 **Readiness Score:** {}% — You're ready to apply!".format(score))
    
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
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <strong>{r.get('title', r.get('role', 'Role'))}</strong> — {r.get('company', 'Various')} 
            <span style="color: #60a5fa;">({r.get('match_pct', 0)}% match)</span>
        </div>
        """, unsafe_allow_html=True)
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
    st.markdown("""
    <div class="main-header">
        <h1>🎯 Job Description Matching</h1>
        <p>See how well your CV matches a specific job description.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded = st.file_uploader("Your CV (PDF)", type=["pdf"], key="jd_cv_upload")
    
    with col2:
        jd_text = st.text_area("Job Description", height=200, placeholder="Paste the full job description here...")
    
    if uploaded and jd_text and st.button("🎯 Calculate Match", use_container_width=True):
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
        color, status = "#3b82f6", "Excellent Match!"
    
    st.markdown(f"""
    <div class="result-card">
        <div style="text-align:center;">
            <div style="font-size: 0.85rem; color: #9ca3af;">MATCH SCORE</div>
            <div style="font-size: 3rem; font-weight: 800; color: {color};">{pct}%</div>
            <div style="font-size: 1rem; font-weight: 600; color: {color};">{status}</div>
        </div>
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
    st.markdown("""
    <div class="main-header">
        <h1>🧠 Career Interest Quiz</h1>
        <p>Discover which AI/ML roles match your thinking style and interests.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.quiz_result:
        render_quiz_results()
        return
    
    if not st.session_state.quiz_responses:
        st.markdown("""
        <div style="text-align:center; max-width: 450px; margin: 2rem auto;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">📋</div>
            <div style="font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem; color: #ffffff;">Ready to discover your career fit?</div>
            <div style="color: #9ca3af; font-size: 0.85rem;">Answer 10 questions about your preferences and thinking style.</div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Quiz", use_container_width=True):
            st.session_state.quiz_responses = {q["id"]: None for q in QUESTIONS}
            st.rerun()
        return
    
    with st.form("quiz_form"):
        for q in QUESTIONS:
            qid = q["id"]
            st.markdown(f"""
            <div class="quiz-question">
                <div class="quiz-question-text">{q["question"]}</div>
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
        
        if st.form_submit_button("📊 Get Results", use_container_width=True):
            result = calculate_interest_score(st.session_state.quiz_responses)
            st.session_state.quiz_result = result
            st.rerun()


def render_quiz_results():
    result = st.session_state.quiz_result
    pct = result["pct"]
    level = result["level"]
    color = "#3b82f6" if level == "HIGH" else ("#f59e0b" if level == "MEDIUM" else "#ef4444")
    
    st.markdown(f"""
    <div class="result-card">
        <div style="text-align:center;">
            <div style="font-size: 3rem; font-weight: 800; color: {color};">{pct}%</div>
            <div style="font-size: 1.1rem; font-weight: 600; margin: 0.5rem 0; color: #ffffff;">Interest Level: {level}</div>
            <div style="color: #9ca3af;">{result["message"]}</div>
        </div>
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
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Take Quiz Again", use_container_width=True):
            st.session_state.quiz_responses = {}
            st.session_state.quiz_result = None
            st.rerun()
    with col2:
        if st.button("← Back to Home", use_container_width=True):
            st.session_state.quiz_responses = {}
            st.session_state.quiz_result = None
            nav_goto("home")


# ============================================================
# ABOUT PAGE
# ============================================================
def render_about():
    st.markdown("""
    <div class="main-header">
        <h1>ℹ️ About</h1>
        <p>Learn more about the developer and this platform.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-card">
        <div class="profile-header">
            <div class="profile-icon">👨‍💻</div>
            <div class="profile-name">Talha Jobayer Zihan</div>
            <div class="profile-title">Researcher & AI/ML Engineer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🔬 Research Interests</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <span class="interest-tag">Natural Language Processing (NLP)</span>
        <span class="interest-tag">Computer Vision</span>
        <span class="interest-tag">Cyber Security</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎓 Academic Affiliation</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #e5e7eb; margin-bottom: 1.5rem;">Department of Computer Science & Engineering<br>Rajshahi University of Engineering & Technology (RUET)</p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🚀 About This Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #9ca3af; line-height: 1.6; margin-bottom: 1.5rem;">AI Career Platform is an intelligent career matching system designed to help job seekers in Bangladesh find the best AI/ML roles based on their CV content, skills, and career preferences.</p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">✨ Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <ul style="color: #9ca3af; line-height: 1.8; margin-bottom: 1.5rem;">
        <li>📄 AI-powered CV analysis and role matching</li>
        <li>🎯 Job Description matching with real-time skill gap analysis</li>
        <li>🧠 Career interest quiz to discover your ideal role</li>
        <li>💰 Salary insights and market demand data</li>
        <li>📈 Personalized career path recommendations</li>
    </ul>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🛠️ Technology Stack</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="tech-stack">
        <span class="tech-pill">Streamlit</span>
        <span class="tech-pill">LangChain</span>
        <span class="tech-pill">Groq LLaMA-3.3-70b</span>
        <span class="tech-pill">FAISS</span>
        <span class="tech-pill">HuggingFace</span>
        <span class="tech-pill">Python</span>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home", use_container_width=True):
        nav_goto("home")


# ============================================================
# CONTACT PAGE
# ============================================================
def render_contact():
    st.markdown("""
    <div class="main-header">
        <h1>📞 Contact Us</h1>
        <p>Get in touch with the developer.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="contact-card">
        <div class="contact-item">
            <div class="contact-icon">👨‍💻</div>
            <div class="contact-label">Developer</div>
            <div class="contact-value">Talha Jobayer Zihan</div>
        </div>
        <div class="contact-item">
            <div class="contact-icon">📱</div>
            <div class="contact-label">Phone</div>
            <div class="contact-value"><a href="tel:01721577792" class="contact-link">+880 1721 577792</a></div>
        </div>
        <div class="contact-item">
            <div class="contact-icon">✉️</div>
            <div class="contact-label">Email</div>
            <div class="contact-value"><a href="mailto:jobayertalha2020@gmail.com" class="contact-link">jobayertalha2020@gmail.com</a></div>
        </div>
        <div class="contact-item">
            <div class="contact-icon">💻</div>
            <div class="contact-label">GitHub</div>
            <div class="contact-value"><a href="https://github.com/jobayertalha" target="_blank" class="contact-link">github.com/jobayertalha</a></div>
        </div>
        <div class="contact-item">
            <div class="contact-icon">🔗</div>
            <div class="contact-label">LinkedIn</div>
            <div class="contact-value"><a href="https://www.linkedin.com/in/talha-jobayer-696a74237/" target="_blank" class="contact-link">linkedin.com/in/talha-jobayer</a></div>
        </div>
        <div class="contact-item">
            <div class="contact-icon">🌐</div>
            <div class="contact-label">Portfolio</div>
            <div class="contact-value"><a href="https://v0-personal-portfolio-site-tau.vercel.app/" target="_blank" class="contact-link">Personal Portfolio</a></div>
        </div>
    </div>
    
    <div class="section-header" style="margin-top: 2rem; text-align: center;">Connect With Me</div>
    <div class="social-grid">
        <a href="https://github.com/jobayertalha" target="_blank" class="social-card">
            <div class="social-icon">💻</div>
            <div class="social-name">GitHub</div>
        </a>
        <a href="https://www.linkedin.com/in/talha-jobayer-696a74237/" target="_blank" class="social-card">
            <div class="social-icon">🔗</div>
            <div class="social-name">LinkedIn</div>
        </a>
        <a href="https://v0-personal-portfolio-site-tau.vercel.app/" target="_blank" class="social-card">
            <div class="social-icon">🌐</div>
            <div class="social-name">Portfolio</div>
        </a>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("← Back to Home", use_container_width=True):
        nav_goto("home")


# ============================================================
# MAIN ROUTER
# ============================================================
def main():
    if not st.session_state.name_entered:
        render_welcome()
        return
    
    render_sidebar()
    
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
    elif page == "contact":
        render_contact()
    else:
        render_home()


if __name__ == "__main__":
    main()
