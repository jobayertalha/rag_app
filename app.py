"""
app.py — AI Career Platform
FIXED: Top navbar always visible, working navigation
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
    initial_sidebar_state="collapsed"  # Collapse sidebar to hide it
)

# ============================================================
# PROFESSIONAL CSS - With Top Navbar
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
[data-testid="stSidebar"] {display: none;} /* Hide sidebar completely */

/* Top Navbar - Always Visible */
.top-navbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 999999;
    background: linear-gradient(135deg, #0d1117 0%, #0a0e1a 100%);
    border-bottom: 1px solid #1f2937;
    padding: 0.75rem 2rem;
    backdrop-filter: blur(10px);
}

.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1400px;
    margin: 0 auto;
}

.nav-logo {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Inter', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.nav-logo-icon {
    font-size: 1.5rem;
}

.nav-menu {
    display: flex;
    gap: 0.25rem;
    background: rgba(255,255,255,0.03);
    padding: 0.25rem;
    border-radius: 40px;
}

.nav-item {
    padding: 0.5rem 1rem;
    border-radius: 30px;
    font-size: 0.85rem;
    font-weight: 500;
    color: #9ca3af;
    cursor: pointer;
    transition: all 0.3s ease;
    background: transparent;
    border: none;
    font-family: 'Inter', sans-serif;
}

.nav-item:hover {
    background: rgba(59, 130, 246, 0.15);
    color: #60a5fa;
}

.nav-item-active {
    background: rgba(59, 130, 246, 0.2);
    color: #60a5fa;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.nav-user {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #1f2937;
    padding: 0.4rem 1rem;
    border-radius: 30px;
    cursor: pointer;
    transition: all 0.3s ease;
    position: relative;
}

.nav-user:hover {
    background: #374151;
}

.user-name {
    color: #e5e7eb;
    font-size: 0.85rem;
    font-weight: 500;
}

.user-arrow {
    color: #9ca3af;
    font-size: 0.7rem;
}

/* User Dropdown Menu */
.user-dropdown {
    position: absolute;
    top: 100%;
    right: 0;
    margin-top: 0.5rem;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 0.5rem;
    min-width: 140px;
    z-index: 1000;
}

.dropdown-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    color: #fca5a5;
    cursor: pointer;
    transition: all 0.3s ease;
}

.dropdown-item:hover {
    background: #374151;
}

/* Main Content - Add padding for fixed navbar */
.main-content {
    padding-top: 70px;
    padding-left: 2rem;
    padding-right: 2rem;
}

/* Main Header */
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
    height: 100%;
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

.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1f2937;
}

/* Welcome Screen */
.welcome-container {
    max-width: 380px;
    margin: 80px auto;
    text-align: center;
}

.welcome-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    color: #ffffff;
}

.welcome-gradient {
    color: #3b82f6;
}

.welcome-subtitle {
    color: #9ca3af;
    margin-bottom: 1.5rem;
    font-size: 0.8rem;
}

.welcome-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.2rem;
}

/* Quiz Start Block */
.quiz-start-container {
    max-width: 380px;
    margin: 40px auto;
    text-align: center;
}

.quiz-start-card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 1.5rem;
}

.quiz-start-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
}

.quiz-start-title {
    font-size: 1rem;
    font-weight: 600;
    color: #ffffff;
    margin-bottom: 0.5rem;
}

.quiz-start-desc {
    color: #9ca3af;
    font-size: 0.75rem;
    margin-bottom: 1rem;
}

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

/* Back button */
.back-home-btn {
    margin-top: 2rem;
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
        st.session_state.show_user_menu = False
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


def toggle_user_menu():
    st.session_state.show_user_menu = not st.session_state.show_user_menu
    st.rerun()


def render_navbar():
    """Render top navbar - always visible"""
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    current_page = st.session_state.page
    
    # Navbar items
    nav_items = [
        ("🏠 Home", "home"),
        ("📄 Analyze CV", "analyze"),
        ("🎯 JD Match", "jd_match"),
        ("🧠 Quiz", "quiz"),
        ("ℹ️ About", "about"),
        ("📞 Contact", "contact")
    ]
    
    # Build navbar HTML
    nav_buttons_html = ""
    for label, page_key in nav_items:
        active_class = "nav-item-active" if current_page == page_key else ""
        nav_buttons_html += f'<button class="nav-item {active_class}" onclick="navigateTo(\'{page_key}\')">{label}</button>'
    
    st.markdown(f"""
    <div class="top-navbar">
        <div class="nav-container">
            <div class="nav-logo">
                <span class="nav-logo-icon">🎯</span>
                <span>AI Career Platform</span>
            </div>
            <div class="nav-menu">
                {nav_buttons_html}
            </div>
            <div class="nav-user" onclick="document.getElementById('user_dropdown_btn').click()">
                <span>👤</span>
                <span class="user-name">{first}</span>
                <span class="user-arrow">▼</span>
            </div>
        </div>
    </div>
    
    <div id="user_dropdown_btn" style="display: none;"></div>
    
    <script>
    function navigateTo(page) {{
        const url = new URL(window.location.href);
        url.searchParams.set('nav', page);
        window.location.href = url.toString();
    }}
    </script>
    """, unsafe_allow_html=True)
    
    # Handle dropdown toggle with a hidden button
    col1, col2, col3 = st.columns([1, 1, 10])
    with col1:
        if st.button("", key="user_dropdown_btn"):
            toggle_user_menu()
    
    # Show dropdown menu if expanded
    if st.session_state.show_user_menu:
        st.markdown("""
        <style>
        .dropdown-container {
            position: fixed;
            top: 60px;
            right: 30px;
            z-index: 1000000;
        }
        </style>
        """, unsafe_allow_html=True)
        with st.container():
            col1, col2, col3 = st.columns([8, 2, 2])
            with col2:
                st.markdown("""
                <div style="background: #1f2937; border: 1px solid #374151; border-radius: 12px; padding: 0.5rem; min-width: 140px;">
                    <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem 1rem; border-radius: 8px; color: #fca5a5; cursor: pointer;">
                        <span>⏻</span>
                        <span>Sign Out</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if st.button("Sign Out", key="signout_dropdown", use_container_width=True):
                    sign_out()
    
    # Handle navigation from URL params
    query_params = st.query_params
    if 'nav' in query_params:
        target = query_params['nav']
        if target in ['home', 'analyze', 'jd_match', 'quiz', 'about', 'contact']:
            if st.session_state.page != target:
                st.session_state.page = target
                st.query_params.clear()
                st.rerun()


# ============================================================
# WELCOME SCREEN
# ============================================================
def render_welcome():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-top: 60px;">
            <div style="font-family: 'Inter', sans-serif; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.5rem;">
                AI <span style="color: #3b82f6;">Career</span> Platform
            </div>
            <p style="color: #9ca3af; margin-bottom: 1.5rem; font-size: 0.75rem;">Your AI-powered career companion</p>
            <div style="background: #111827; border: 1px solid #1f2937; border-radius: 16px; padding: 1.2rem;">
                <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 1rem; color: #e5e7eb;">👋 Welcome! What's your name?</div>
        """, unsafe_allow_html=True)
        
        with st.form(key="welcome_form"):
            name = st.text_input("Name", placeholder="e.g. Talha", label_visibility="collapsed")
            submit = st.form_submit_button("✨ Get Started →", use_container_width=True, type="primary")
            
            if submit:
                if name and name.strip():
                    st.session_state.candidate_name = name.strip()
                    st.session_state.name_entered = True
                    st.rerun()
                else:
                    st.error("Please enter your name")
        
        st.markdown("</div></div></div>", unsafe_allow_html=True)


# ============================================================
# HOME PAGE
# ============================================================
def render_home():
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    
    st.markdown(f"""
    <div class="main-content">
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
            <div class="feature-desc">Upload your CV and get matched with the best AI/ML roles.</div>
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
            <div class="feature-desc">Paste a job description and see how well your CV aligns.</div>
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
    
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ANALYZE PAGE
# ============================================================
def render_analyze():
    st.markdown("""
    <div class="main-content">
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
    
    st.markdown("---")
    if st.button("← Back to Home", key="back_home_analyze", use_container_width=False):
        nav_goto("home")
    
    st.markdown("</div>", unsafe_allow_html=True)


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


# ============================================================
# JD MATCH PAGE
# ============================================================
def render_jd_match():
    st.markdown("""
    <div class="main-content">
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
    
    st.markdown("---")
    if st.button("← Back to Home", key="back_home_jd", use_container_width=False):
        nav_goto("home")
    
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


# ============================================================
# QUIZ PAGE
# ============================================================
def render_quiz():
    st.markdown("""
    <div class="main-content">
        <div class="main-header">
            <h1>🧠 Career Interest Quiz</h1>
            <p>Discover which AI/ML roles match your thinking style and interests.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.quiz_result:
        render_quiz_results()
        st.markdown("---")
        if st.button("← Back to Home", key="back_home_quiz", use_container_width=False):
            nav_goto("home")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    if not st.session_state.quiz_responses:
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            st.markdown("""
            <div class="quiz-start-container">
                <div class="quiz-start-card">
                    <div class="quiz-start-icon">📋</div>
                    <div class="quiz-start-title">Ready to discover your career fit?</div>
                    <div class="quiz-start-desc">Answer 10 questions about your preferences and thinking style.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Start Quiz", use_container_width=True):
                st.session_state.quiz_responses = {q["id"]: None for q in QUESTIONS}
                st.rerun()
        
        st.markdown("---")
        if st.button("← Back to Home", key="back_home_quiz_start", use_container_width=False):
            nav_goto("home")
        st.markdown("</div>", unsafe_allow_html=True)
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
    
    st.markdown("</div>", unsafe_allow_html=True)


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
    <div class="main-content">
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
    
    st.markdown("---")
    if st.button("← Back to Home", key="back_home_about", use_container_width=False):
        nav_goto("home")
    
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CONTACT PAGE
# ============================================================
def render_contact():
    st.markdown("""
    <div class="main-content">
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
    
    st.markdown("---")
    if st.button("← Back to Home", key="back_home_contact", use_container_width=False):
        nav_goto("home")
    
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MAIN ROUTER
# ============================================================
def main():
    if not st.session_state.name_entered:
        render_welcome()
        return
    
    render_navbar()
    
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
