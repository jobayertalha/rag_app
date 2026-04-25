"""
app.py — AI Career Platform
Clean White & Blue Theme (Like Reference Image)
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
# CSS - Clean White & Blue Theme (Like Reference Image)
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

.stApp {
    background: #ffffff;
    min-height: 100vh;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Sidebar collapse button */
[data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    background: #f0f2f5 !important;
    border-radius: 8px !important;
    margin: 0.5rem !important;
    z-index: 999999 !important;
}

[data-testid="stSidebarCollapseButton"] svg {
    fill: #1a73e8 !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #f8f9fa;
    border-right: 1px solid #e8eaed;
    padding-top: 0.2rem !important;
}

/* Sidebar buttons - clean white/blue theme */
[data-testid="stSidebar"] .stButton > button {
    border-radius: 10px !important;
    padding: 0.5rem 0.25rem !important;
    margin-bottom: 0.35rem !important;
    transition: all 0.2s ease !important;
    white-space: pre-line !important;
    line-height: 1.3 !important;
    height: auto !important;
    min-height: 56px !important;
    font-weight: 500 !important;
    font-size: 0.7rem !important;
    background: transparent !important;
    border: 1px solid #dadce0 !important;
    color: #3c4043 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(26, 115, 232, 0.04) !important;
    border-color: #1a73e8 !important;
    color: #1a73e8 !important;
    transform: translateY(-1px);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: rgba(26, 115, 232, 0.08) !important;
    border: 1px solid #1a73e8 !important;
    color: #1a73e8 !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background: rgba(26, 115, 232, 0.12) !important;
    border-color: #1557b0 !important;
    color: #1557b0 !important;
}

/* Main content */
.main-content {
    padding: 0.5rem 2rem 2rem 2rem;
}

/* Main header */
.main-header {
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 2px solid #e8eaed;
}

.main-header h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #202124;
    margin-bottom: 0.25rem;
}

.main-header p {
    color: #5f6368;
    font-size: 0.9rem;
}

/* Feature cards */
.feature-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s ease;
    height: 100%;
}

.feature-card:hover {
    transform: translateY(-4px);
    border-color: #1a73e8;
    box-shadow: 0 20px 40px rgba(26, 115, 232, 0.1);
}

.feature-icon {
    font-size: 2rem;
    margin-bottom: 0.75rem;
}

.feature-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #202124;
    margin-bottom: 0.5rem;
}

.feature-desc {
    color: #5f6368;
    font-size: 0.8rem;
    line-height: 1.4;
}

.feature-tags {
    margin-top: 0.75rem;
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
}

.feature-tag {
    background: #f0f2f5;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    font-size: 0.65rem;
    color: #1a73e8;
}

/* Result cards */
.result-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
}

.match-score {
    text-align: center;
    padding: 1rem;
}

.match-percentage {
    font-size: 2.5rem;
    font-weight: 800;
    color: #1a73e8;
}

/* Skill chips */
.skill-chip {
    display: inline-block;
    background: #f0f2f5;
    border: 1px solid #dadce0;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    color: #3c4043;
    margin: 0.2rem;
}

.gap-chip {
    background: #fce8e6;
    border-color: #ea868f;
    color: #c5221f;
}

/* Form elements */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 1px solid #dadce0 !important;
    border-radius: 12px !important;
    color: #202124 !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2) !important;
}

/* Button styling */
.stButton > button {
    background: #1a73e8 !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

.stButton > button:hover {
    background: #1557b0 !important;
}

/* Contact & About pages */
.contact-card, .about-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 1.5rem;
}

.contact-item {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid #e8eaed;
}

.contact-item:last-child {
    border-bottom: none;
}

.contact-icon {
    font-size: 1.2rem;
    min-width: 40px;
    color: #1a73e8;
}

.contact-label {
    font-weight: 600;
    color: #202124;
    min-width: 90px;
    font-size: 0.85rem;
}

.contact-value {
    color: #5f6368;
    font-size: 0.85rem;
}

.contact-link {
    color: #1a73e8;
    text-decoration: none;
}

.social-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}

.social-card {
    background: #f8f9fa;
    border: 1px solid #e8eaed;
    border-radius: 12px;
    padding: 0.8rem;
    text-align: center;
    transition: all 0.3s ease;
    text-decoration: none;
}

.social-card:hover {
    background: #f0f2f5;
    transform: translateY(-3px);
}

.social-icon {
    font-size: 1.3rem;
    margin-bottom: 0.3rem;
}

.social-name {
    color: #3c4043;
    font-size: 0.75rem;
    font-weight: 500;
}

.interest-tag {
    display: inline-block;
    background: #f0f2f5;
    border: 1px solid #dadce0;
    border-radius: 20px;
    padding: 0.25rem 0.8rem;
    font-size: 0.7rem;
    color: #1a73e8;
    margin: 0.2rem;
}

.tech-stack {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.8rem;
}

.tech-pill {
    background: #f0f2f5;
    border: 1px solid #dadce0;
    border-radius: 20px;
    padding: 0.2rem 0.6rem;
    font-size: 0.65rem;
    color: #3c4043;
}

.profile-header {
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #e8eaed;
}

.profile-icon {
    font-size: 3rem;
    margin-bottom: 0.3rem;
}

.profile-name {
    font-size: 1.3rem;
    font-weight: 700;
    color: #202124;
    margin-bottom: 0.2rem;
}

.profile-title {
    color: #1a73e8;
    font-size: 0.8rem;
}

.section-header {
    font-size: 0.9rem;
    font-weight: 600;
    color: #202124;
    margin: 1rem 0 0.5rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #e8eaed;
}

/* Welcome screen */
.welcome-container {
    max-width: 380px;
    margin: 60px auto;
    text-align: center;
}

.welcome-title {
    font-size: 1.8rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    color: #202124;
}

.welcome-gradient {
    color: #1a73e8;
    background: linear-gradient(135deg, #1a73e8, #4285f4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.welcome-subtitle {
    color: #5f6368;
    margin-bottom: 1.2rem;
    font-size: 0.85rem;
}

.welcome-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Quiz styles */
.quiz-start-container {
    max-width: 380px;
    margin: 30px auto;
    text-align: center;
}

.quiz-start-card {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 1.2rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.quiz-start-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
}

.quiz-start-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #202124;
    margin-bottom: 0.3rem;
}

.quiz-start-desc {
    color: #5f6368;
    font-size: 0.7rem;
    margin-bottom: 0.8rem;
}

.quiz-question {
    background: #ffffff;
    border: 1px solid #e8eaed;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

.quiz-question-text {
    font-weight: 600;
    color: #202124;
    margin-bottom: 0.8rem;
    font-size: 0.9rem;
}

/* Radio buttons */
.stRadio > div {
    gap: 0.5rem;
}

.stRadio label {
    color: #3c4043 !important;
    font-size: 0.8rem !important;
}

/* Success/Warning/Info boxes */
.stAlert {
    border-radius: 12px !important;
}

/* Expander */
.streamlit-expanderHeader {
    color: #202124 !important;
    background: #f8f9fa !important;
    border-radius: 12px !important;
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
    st.session_state.cv_text = None
    st.session_state.agent = None
    st.session_state.analysis_raw = None
    st.session_state.retrieved = None
    st.session_state.jd_match_result = None
    st.session_state.quiz_responses = {}
    st.session_state.quiz_result = None
    st.rerun()


def render_sidebar():
    """Sidebar with clean white/blue theme"""
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    current_page = st.session_state.page
    
    with st.sidebar:
        # Brand - Clean white/blue
        st.markdown("""
        <div style="text-align: center; padding: 0.8rem 0 0.6rem 0; border-bottom: 1px solid #e8eaed; margin-bottom: 0.8rem;">
            <div style="font-size: 1.8rem; margin-bottom: 0.2rem;">🎯</div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.9rem; font-weight: 700; background: linear-gradient(135deg, #1a73e8, #4285f4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">AI Career Platform</div>
        </div>
        """, unsafe_allow_html=True)
        
        # User name
        st.markdown(f"""
        <div style="background: #f0f2f5; border-radius: 20px; padding: 0.35rem 0.6rem; margin-bottom: 0.8rem; text-align: center;">
            <span style="color: #1a73e8; font-size: 0.75rem; font-weight: 500;">👤 {first}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation buttons
        nav_items = [
            ("🏠 Home", "home"),
            ("📄 Analyze CV", "analyze"),
            ("🎯 JD Match", "jd_match"),
            ("🧠 Quiz", "quiz"),
            ("ℹ️ About", "about"),
            ("📞 Contact", "contact")
        ]
        
        for label, page_key in nav_items:
            if current_page == page_key:
                st.button(label, key=f"nav_{page_key}", use_container_width=True, type="primary")
            else:
                if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                    nav_goto(page_key)
        
        st.markdown("<div style='margin: 0.8rem 0;'></div>", unsafe_allow_html=True)
        
        # Sign out button
        if st.button("⏻ Sign Out", key="signout_btn", use_container_width=True):
            sign_out()
        
        st.caption("© 2025 AI Career Platform")


# ============================================================
# WELCOME SCREEN
# ============================================================
def render_welcome():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    
    with col2:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-title">
                AI <span class="welcome-gradient">Career</span> Platform
            </div>
            <p class="welcome-subtitle">Your AI-powered career companion</p>
            <div class="welcome-card">
                <div style="font-size: 0.85rem; font-weight: 600; margin-bottom: 0.8rem; color: #202124;">👋 Welcome! What's your name?</div>
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
            <div style="font-size: 0.75rem; color: #5f6368; margin-bottom: 0.3rem;">MATCH SCORE</div>
            <div class="match-percentage">{top_match.get('match_pct', 0)}%</div>
            <h3 style="color: #202124; margin-top: 0.3rem; font-size: 1rem;">{top_match.get('title', top_match.get('role', 'AI Professional'))}</h3>
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
        <div style="margin-bottom: 0.8rem;">
            <strong>{r.get('title', r.get('role', 'Role'))}</strong> — {r.get('company', 'Various')} 
            <span style="color: #1a73e8;">({r.get('match_pct', 0)}% match)</span>
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
        color, status = "#c5221f", "Low Match"
    elif pct < 60:
        color, status = "#e37400", "Partial Match"
    elif pct < 80:
        color, status = "#188038", "Good Match"
    else:
        color, status = "#1a73e8", "Excellent Match!"
    
    st.markdown(f"""
    <div class="result-card">
        <div style="text-align:center;">
            <div style="font-size: 0.75rem; color: #5f6368;">MATCH SCORE</div>
            <div style="font-size: 2.5rem; font-weight: 800; color: {color};">{pct}%</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: {color};">{status}</div>
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
    color = "#1a73e8" if level == "HIGH" else ("#e37400" if level == "MEDIUM" else "#c5221f")
    
    st.markdown(f"""
    <div class="result-card">
        <div style="text-align:center;">
            <div style="font-size: 2.5rem; font-weight: 800; color: {color};">{pct}%</div>
            <div style="font-size: 1rem; font-weight: 600; margin: 0.3rem 0; color: #202124;">Interest Level: {level}</div>
            <div style="color: #5f6368; font-size: 0.85rem;">{result["message"]}</div>
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
    <div style="margin-bottom: 1rem;">
        <span class="interest-tag">Natural Language Processing (NLP)</span>
        <span class="interest-tag">Computer Vision</span>
        <span class="interest-tag">Cyber Security</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🎓 Academic Affiliation</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #3c4043; margin-bottom: 1rem; font-size: 0.85rem;">Department of Computer Science & Engineering<br>Rajshahi University of Engineering & Technology (RUET)</p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">🚀 About This Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: #5f6368; line-height: 1.5; margin-bottom: 1rem; font-size: 0.85rem;">AI Career Platform is an intelligent career matching system designed to help job seekers in Bangladesh find the best AI/ML roles based on their CV content, skills, and career preferences.</p>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">✨ Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <ul style="color: #5f6368; line-height: 1.6; margin-bottom: 1rem; font-size: 0.8rem;">
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
    
    <div class="section-header" style="margin-top: 1.5rem; text-align: center;">Connect With Me</div>
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
