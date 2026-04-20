"""
app.py — Professional CV Analyzer Platform
Multi-feature platform with CV Analysis, JD Matching, AI/ML Quiz, and more
"""

import streamlit as st
import tempfile
import os
import re
import json
from datetime import datetime

from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context, load_roles, match_cv_with_jd, score_ai_ml_readiness

# Page configuration
st.set_page_config(
    page_title="CV Analyzer Pro | AI Career Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# PROFESSIONAL DARK THEME CSS
# ============================================================
st.markdown("""
<style>
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
footer {visibility: hidden;}
.stApp {
    background: linear-gradient(135deg, #0a0a14 0%, #0f0f20 100%);
}

/* Professional Navbar */
.navbar {
    background: rgba(10, 10, 20, 0.95);
    backdrop-filter: blur(10px);
    border-bottom: 1px solid rgba(168, 85, 247, 0.2);
    padding: 0.75rem 2rem;
    margin-bottom: 2rem;
    position: sticky;
    top: 0;
    z-index: 1000;
}

.nav-container {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1400px;
    margin: 0 auto;
}

.nav-logo {
    font-size: 1.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-decoration: none;
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
    font-size: 0.9rem;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.2s ease;
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

.user-profile {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(168, 85, 247, 0.1);
    padding: 0.4rem 1rem;
    border-radius: 30px;
    cursor: pointer;
    transition: all 0.2s;
}

.user-profile:hover {
    background: rgba(168, 85, 247, 0.2);
}

.user-avatar {
    width: 32px;
    height: 32px;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
    font-weight: 600;
}

.user-name {
    color: #e2e8f0;
    font-size: 0.9rem;
    font-weight: 500;
}

/* Card styles */
.card {
    background: rgba(15, 15, 32, 0.8);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(168, 85, 247, 0.15);
    border-radius: 20px;
    padding: 1.5rem;
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.card:hover {
    border-color: rgba(168, 85, 247, 0.3);
    transform: translateY(-2px);
}

.gradient-text {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Hero section */
.hero-section {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    margin-bottom: 1rem;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: #94a3b8;
    max-width: 600px;
    margin: 0 auto;
}

/* Feature grid */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
    margin: 2rem 0;
}

.feature-card {
    background: rgba(15, 15, 32, 0.6);
    border: 1px solid rgba(168, 85, 247, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.3s;
}

.feature-card:hover {
    border-color: #a855f7;
    transform: translateY(-4px);
}

.feature-icon {
    font-size: 2.5rem;
    margin-bottom: 1rem;
}

.feature-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}

.feature-desc {
    font-size: 0.85rem;
    color: #64748b;
}

/* Skill chips */
.skill-chip {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    font-size: 0.75rem;
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin: 0.25rem;
}

.gap-chip {
    background: rgba(239, 68, 68, 0.15);
    border-color: rgba(239, 68, 68, 0.3);
    color: #fca5a5;
}

/* Quiz styles */
.quiz-question {
    background: rgba(15, 15, 32, 0.6);
    border: 1px solid rgba(168, 85, 247, 0.15);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.quiz-option {
    background: rgba(30, 30, 58, 0.5);
    border: 1px solid rgba(168, 85, 247, 0.2);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    cursor: pointer;
    transition: all 0.2s;
}

.quiz-option:hover {
    background: rgba(168, 85, 247, 0.1);
    border-color: #a855f7;
}

.score-badge {
    font-size: 2rem;
    font-weight: 800;
    text-align: center;
    padding: 1rem;
    border-radius: 20px;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.6rem 1.5rem;
    font-weight: 600;
    transition: all 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(168, 85, 247, 0.3);
}

/* File uploader */
.upload-container {
    border: 2px dashed rgba(168, 85, 247, 0.3);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    background: rgba(15, 15, 32, 0.4);
}

/* Divider */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #a855f7, #ec4899, transparent);
    margin: 1.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "page": "home",
        "cv_text": None,
        "candidate_name": "",
        "name_entered": False,
        "agent": None,
        "analysis_raw": None,
        "retrieved": None,
        "matched_companies": [],
        "jd_result": None,
        "quiz_answers": {},
        "quiz_completed": False,
        "quiz_score": 0,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# ============================================================
# NAVIGATION BAR
# ============================================================
def render_navbar():
    """Render professional navigation bar"""
    name = st.session_state.candidate_name or "Guest"
    first_letter = name[0].upper() if name else "G"
    
    # Navigation items
    nav_items = [
        ("🏠 Home", "home"),
        ("📊 Analyze", "analyze"),
        ("🎯 Matching", "matching"),
        ("📝 Quiz", "quiz"),
        ("ℹ️ About", "about"),
        ("📞 Contact", "contact"),
    ]
    
    # Custom HTML for navbar
    nav_html = f'''
    <div class="navbar">
        <div class="nav-container">
            <div style="display: flex; align-items: center; gap: 2rem;">
                <span class="nav-logo">🎯 CV Analyzer Pro</span>
                <div class="nav-links">
    '''
    
    for label, page_id in nav_items:
        active_class = "nav-btn-active" if st.session_state.page == page_id else ""
        nav_html += f'<button class="nav-btn {active_class}" onclick="parent.postMessage({{type: "streamlit:setComponentValue", value: "nav_{page_id}"}}, "*")">{label}</button>'
    
    nav_html += f'''
                </div>
            </div>
            <div class="user-profile">
                <div class="user-avatar">{first_letter}</div>
                <span class="user-name">{name}</span>
            </div>
        </div>
    </div>
    '''
    
    st.markdown(nav_html, unsafe_allow_html=True)
    
    # Handle navigation via columns (reliable method)
    col1, col2, col3, col4, col5, col6 = st.columns([1.2, 0.8, 0.8, 0.8, 0.8, 0.8])
    
    with col1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        if st.button("📊 Analyze", key="nav_analyze", use_container_width=True):
            if st.session_state.cv_text:
                st.session_state.page = "analyze"
            else:
                st.warning("Please upload and analyze your CV first on the Home page")
            st.rerun()
    with col3:
        if st.button("🎯 Matching", key="nav_matching", use_container_width=True):
            if st.session_state.cv_text:
                st.session_state.page = "matching"
            else:
                st.warning("Please upload your CV first on the Home page")
            st.rerun()
    with col4:
        if st.button("📝 Quiz", key="nav_quiz", use_container_width=True):
            st.session_state.page = "quiz"
            st.rerun()
    with col5:
        if st.button("ℹ️ About", key="nav_about", use_container_width=True):
            st.session_state.page = "about"
            st.rerun()
    with col6:
        if st.button("📞 Contact", key="nav_contact", use_container_width=True):
            st.session_state.page = "contact"
            st.rerun()
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)


# ============================================================
# HOME PAGE - Welcome & CV Upload
# ============================================================
def render_home():
    """Home page with welcome message and CV upload"""
    
    if not st.session_state.name_entered:
        # Name entry screen
        st.markdown("""
        <div class="hero-section">
            <div class="hero-title">
                Welcome to <span class="gradient-text">CV Analyzer Pro</span>
            </div>
            <div class="hero-subtitle">
                AI-powered career matching platform for data science & AI professionals
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 1.5, 1])
        with col2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 👋 Let's get started!")
                st.markdown("What's your name?")
                
                name = st.text_input("Name", placeholder="e.g., Talha Jobayer", label_visibility="collapsed")
                
                if st.button("✨ Start Your Journey →", use_container_width=True, type="primary"):
                    if name and name.strip():
                        st.session_state.candidate_name = name.strip()
                        st.session_state.name_entered = True
                        st.rerun()
                    else:
                        st.error("Please enter your name to continue")
                st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Main home page content
    name = st.session_state.candidate_name.split()[0]
    
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-title">
            Hello, {name}! 👋
        </div>
        <div class="hero-subtitle">
            Ready to take your career to the next level?
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # CV Upload Section
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📄 Upload Your CV")
    st.markdown("Support PDF format. Your CV will be analyzed for AI/ML role compatibility.")
    
    uploaded_file = st.file_uploader("Choose CV file", type=["pdf"], label_visibility="collapsed")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        if uploaded_file and st.button("🚀 Analyze CV", type="primary", use_container_width=True):
            process_cv_analysis(uploaded_file)
    
    if st.session_state.cv_text:
        st.success("✅ CV loaded and ready for analysis!")
        st.info(f"📊 CV contains {len(st.session_state.cv_text)} characters")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features grid
    st.markdown("### ✨ Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">CV Analysis</div>
            <div class="feature-desc">Get detailed analysis of your CV's strengths and weaknesses for AI/ML roles</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">JD Matching</div>
            <div class="feature-desc">Match your CV against any job description and get compatibility scores</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📝</div>
            <div class="feature-title">AI/ML Quiz</div>
            <div class="feature-desc">Test your knowledge and get personalized career recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick stats if CV is loaded
    if st.session_state.cv_text and st.session_state.retrieved:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📈 Your Quick Stats")
        
        readiness = st.session_state.retrieved.get("readiness", {})
        top_match = st.session_state.retrieved.get("top_match", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Readiness Score", f"{readiness.get('total_score', 0)}%")
        with col2:
            st.metric("Top Match", top_match.get("match_pct", 0), "%")
        with col3:
            st.metric("Skills Found", st.session_state.retrieved.get("readiness", {}).get("stats", {}).get("ai_skills_found", 0))
        with col4:
            st.metric("Target Roles", len(st.session_state.retrieved.get("all_matches", [])))


def process_cv_analysis(uploaded_file):
    """Process CV upload and run analysis"""
    with st.spinner("🔍 Analyzing your CV..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            cv_text = extract_cv_text(tmp.name)
            os.unlink(tmp.name)
        
        st.session_state.cv_text = cv_text
        st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
        st.session_state.matched_companies = match_companies(cv_text)
        st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
        
        # Run analysis
        raw = run_agent(st.session_state.agent, 
                        "Analyse this CV. Follow tags: TOP_ROLE, MATCH_PCT, WHY_RIGHT, SKILL_GAPS, RESUME_ADD, CAREER_PATH")
        st.session_state.analysis_raw = raw
        st.session_state.page = "analyze"
        st.rerun()


def match_companies(cv_text: str) -> list:
    """Match CV with companies from knowledge base"""
    cv_lower = cv_text.lower()
    companies = []
    try:
        roles = load_roles()
        for role in roles:
            if role.get("company"):
                skills = role.get("skills", [])
                found = sum(1 for s in skills if s.lower() in cv_lower)
                score = (found / max(len(skills), 1)) * 100
                companies.append({
                    "name": role["company"],
                    "role": role.get("title", role.get("role")),
                    "match_score": round(score, 1),
                    "location": role.get("location", "Dhaka"),
                })
        companies.sort(key=lambda x: x["match_score"], reverse=True)
        return companies[:6]
    except:
        return []


# ============================================================
# ANALYZE PAGE - CV Analysis Results
# ============================================================
def render_analyze():
    """Display detailed CV analysis results"""
    if not st.session_state.cv_text:
        st.warning("⚠️ Please upload your CV first on the Home page.")
        if st.button("← Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    st.markdown("### 📊 CV Analysis Report")
    
    # Parse analysis
    parsed = parse_analysis(st.session_state.analysis_raw or "") if st.session_state.analysis_raw else {}
    retrieved = st.session_state.retrieved or {}
    readiness = retrieved.get("readiness", {})
    top_match = retrieved.get("top_match", {})
    
    # Hero Score Card
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        score = parsed.get("match_pct", top_match.get("match_pct", 0))
        score_int = int(score) if score else 0
        color = "#ef4444" if score_int < 40 else "#f59e0b" if score_int < 70 else "#10b981"
        
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div style="font-size: 1rem; color: #94a3b8;">Career Match Score</div>
            <div style="font-size: 4rem; font-weight: 800; color: {color};">{score_int}%</div>
            <div style="font-size: 1.2rem; font-weight: 600; margin-top: 0.5rem;">{parsed.get('top_role', top_match.get('title', 'AI Professional'))}</div>
            <div style="color: #64748b; margin-top: 0.5rem;">{parsed.get('why_right', 'Great alignment with your profile!')[:150]}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for detailed analysis
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Overview", "🎯 Role Matches", "🔧 Skill Gaps", "📈 Career Path", "🏢 Companies"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📊 Readiness Assessment")
            
            total_score = readiness.get("total_score", 0)
            if total_score < 30:
                st.warning(f"🔴 Readiness Score: {total_score}% - Focus on fundamentals")
                st.markdown("""
                **Recommendations:**
                - Complete Python and ML basics
                - Build 2-3 portfolio projects
                - Get foundational certifications
                """)
            elif total_score < 60:
                st.info(f"🟡 Readiness Score: {total_score}% - Building momentum")
                st.markdown("""
                **Recommendations:**
                - Take advanced ML courses
                - Build real-world projects
                - Gain cloud experience
                """)
            else:
                st.success(f"🟢 Readiness Score: {total_score}% - Ready to apply!")
                st.markdown("""
                **Recommendations:**
                - Update LinkedIn and portfolio
                - Start applying to target companies
                - Prepare for technical interviews
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 📈 Key Metrics")
            stats = readiness.get("stats", {})
            st.metric("AI/ML Skills Found", stats.get("ai_skills_found", 0))
            st.metric("Projects Detected", stats.get("projects_found", 0))
            st.metric("Certifications", stats.get("certificates_found", 0))
            st.metric("Has AI Experience", "✅" if stats.get("has_ai_experience") else "❌")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Breakdown
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Score Breakdown")
        breakdown = readiness.get("breakdown", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Experience", f"{breakdown.get('experience', 0)}/30")
        with col2:
            st.metric("Projects", f"{breakdown.get('projects', 0)}/25")
        with col3:
            st.metric("Certificates", f"{breakdown.get('certificates', 0)}/20")
        with col4:
            st.metric("Skills", f"{breakdown.get('skills', 0)}/25")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### 🎯 Best Matching Roles")
        all_matches = retrieved.get("all_matches", [])
        
        for i, role in enumerate(all_matches[:4]):
            with st.expander(f"{i+1}. {role.get('title', role.get('role', 'Unknown'))} - {role.get('match_pct', 0)}% Match", expanded=i==0):
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**Company:** {role.get('company', 'Various')}")
                    st.markdown(f"**Category:** {role.get('category', 'N/A')}")
                    st.markdown(f"**Location:** {role.get('location', 'Dhaka')}")
                    if role.get('description'):
                        st.markdown(f"**Description:** {role.get('description', '')[:200]}...")
                with col2:
                    st.markdown(f"**💰 Salary Range:** ৳{role.get('salary_min', 0):,} - ৳{role.get('salary_max', 0):,}")
                    st.markdown(f"**Market Demand:** {role.get('market_demand', 'Medium')}")
                
                st.markdown("**Required Skills:**")
                skills = role.get("skills", [])
                cols = st.columns(4)
                for idx, skill in enumerate(skills[:8]):
                    with cols[idx % 4]:
                        st.markdown(f"• {skill}")
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### ❌ Skill Gaps to Fill")
            gaps = parsed.get("skill_gaps") or retrieved.get("skill_gaps", [])
            if gaps:
                for gap in gaps[:8]:
                    st.markdown(f'<span class="skill-chip gap-chip">{gap}</span>', unsafe_allow_html=True)
            else:
                st.success("Great! No major skill gaps detected.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### ✅ Skills to Add to Resume")
            resume_skills = parsed.get("resume_add") or retrieved.get("resume_skills", [])
            if resume_skills:
                for skill in resume_skills[:8]:
                    st.markdown(f'<span class="skill-chip">{skill}</span>', unsafe_allow_html=True)
            else:
                st.info("Your CV already has good skill alignment.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Learning resources
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📚 Recommended Learning Resources")
        st.markdown("""
        - **Coursera:** Machine Learning Specialization (Andrew Ng)
        - **Fast.ai:** Practical Deep Learning
        - **Hugging Face:** NLP Course
        - **Kaggle:** Data Science competitions
        - **DeepLearning.AI:** LLM and RAG courses
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🗺️ Your Career Pathway")
        career_path = parsed.get("career_path", [])
        
        if career_path:
            for i, step in enumerate(career_path[:5]):
                st.markdown(f"**Step {i+1}:** {step}")
                if i < len(career_path) - 1:
                    st.markdown("↓")
        else:
            # Default career path
            st.markdown("""
            **1. Foundation (0-6 months)**
            - Complete Python and ML basics
            - Build 2-3 portfolio projects
            
            **2. Skill Building (6-12 months)**
            - Advanced ML/DL courses
            - Kaggle competitions
            - Cloud certifications
            
            **3. Professional (1-2 years)**
            - Apply for entry-level roles
            - Build professional network
            - Contribute to open source
            
            **4. Growth (2-4 years)**
            - Specialize in AI subfield
            - Lead projects
            - Mentor juniors
            """)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Salary expectations
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 💰 Salary Expectations (Bangladesh Market)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Entry Level", "25k - 40k", "BDT/month")
        with col2:
            st.metric("Mid Level", "45k - 70k", "BDT/month")
        with col3:
            st.metric("Senior Level", "80k - 150k+", "BDT/month")
        st.caption("*Based on current market data for AI/ML roles in Bangladesh")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown("#### 🏢 Recommended Companies")
        companies = st.session_state.matched_companies
        
        if companies:
            for company in companies[:6]:
                st.markdown(f"""
                <div style="background: rgba(15, 15, 32, 0.6); border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem;">
                    <strong>{company['name']}</strong><br>
                    Role: {company['role']} | Match: {company['match_score']}%<br>
                    📍 {company['location']}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No company matches found. Try uploading a more detailed CV.")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Re-analyze CV", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()
    with col2:
        if st.button("🎯 Try JD Matching", use_container_width=True):
            st.session_state.page = "matching"
            st.rerun()


def parse_analysis(text: str) -> dict:
    """Parse LLM analysis response"""
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
        "skill_gaps": get_list("SKILL_GAPS"),
        "resume_add": get_list("RESUME_ADD"),
        "career_path": get_list("CAREER_PATH"),
    }


# ============================================================
# MATCHING PAGE - JD vs CV Matching
# ============================================================
def render_matching():
    """Job Description matching page"""
    if not st.session_state.cv_text:
        st.warning("⚠️ Please upload your CV first on the Home page.")
        if st.button("← Go to Home"):
            st.session_state.page = "home"
            st.rerun()
        return
    
    st.markdown("### 🎯 Job Description Matching")
    st.markdown("Paste a job description to see how well your CV matches")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📄 Your CV")
        st.success(f"✅ CV loaded: {len(st.session_state.cv_text)} characters")
        st.caption(f"Name: {st.session_state.candidate_name}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 💼 Job Description")
        jd_text = st.text_area(
            "Paste JD here",
            height=200,
            placeholder="Paste the full job description here...\n\nExample:\nWe are looking for a Machine Learning Engineer with experience in Python, TensorFlow, and NLP...",
            label_visibility="collapsed"
        )
        
        if jd_text and st.button("🔍 Calculate Match", type="primary", use_container_width=True):
            calculate_jd_match(jd_text)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display results if available
    if st.session_state.jd_result:
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        render_jd_results()


def calculate_jd_match(jd_text: str):
    """Calculate match between CV and JD"""
    with st.spinner("Analyzing match..."):
        cv_text = st.session_state.cv_text
        cv_lower = cv_text.lower()
        jd_lower = jd_text.lower()
        
        # Extract keywords from JD
        keywords = set()
        skill_list = ['python', 'sql', 'tensorflow', 'pytorch', 'langchain', 'rag', 'llm', 
                      'nlp', 'computer vision', 'docker', 'kubernetes', 'aws', 'gcp', 
                      'pandas', 'numpy', 'scikit-learn', 'git', 'mlflow', 'huggingface',
                      'deep learning', 'machine learning', 'data science', 'analytics']
        
        for skill in skill_list:
            if skill in jd_lower:
                keywords.add(skill)
        
        found = [kw for kw in keywords if kw in cv_lower]
        missing = [kw for kw in keywords if kw not in cv_lower]
        
        # Calculate weighted score
        total_weight = len(keywords)
        found_weight = len(found)
        pct = int((found_weight / max(total_weight, 1)) * 100)
        
        # Get role-based matching from RAG
        rag_match = match_cv_with_jd(cv_text, jd_text)
        
        st.session_state.jd_result = {
            "match_pct": min(95, pct),
            "matched": found[:15],
            "missing": missing[:15],
            "found_count": len(found),
            "total": len(keywords),
            "rag_roles": rag_match.get("similar_roles", []),
            "jd_text": jd_text
        }


def render_jd_results():
    """Display JD matching results"""
    r = st.session_state.jd_result
    pct = r["match_pct"]
    
    # Determine color and message
    if pct < 30:
        color, status, message = "#ef4444", "Low Match", "Significant skill gaps detected. Focus on building missing skills."
    elif pct < 60:
        color, status, message = "#f59e0b", "Partial Match", "You have some relevant skills. Fill the gaps to improve."
    elif pct < 80:
        color, status, message = "#10b981", "Good Match", "Strong alignment! You're competitive for this role."
    else:
        color, status, message = "#06b6d4", "Excellent Match!", "Perfect fit! You're highly qualified for this position."
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div style="font-size: 1rem; color: #94a3b8;">JD Match Score</div>
            <div style="font-size: 4rem; font-weight: 800; color: {color};">{pct}%</div>
            <div style="font-size: 1.2rem; font-weight: 600; color: {color};">{status}</div>
            <div style="color: #64748b; margin-top: 0.5rem;">{message}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📊 Match Summary")
        st.metric("Keywords Found", f"{r['found_count']}/{r['total']}")
        st.metric("Match Percentage", f"{pct}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Matched skills
    if r["matched"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ✅ Matched Skills")
        for skill in r["matched"][:10]:
            st.markdown(f'<span class="skill-chip">✓ {skill}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Missing skills
    if r["missing"]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### ❌ Missing Skills (Priority to Learn)")
        for skill in r["missing"][:10]:
            st.markdown(f'<span class="skill-chip gap-chip">{skill}</span>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**📚 How to acquire these skills:**")
        st.markdown("""
        - **Online Courses:** Coursera, Udemy, Fast.ai
        - **Hands-on Projects:** Build portfolio projects
        - **Certifications:** AWS, TensorFlow, DeepLearning.AI
        - **Practice:** Kaggle competitions, LeetCode
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Similar roles from database
    if r.get("rag_roles"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🔍 Similar Roles in Our Database")
        for role in r["rag_roles"][:3]:
            st.markdown(f"""
            - **{role.get('title', role.get('role', 'Unknown'))}** at {role.get('company', 'Various')}
              - Match: {role.get('match_pct', 0)}%
              - Salary: ৳{role.get('salary_min', 0):,} - ৳{role.get('salary_max', 0):,}
            """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 New Match", use_container_width=True):
            st.session_state.jd_result = None
            st.rerun()


# ============================================================
# QUIZ PAGE - AI/ML Career Alignment Quiz
# ============================================================
def render_quiz():
    """AI/ML career alignment quiz"""
    st.markdown("### 📝 AI/ML Career Alignment Quiz")
    st.markdown("Answer these questions to discover your fit for an AI/ML career")
    
    # Quiz questions
    questions = [
        {
            "id": 1,
            "text": "How comfortable are you with Python programming?",
            "options": [
                ("I don't know Python", 0),
                ("Basic syntax only", 25),
                ("Intermediate (can write functions/classes)", 75),
                ("Advanced (libraries, OOP, debugging)", 100)
            ]
        },
        {
            "id": 2,
            "text": "What's your experience with Mathematics for ML?",
            "options": [
                ("Limited math background", 0),
                ("Basic statistics and algebra", 25),
                ("Good understanding of linear algebra & calculus", 75),
                ("Strong mathematical foundation", 100)
            ]
        },
        {
            "id": 3,
            "text": "Which best describes your ML framework experience?",
            "options": [
                ("None", 0),
                ("Basic scikit-learn only", 25),
                ("TensorFlow or PyTor basics", 75),
                ("Production experience with TF/PyTorch", 100)
            ]
        },
        {
            "id": 4,
            "text": "Have you worked on any ML/AI projects?",
            "options": [
                ("No projects", 0),
                ("1-2 tutorial projects", 25),
                ("3-5 personal projects", 75),
                ("Professional or research projects", 100)
            ]
        },
        {
            "id": 5,
            "text": "How familiar are you with Data Science concepts?",
            "options": [
                ("Not familiar", 0),
                ("Basic understanding", 25),
                ("Good working knowledge", 75),
                ("Expert level", 100)
            ]
        },
        {
            "id": 6,
            "text": "What's your experience with Cloud Platforms (AWS/GCP/Azure)?",
            "options": [
                ("No experience", 0),
                ("Basic understanding", 25),
                ("Some hands-on experience", 75),
                ("Certified/Production experience", 100)
            ]
        },
        {
            "id": 7,
            "text": "How interested are you in keeping up with AI research?",
            "options": [
                ("Not interested", 0),
                ("Occasionally read news", 25),
                ("Follow key researchers/companies", 75),
                ("Actively read papers and implement", 100)
            ]
        },
        {
            "id": 8,
            "text": "What's your familiarity with LLMs and RAG?",
            "options": [
                ("What's that?", 0),
                ("Heard about them", 25),
                ("Used APIs like GPT", 75),
                ("Built applications with RAG/LangChain", 100)
            ]
        },
        {
            "id": 9,
            "text": "How would you rate your problem-solving skills?",
            "options": [
                ("Need improvement", 0),
                ("Average", 25),
                ("Good", 75),
                ("Excellent", 100)
            ]
        },
        {
            "id": 10,
            "text": "Are you willing to continuously learn and adapt?",
            "options": [
                ("Not really", 0),
                ("Maybe", 25),
                ("Yes, I enjoy learning", 75),
                ("Absolutely, it's essential", 100)
            ]
        }
    ]
    
    # Progress tracking
    if not st.session_state.quiz_completed:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Progress bar
        answered = len(st.session_state.quiz_answers)
        progress = answered / len(questions)
        st.progress(progress, text=f"Progress: {answered}/{len(questions)} questions answered")
        
        # Show current question
        for q in questions:
            if str(q["id"]) not in st.session_state.quiz_answers:
                st.markdown(f"#### Question {q['id']}: {q['text']}")
                
                for option_text, score in q["options"]:
                    if st.button(option_text, key=f"q_{q['id']}_{option_text[:20]}", use_container_width=True):
                        st.session_state.quiz_answers[str(q["id"])] = {"text": option_text, "score": score}
                        st.rerun()
                break
        
        # Show completed if all answered
        if len(st.session_state.quiz_answers) == len(questions):
            if st.button("📊 Generate My Report", type="primary", use_container_width=True):
                calculate_quiz_score(questions)
                st.session_state.quiz_completed = True
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Show results
        render_quiz_results(questions)


def calculate_quiz_score(questions):
    """Calculate quiz score and generate recommendations"""
    total_score = 0
    max_score = len(questions) * 100
    
    for q in questions:
        answer = st.session_state.quiz_answers.get(str(q["id"]), {})
        total_score += answer.get("score", 0)
    
    percentage = (total_score / max_score) * 100
    st.session_state.quiz_score = percentage


def render_quiz_results(questions):
    """Display quiz results and recommendations"""
    percentage = st.session_state.quiz_score
    
    if percentage >= 50:
        color = "#10b981"
        recommendation = "✅ **AI/ML Career Recommended!**"
        message = "Your skills and interests align well with an AI/ML career path. With focused effort, you can build a successful career in this field."
    else:
        color = "#f59e0b"
        recommendation = "⚠️ **Consider Exploring Other Domains**"
        message = "While you have some interest, consider strengthening fundamentals or exploring related fields like Data Analysis, Software Engineering, or DevOps first."
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown(f"""
        <div class="card" style="text-align: center;">
            <div style="font-size: 1rem; color: #94a3b8;">Your Interest & Readiness Score</div>
            <div style="font-size: 4rem; font-weight: 800; color: {color};">{percentage:.0f}%</div>
            <div style="font-size: 1.1rem; font-weight: 600; margin-top: 0.5rem;">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📋 Fitment Report")
        st.markdown(message)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Detailed breakdown
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📊 Question-wise Breakdown")
    
    for q in questions:
        answer = st.session_state.quiz_answers.get(str(q["id"]), {})
        score = answer.get("score", 0)
        score_pct = score / 100
        
        st.markdown(f"""
        <div style="margin-bottom: 1rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="color: #94a3b8;">Q{q['id']}: {q['text'][:50]}...</span>
                <span style="color: {'#10b981' if score >= 75 else '#f59e0b' if score >= 50 else '#ef4444'};">{score}%</span>
            </div>
            <div style="background: rgba(30, 30, 58, 0.5); border-radius: 10px; height: 8px;">
                <div style="background: {'#10b981' if score >= 75 else '#f59e0b' if score >= 50 else '#ef4444'}; width: {score}%; height: 100%; border-radius: 10px;"></div>
            </div>
            <div style="color: #64748b; font-size: 0.75rem; margin-top: 0.25rem;">Answer: {answer.get('text', 'Not answered')}</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recommendations
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 🎯 Personalized Recommendations")
    
    if percentage >= 70:
        st.markdown("""
        **You're ready to pursue AI/ML! Next steps:**
        - Start applying for entry-level AI/ML roles
        - Build a portfolio of 3-5 strong projects
        - Get cloud certifications (AWS/Azure)
        - Network with industry professionals
        - Prepare for technical interviews (LeetCode, ML system design)
        """)
    elif percentage >= 50:
        st.markdown("""
        **You're on the right track! Focus on:**
        - Strengthening Python and math fundamentals
        - Completing structured ML courses (Andrew Ng's specialization)
        - Building 2-3 portfolio projects
        - Participating in Kaggle competitions
        - Learning production ML tools (Docker, Git, MLOps)
        """)
    else:
        st.markdown("""
        **Consider this path before AI/ML:**
        - Start with Python programming fundamentals
        - Take introductory data science courses
        - Explore if Data Analytics or Software Engineering interests you
        - Build small projects to discover your passion
        - Consider a structured bootcamp or certification program
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.quiz_answers = {}
            st.session_state.quiz_completed = False
            st.session_state.quiz_score = 0
            st.rerun()
    with col2:
        if st.button("📊 Analyze My CV", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()


# ============================================================
# ABOUT PAGE
# ============================================================
def render_about():
    """About page with profile information"""
    st.markdown("### ℹ️ About Us")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size: 4rem; margin-bottom: 1rem;">👨‍💻</div>
        <h3>Talha Jobayer Zihan</h3>
        <p style="color: #a855f7;">AI/ML Engineer & Researcher</p>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        #### About the Creator
        
        Talha Jobayer Zihan is an AI/ML Engineer and Researcher passionate about building intelligent systems 
        that solve real-world problems. With expertise in machine learning, deep learning, and natural language 
        processing, Talha has worked on various projects ranging from computer vision to LLM-based applications.
        
        **Research Interests:**
        - Large Language Models (LLMs)
        - Retrieval-Augmented Generation (RAG)
        - Computer Vision
        - MLOps and Model Deployment
        
        **Mission:**
        To democratize AI career guidance and help aspiring data scientists find their perfect career path.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    #### 🎓 Affiliation
    
    **Department of Computer Science & Engineering**  
    **Rajshahi University of Engineering & Technology (RUET)**
    
    ---
    
    #### 🏆 Platform Features
    
    - **Intelligent CV Analysis:** Leveraging RAG and vector search for accurate role matching
    - **Real-time JD Matching:** Compare your CV against any job description
    - **Career Readiness Score:** Data-driven assessment of your AI/ML career readiness
    - **Personalized Recommendations:** Tailored advice based on your unique profile
    - **Market Insights:** Up-to-date salary ranges and demand for AI roles in Bangladesh
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# CONTACT PAGE
# ============================================================
def render_contact():
    """Contact page with contact information"""
    st.markdown("### 📞 Contact Us")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📱 Get in Touch")
        
        st.markdown("""
        **📞 Phone**  
        +880 1721 577792
        
        **📧 Email**  
        jobayertalha2020@gmail.com
        
        **📍 Location**  
        Dhaka, Bangladesh
        
        **🕒 Business Hours**  
        Sunday - Thursday: 9:00 AM - 6:00 PM
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🌐 Connect With Me")
        
        st.markdown("""
        **Professional Network:**
        
        - 🔗 **LinkedIn:** [linkedin.com/in/talha-jobayer](https://linkedin.com/in/talha-jobayer)
        - 💻 **GitHub:** [github.com/talha-jobayer](https://github.com/talha-jobayer)
        - 🐦 **Twitter/X:** [@talha_jobayer](https://twitter.com/talha_jobayer)
        - 📝 **Medium:** [medium.com/@talha-jobayer](https://medium.com/@talha-jobayer)
        
        ---
        
        #### 📧 Send a Message
        
        For collaborations, inquiries, or feedback, feel free to reach out via email or connect on LinkedIn.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Contact form
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📝 Quick Message")
    
    with st.form("contact_form"):
        name = st.text_input("Your Name", value=st.session_state.candidate_name)
        email = st.text_input("Your Email")
        message = st.text_area("Message", height=100)
        
        if st.form_submit_button("Send Message", type="primary", use_container_width=True):
            if name and email and message:
                st.success("✅ Message sent! We'll get back to you soon.")
                # In production, add email sending logic here
            else:
                st.error("Please fill in all fields")
    
    st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    """Main application entry point"""
    
    # Show name entry first
    if not st.session_state.name_entered:
        render_home()
        return
    
    # Render navigation
    render_navbar()
    
    # Page routing
    pages = {
        "home": render_home,
        "analyze": render_analyze,
        "matching": render_matching,
        "quiz": render_quiz,
        "about": render_about,
        "contact": render_contact,
    }
    
    current_page = st.session_state.page
    if current_page in pages:
        pages[current_page]()
    else:
        render_home()


if __name__ == "__main__":
    main()
