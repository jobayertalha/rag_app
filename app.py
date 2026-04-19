"""
app.py — CV Analyzer with Clean Single Navigation Bar
"""

import streamlit as st
import tempfile
import os
import re

from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context, load_roles

st.set_page_config(
    page_title="CV Analyzer | AI Career Match",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS - Clean minimal design
# ============================================================
st.markdown("""
<style>
/* Hide Streamlit default elements */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}
footer {visibility: hidden;}
.stApp { background: linear-gradient(135deg, #0a0a14 0%, #0f0f20 100%); }

* { font-family: 'DM Sans', sans-serif; }

/* Navigation - Single clean bar */
.single-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 15, 32, 0.95);
    border-bottom: 1px solid #2d2d5a;
    padding: 0.5rem 2rem;
    margin-bottom: 1.5rem;
}
.nav-left { display: flex; gap: 0.25rem; align-items: center; flex-wrap: wrap; }
.nav-right { position: relative; }
.nav-item {
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
.nav-item:hover { background: rgba(168, 85, 247, 0.1); color: #a855f7; }
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.user-btn {
    background: transparent;
    border: 1px solid #2d2d5a;
    color: #94a3b8;
    padding: 0.3rem 1rem;
    border-radius: 20px;
    font-size: 0.8rem;
    cursor: pointer;
}
.user-btn:hover { background: rgba(168, 85, 247, 0.1); border-color: #a855f7; }

/* Dropdown menu */
.dropdown-menu {
    position: absolute;
    top: 40px;
    right: 0;
    background: #0f0f20;
    border: 1px solid #2d2d5a;
    border-radius: 12px;
    padding: 0.5rem;
    min-width: 160px;
    z-index: 1000;
}
.dropdown-text { padding: 0.3rem 0.8rem; font-size: 0.7rem; color: #64748b; }
.dropdown-name { padding: 0.3rem 0.8rem; font-size: 0.8rem; font-weight: 600; color: #cbd5e1; }
.dropdown-divider { height: 1px; background: #2d2d5a; margin: 0.3rem 0; }
.dropdown-signout { padding: 0.5rem 0.8rem; font-size: 0.8rem; color: #ef4444; cursor: pointer; border-radius: 8px; }
.dropdown-signout:hover { background: rgba(239, 68, 68, 0.1); }

/* Cards */
.card { background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 16px; padding: 1.2rem; margin-bottom: 1rem; }
.skill-chip { display: inline-block; background: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); color: #a5b4fc; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 6px; margin: 0.2rem; }
.gap-chip { background: rgba(239, 68, 68, 0.15); border-color: rgba(239, 68, 68, 0.3); color: #fca5a5; }
.add-chip { background: rgba(34, 197, 94, 0.15); border-color: rgba(34, 197, 94, 0.3); color: #86efac; }
.hero-match { font-size: 2.5rem; font-weight: 800; background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }

/* Mode selector */
.mode-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin: 1.5rem 0; }
.mode-card { background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 20px; padding: 1.5rem; text-align: center; transition: all 0.3s; }
.mode-card:hover { border-color: #7c3aed; transform: translateY(-2px); }
.mode-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.mode-title { font-size: 1.1rem; font-weight: 700; color: #f1f5f9; }
.mode-desc { font-size: 0.7rem; color: #64748b; margin-top: 0.3rem; }

/* JD Match */
.jd-match-card { background: linear-gradient(135deg, #0f0f20 0%, #1a0f35 100%); border: 1px solid #2d2060; border-radius: 20px; padding: 1.5rem; text-align: center; margin-bottom: 1rem; }
.jd-match-score { font-size: 3rem; font-weight: 800; }

/* Welcome */
.welcome-card { background: #0f0f20; border: 1px solid #1e1e3a; border-radius: 24px; padding: 2rem; text-align: center; max-width: 400px; margin: 2rem auto; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
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
    }


def match_companies(cv_text: str, _) -> list:
    cv_lower = cv_text.lower()
    companies = []
    try:
        roles = load_roles()
        for role in roles:
            if role.get("company"):
                skills = role.get("skills", [])
                found = sum(1 for s in skills if s.lower() in cv_lower)
                score = (found / max(len(skills), 1)) * 100
                salary = role.get("salary", {})
                salary_str = f"৳{salary['junior']}" if salary.get("junior") and salary['junior'] != "0" else ""
                companies.append({
                    "name": role["company"],
                    "role": role.get("title", role.get("role")),
                    "match_score": round(score, 1),
                    "salary_range": salary_str,
                    "location": role.get("location", "Dhaka"),
                })
        companies.sort(key=lambda x: x["match_score"], reverse=True)
        return companies[:6]
    except:
        return []


def calculate_jd_match_score(cv_text: str, jd_text: str) -> dict:
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    
    keywords = set()
    for skill in ['python', 'sql', 'tensorflow', 'pytorch', 'langchain', 'rag', 'llm', 'nlp', 'docker', 'kubernetes', 'aws', 'gcp', 'azure', 'pandas', 'numpy', 'scikit-learn', 'git', 'linux']:
        if skill in jd_lower:
            keywords.add(skill)
    
    found = [kw for kw in keywords if kw in cv_lower]
    missing = [kw for kw in keywords if kw not in cv_lower]
    match_pct = int((len(found) / max(len(keywords), 1)) * 100)
    
    return {
        "match_pct": min(95, match_pct),
        "matched_keywords": found[:15],
        "missing_keywords": missing[:15],
        "found_count": len(found),
        "total_keywords": len(keywords),
    }


def render_navbar():
    """Single navigation bar - NO DUPLICATES"""
    name = st.session_state.candidate_name
    first_name = name.split()[0] if name else "Guest"
    
    # Using HTML for clean nav bar
    st.markdown(f'''
    <div class="single-nav">
        <div class="nav-left">
            <button class="nav-item nav-logo" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'home'}}, '*')">📄 CV Analyzer</button>
            <button class="nav-item" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'home'}}, '*')">🏠 Home</button>
            <button class="nav-item" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'analyze'}}, '*')">📊 Analysis</button>
            <button class="nav-item" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'chat'}}, '*')">💬 Chat</button>
            <button class="nav-item" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'career'}}, '*')">🎯 Career Rec</button>
        </div>
        <div class="nav-right">
            <button class="user-btn" id="userBtn" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'toggle_dropdown'}}, '*')">👋 {first_name} ▼</button>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Handle button clicks via Streamlit
    col1, col2, col3, col4, col5, col6 = st.columns([1.2, 0.8, 0.8, 0.8, 0.8, 0.8])
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
            else:
                st.warning("Please analyze your CV first on Home page")
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
        if st.button(f"👋 {first_name}", key="nav_user", use_container_width=True):
            st.session_state.show_dropdown = not st.session_state.show_dropdown
            st.rerun()
    
    # Dropdown menu
    if st.session_state.show_dropdown:
        st.markdown(f'''
        <div class="dropdown-menu">
            <div class="dropdown-text">Signed in as</div>
            <div class="dropdown-name">{name}</div>
            <div class="dropdown-divider"></div>
            <div class="dropdown-signout" onclick="parent.postMessage({{type: 'streamlit:setComponentValue', value: 'signout'}}, '*')">🚪 Sign Out / Change Name</div>
        </div>
        ''', unsafe_allow_html=True)
        
        if st.button("Confirm Sign Out", key="signout_confirm"):
            st.session_state.name_entered = False
            st.session_state.cv_text = None
            st.session_state.analysis_raw = None
            st.session_state.retrieved = None
            st.session_state.agent = None
            st.session_state.messages = []
            st.session_state.show_dropdown = False
            st.rerun()
    
    st.markdown("<hr style='margin: 0.5rem 0; border-color: #1a1a30;'>", unsafe_allow_html=True)


def render_welcome():
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <div style='font-family: Syne, sans-serif; font-size: 3rem; font-weight: 800;'>CV <span style='background: linear-gradient(135deg, #a855f7, #ec4899); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>Analyzer</span></div>
        <p style='color: #64748b;'>AI-powered career matching for data & AI roles</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 1rem;">👋 Welcome!</div>', unsafe_allow_html=True)
        with st.form(key="name_form"):
            name = st.text_input("Your name", placeholder="Talha Jobayer", label_visibility="collapsed")
            if st.form_submit_button("✨ Start Analysis →", use_container_width=True, type="primary"):
                if name and name.strip():
                    st.session_state.candidate_name = name.strip()
                    st.session_state.name_entered = True
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


def render_home():
    name = st.session_state.candidate_name
    first_name = name.split()[0]
    
    st.markdown(f"<h2 style='text-align: center;'>Hello, {first_name}! 👋</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #64748b;'>Ready to analyze your CV?</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>📄</div>
            <div class='mode-title'>Analyze My CV</div>
            <div class='mode-desc'>Get matched with AI/ML roles from our knowledge base</div>
            <div class='mode-desc' style='color: #a855f7;'>Skills | Roles | Salary | Career Path</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📄 Analyze My CV", key="home_cv", use_container_width=True):
            st.session_state.analysis_submode = "cv_analysis"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>🎯</div>
            <div class='mode-title'>Match with Job Description</div>
            <div class='mode-desc'>Paste a JD and see how well your CV matches</div>
            <div class='mode-desc' style='color: #10b981;'>Match % | Missing Skills | Companies</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Match with JD", key="home_jd", use_container_width=True):
            st.session_state.analysis_submode = "jd_match"
            st.rerun()
    
    if st.session_state.analysis_submode == "cv_analysis":
        st.markdown("---")
        uploaded = st.file_uploader("Upload your CV (PDF)", type=["pdf"], key="cv_upload")
        if uploaded and st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            process_cv_analysis(uploaded)
    
    elif st.session_state.analysis_submode == "jd_match":
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            uploaded = st.file_uploader("Your CV (PDF)", type=["pdf"], key="jd_cv_upload")
        with col_b:
            jd_text = st.text_area("Job Description", height=150, placeholder="Paste the job description here...")
        if uploaded and jd_text and st.button("🎯 Calculate Match", type="primary", use_container_width=True):
            process_jd_match(uploaded, jd_text)


def process_cv_analysis(uploaded):
    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            cv_text = extract_cv_text(tmp.name)
            os.unlink(tmp.name)
        
        st.session_state.cv_text = cv_text
        st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
        st.session_state.matched_companies = match_companies(cv_text, [])
        st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
        raw = run_agent(st.session_state.agent, "Analyse this CV. Follow tags: TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP")
        st.session_state.analysis_raw = raw
    
    st.session_state.page = "analyze"
    st.rerun()


def process_jd_match(uploaded, jd_text):
    with st.spinner("Processing..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            cv_text = extract_cv_text(tmp.name)
            os.unlink(tmp.name)
        
        st.session_state.cv_text = cv_text
        st.session_state.jd_match_result = calculate_jd_match_score(cv_text, jd_text)
        st.session_state.matched_companies = match_companies(cv_text, [])
    
    st.session_state.page = "jd_result"
    st.rerun()


def render_jd_result():
    r = st.session_state.jd_match_result
    if not r:
        st.warning("No result")
        return
    
    pct = r["match_pct"]
    color = "#ef4444" if pct < 30 else "#f59e0b" if pct < 60 else "#10b981" if pct < 80 else "#06b6d4"
    status = ["Low Match", "Partial Match", "Good Match", "Excellent Match!"][[30,60,80,100].index(min([x for x in [30,60,80,100] if x > pct]))]
    
    st.markdown(f"""
    <div class='jd-match-card'>
        <div style='color:{color};'>JD Match Score</div>
        <div class='jd-match-score' style='color:{color};'>{pct}%</div>
        <div style='color:#f1f5f9;'>{status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Keywords Found", f"{r['found_count']}/{r['total_keywords']}")
    with col2:
        if st.session_state.matched_companies:
            st.markdown("**🏢 Companies to target**")
            for c in st.session_state.matched_companies[:3]:
                st.markdown(f"- {c['name']} ({c['match_score']}%)")
    
    if r.get("missing_keywords"):
        st.markdown("**❌ Add these to your CV**")
        st.markdown(" ".join(f"<span class='skill-chip gap-chip'>{kw}</span>" for kw in r["missing_keywords"][:8]), unsafe_allow_html=True)
    
    if st.button("← New Match", use_container_width=True):
        st.session_state.page = "home"
        st.session_state.analysis_submode = "jd_match"
        st.rerun()


def render_analysis():
    if not st.session_state.cv_text:
        st.warning("Please analyze your CV first")
        return
    
    r = st.session_state.retrieved or {}
    matches = r.get("all_matches", [])
    top = r.get("top_match", {})
    readiness = r.get("readiness", {})
    parsed = parse_analysis(st.session_state.analysis_raw or "") if st.session_state.analysis_raw else {}
    
    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🏢 Companies", "💰 Salary", "🔧 Skills", "🗺️ Path"])
    
    with t1:
        st.markdown(f"<div class='hero-match'>{parsed.get('match_pct', top.get('match_pct', 0))}% Match</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{parsed.get('top_role', top.get('title', 'AI Professional'))}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:#94a3b8;'>{parsed.get('why_right', 'Great alignment!')}</p>", unsafe_allow_html=True)
        score = readiness.get("total_score", 0)
        if score < 30:
            st.warning(f"🔴 Readiness: {score}% - Build fundamentals")
        elif score < 60:
            st.warning(f"🟡 Readiness: {score}% - Keep going")
        else:
            st.success(f"🟢 Readiness: {score}% - Ready to apply!")
    
    with t2:
        for c in st.session_state.matched_companies[:5]:
            st.markdown(f"**{c['name']}** - {c['role']} ({c['match_score']}%)\n📍 {c['location']}")
            if c.get("salary_range"):
                st.caption(f"💰 {c['salary_range']}/month")
    
    with t3:
        for role in matches[:4]:
            if role.get("salary_min"):
                st.metric(role.get("title"), f"৳{role['salary_min']:,} - ৳{role['salary_max']:,}")
    
    with t4:
        gaps = parsed.get("skill_gaps") or r.get("skill_gaps", [])
        for g in gaps[:6]:
            st.markdown(f"<span class='skill-chip gap-chip'>{g}</span>", unsafe_allow_html=True)
        recs = parsed.get("resume_add") or r.get("resume_skills", [])
        for sk in recs[:6]:
            st.markdown(f"<span class='skill-chip add-chip'>+ {sk}</span>", unsafe_allow_html=True)
    
    with t5:
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
    
    r = st.session_state.retrieved or {}
    score = r.get("readiness", {}).get("total_score", 0)
    
    if score < 30:
        st.warning("🔴 **Focus on Fundamentals**\n\n- Complete Python basics\n- Take ML courses\n- Build 2-3 projects")
    elif score < 60:
        st.warning("🟡 **Building Momentum**\n\n- Take advanced courses\n- Build portfolio\n- Get certifications")
    else:
        st.success("🟢 **Ready for Job Search!**\n\n- Update LinkedIn\n- Apply to matching companies\n- Prepare for interviews")


# ============================================================
# MAIN
# ============================================================
def main():
    if not st.session_state.name_entered:
        render_welcome()
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
