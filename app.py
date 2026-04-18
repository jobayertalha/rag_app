"""
app.py — CV Analyzer with Navigation Bar & Fixed Matching
"""

import streamlit as st
import tempfile
import os
import re

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
.match-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    margin-left: 0.5rem;
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
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE INIT
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
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
        uploaded_cv = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")
        if uploaded_cv:
            st.success(f"✅ {uploaded_cv.name}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card"><div class="card-title">📝 Job Description (Optional)</div>', unsafe_allow_html=True)
        jd_input = st.text_area("Paste JD here", height=150, placeholder="Paste a job description to bias the match toward a specific role...", label_visibility="collapsed")
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
                    
                    from agent import extract_cv_text
                    cv_text = extract_cv_text(tmp_path)
                    os.unlink(tmp_path)
                
                st.session_state.cv_text = cv_text
                st.session_state.jd_text = jd_input.strip()
                st.session_state.messages = []
                
                with st.spinner("🔍 Matching with FAISS..."):
                    from rag import retrieve_context
                    st.session_state.retrieved = retrieve_context(cv_text, jd_input.strip(), k=5)
                
                with st.spinner("🤖 Generating AI analysis..."):
                    from agent import build_agent
                    st.session_state.agent = build_agent(cv_text, jd_input.strip(), st.session_state.candidate_name)
                    
                    from agent import run_agent
                    raw = run_agent(st.session_state.agent,
                        "Analyse this candidate's CV and give a full career match. "
                        "Follow EXACTLY these tags:\n"
                        "TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
                        "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY\n"
                        "Be specific — reference actual CV skills throughout."
                    )
                    st.session_state.analysis_raw = raw
                
                st.session_state.page = "analyze"
                st.success("✅ Analysis complete!")
                st.rerun()
            else:
                st.warning("⚠️ Please upload your CV first")
    
    # Features section
    st.markdown("""
    <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 2rem;'>
        <div class='card' style='text-align: center;'>
            <div style='font-size: 2rem;'>🎯</div>
            <div style='font-weight: 600; color: #cbd5e1;'>Smart Matching</div>
            <div style='font-size: 0.75rem; color: #64748b;'>FAISS vector search vs real JDs</div>
        </div>
        <div class='card' style='text-align: center;'>
            <div style='font-size: 2rem;'>🤖</div>
            <div style='font-weight: 600; color: #cbd5e1;'>AI Analysis</div>
            <div style='font-size: 0.75rem; color: #64748b;'>Personalized recommendations</div>
        </div>
        <div class='card' style='text-align: center;'>
            <div style='font-size: 2rem;'>💬</div>
            <div style='font-weight: 600; color: #cbd5e1;'>Interactive Chat</div>
            <div style='font-size: 0.75rem; color: #64748b;'>Ask follow-up questions</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_analysis():
    """Display analysis results with score card."""
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
    
    # ========== SCORE CARD (New Feature) ==========
    if readiness:
        render_score_card(
            readiness.get("total_score", 0),
            readiness.get("level", "Not Ready"),
            readiness.get("recommendation", "")
        )
        
        # Show breakdown
        if readiness.get("breakdown"):
            render_breakdown_chart(readiness["breakdown"])
    
    st.markdown("---")
    
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
    
    # All matches grid
    if all_matches:
        st.markdown("### 📊 Role Breakdown")
        cols = st.columns(min(len(all_matches), 4))
        for i, role in enumerate(all_matches[:4]):
            with cols[i]:
                skills = role.get("skills", [])[:4]
                skills_html = "".join(f"<span class='skill-chip'>{s}</span>" for s in skills)
                
                # Color based on match percentage
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
    
    # Rest of your existing analysis display...
    # (skill gaps, resume tips, career path, runner up)
    
    # ... keep your existing code for these sections ...
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
        "What is my career path?",
        "Find ML Engineer jobs",
        "What's my market value?"
    ]
    for i, q in enumerate(questions):
        with q_cols[i % 3]:
            if st.button(q, key=f"q_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        from agent import run_agent
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
                from agent import run_agent
                resp = run_agent(st.session_state.agent, prompt)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
        st.rerun()

def render_score_card(score: float, level: str, recommendation: str):
    """Render a beautiful score card with range-based feedback."""
    
    # Determine color and icon based on score
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
    
    # Add score range explanation
    st.markdown(f"""
    <div style='display: flex; gap: 1rem; justify-content: space-between; margin-top: 0.5rem; font-size: 0.7rem;'>
        <div style='flex: 1; text-align: center; padding: 0.5rem; background: #0f0f20; border-radius: 10px; border-left: 3px solid #ef4444;'>
            <span style='color: #ef4444;'>0-30%</span>
            <span style='color: #64748b;'> | Consider other roles first</span>
        </div>
        <div style='flex: 1; text-align: center; padding: 0.5rem; background: #0f0f20; border-radius: 10px; border-left: 3px solid #f59e0b;'>
            <span style='color: #f59e0b;'>30-60%</span>
            <span style='color: #64748b;'> | Build more skills</span>
        </div>
        <div style='flex: 1; text-align: center; padding: 0.5rem; background: #0f0f20; border-radius: 10px; border-left: 3px solid #10b981;'>
            <span style='color: #10b981;'>60-75%</span>
            <span style='color: #64748b;'> | Almost ready</span>
        </div>
        <div style='flex: 1; text-align: center; padding: 0.5rem; background: #0f0f20; border-radius: 10px; border-left: 3px solid #06b6d4;'>
            <span style='color: #06b6d4;'>75-100%</span>
            <span style='color: #64748b;'> | Ready to apply!</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_breakdown_chart(breakdown: dict):
    """Render score breakdown as a simple bar chart."""
    st.markdown("### 📊 Score Breakdown")
    
    categories = [
        ("💼 Experience", breakdown.get("experience", 0), 30),
        ("📁 Projects", breakdown.get("projects", 0), 25),
        ("📜 Certificates", breakdown.get("certificates", 0), 20),
        ("🔧 Skills", breakdown.get("skills", 0), 25),
    ]
    
    for label, score, max_score in categories:
        percent = (score / max_score) * 100 if max_score > 0 else 0
        bar_width = percent
        st.markdown(f"""
        <div style='margin-bottom: 0.8rem;'>
            <div style='display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8;'>
                <span>{label}</span>
                <span>{score}/{max_score}</span>
            </div>
            <div style='background: #1a1a30; border-radius: 10px; height: 8px; overflow: hidden;'>
                <div style='width: {bar_width}%; background: linear-gradient(90deg, #a855f7, #ec4899); height: 100%; border-radius: 10px;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN APP
# ============================================================
def main():
    # Name entry screen
    # ============================================================
# NAME ENTRY SCREEN - PRESS ENTER TO CONTINUE
# Wrap in a form to capture Enter key
with st.form(key="name_form"):
    name_val = st.text_input("name", placeholder="e.g. Talha Jobayer", label_visibility="collapsed", key="name_field")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Get Started →", use_container_width=True, type="primary")
    
    if submitted and name_val.strip():
        st.session_state.candidate_name = name_val.strip()
        st.session_state.name_entered = True
        st.rerun()
    elif submitted:
        st.warning("Please enter your name")
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
