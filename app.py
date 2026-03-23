"""
app.py — AI Career Match
Dark theme UI matching the career match screenshots.

Flow:
  1. Welcome screen: ask for name only
  2. Main: upload CV + optional JD → run analysis
  3. Results: hero match card, full breakdown, why right for you,
     next steps, salary, market demand, skill gaps, resume tips,
     career path, runner-up
  4. Chat: ask follow-up questions

Pipeline: User CV + JD → Embedding → FAISS → Retrieve Roles/Skills/Paths → LLM → Advice
"""

import streamlit as st
import tempfile
import os
import re

st.set_page_config(
    page_title="AI Career Match",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Dark Theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a14;
    color: #e2e8f0;
}
.stApp { background-color: #0a0a14; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0f0f20 !important;
    border-right: 1px solid #1a1a30;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] .sidebar-brand {
    color: #a78bfa !important;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
}

div[data-testid="stSidebar"] .stButton > button {
    background: #13132a;
    color: #a5b4fc !important;
    border: 1px solid #1e1e3a;
    border-radius: 8px;
    width: 100%;
    text-align: left;
    padding: 0.45rem 0.75rem;
    font-size: 0.78rem;
    margin-bottom: 4px;
    transition: all 0.15s;
    font-family: 'DM Sans', sans-serif;
}
div[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e1e3a;
    border-color: #4c4c8a;
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7c3aed 0%, #db2777 100%) !important;
    color: #fff !important;
    border: none;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 0.6rem 0.75rem;
    border-radius: 10px;
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    opacity: 0.92;
}

/* Main area */
section[data-testid="stMain"] { background-color: #0a0a14; }
.block-container { padding-top: 1.5rem; max-width: 960px; }

/* Welcome screen */
.welcome-wrap {
    min-height: 80vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2rem;
}
.welcome-logo {
    font-size: 3.5rem;
    margin-bottom: 1.2rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-8px); }
}
.welcome-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
    line-height: 1.15;
}
.welcome-title span {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.welcome-sub {
    font-size: 1rem;
    color: #64748b;
    max-width: 420px;
    line-height: 1.7;
    margin-bottom: 2.5rem;
}
.pipeline-mini {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.68rem;
    color: #334155;
    margin-bottom: 2.5rem;
    flex-wrap: wrap;
    justify-content: center;
}
.pm-step {
    background: #13132a;
    border: 1px solid #1e1e3a;
    border-radius: 4px;
    padding: 2px 8px;
    color: #4c4c8a;
}
.pm-arrow { color: #1e1e3a; }

/* Name input area */
.name-input-card {
    background: #13132a;
    border: 1px solid #1e1e3a;
    border-radius: 18px;
    padding: 2rem 2.5rem;
    max-width: 460px;
    width: 100%;
    margin: 0 auto;
}
.name-input-label {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 0.75rem;
}

/* Pipeline strip */
.pipeline {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.68rem;
    color: #334155;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
}
.ps {
    background: #13132a;
    border: 1px solid #1e1e3a;
    border-radius: 4px;
    padding: 2px 8px;
    color: #4c4c8a;
}
.ps.on { border-color: #4ade80; color: #4ade80; }
.pa { color: #1e1e3a; }

/* Name header */
.name-header {
    text-align: center;
    margin-bottom: 1.5rem;
}
.name-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.9rem;
    font-weight: 800;
    color: #f1f5f9;
    margin-bottom: 0.3rem;
}
.name-header p {
    font-size: 0.875rem;
    color: #64748b;
}

/* Hero card */
.hero-card {
    background: linear-gradient(140deg, #13132a 0%, #1a0f35 60%, #160f2a 100%);
    border: 1px solid #2d2060;
    border-radius: 22px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(168,85,247,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: #fff;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 4px 14px;
    border-radius: 20px;
    margin-bottom: 1rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.hero-emoji { font-size: 2.4rem; margin-bottom: 0.4rem; }
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    margin-bottom: 0.2rem;
}
.hero-match {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem;
    line-height: 1.1;
}
.hero-desc {
    font-size: 0.9rem;
    color: #94a3b8;
    max-width: 520px;
    margin: 0 auto 1.2rem;
    line-height: 1.7;
}
.skill-chip {
    display: inline-block;
    font-size: 0.7rem;
    background: rgba(99, 102, 241, 0.12);
    color: #a5b4fc;
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 6px;
    padding: 3px 10px;
    margin: 3px;
}

/* Section header */
.section-header {
    font-size: 0.72rem;
    font-weight: 700;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 1.5rem 0 0.75rem;
    font-family: 'DM Sans', sans-serif;
}

/* Breakdown cards */
.breakdown-card {
    background: #0f0f20;
    border: 1px solid #1a1a30;
    border-radius: 14px;
    padding: 1.1rem 1rem;
    height: 100%;
}
.breakdown-card-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.6rem;
}
.breakdown-emoji { font-size: 1rem; }
.breakdown-role {
    font-size: 0.78rem;
    font-weight: 600;
    color: #94a3b8;
    line-height: 1.3;
}
.breakdown-pct {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.5rem;
    line-height: 1;
}
.breakdown-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.breakdown-chip {
    font-size: 0.62rem;
    background: #13132a;
    color: #64748b;
    border-radius: 4px;
    padding: 2px 6px;
    border: 1px solid #1a1a30;
}

/* Detail card */
.detail-card {
    background: #0f0f20;
    border: 1px solid #1a1a30;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.detail-card-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.detail-card-desc {
    font-size: 0.875rem;
    color: #94a3b8;
    line-height: 1.7;
    margin-bottom: 1rem;
}

/* Info boxes */
.info-box {
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
}
.info-box-label {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Sans', sans-serif;
}
.info-box-content {
    font-size: 0.85rem;
    line-height: 1.65;
    color: #cbd5e1;
}
.info-box.next-steps { background: #0d1a2e; border: 1px solid #1a3a5c; }
.info-box.next-steps .info-box-label { color: #60a5fa; }
.info-box.salary { background: #0a1a0e; border: 1px solid #14532d; }
.info-box.salary .info-box-label { color: #4ade80; }
.info-box.demand { background: #0d0d20; border: 1px solid #1e1e4a; }
.info-box.demand .info-box-label { color: #818cf8; }
.info-box.resume { background: #0a1a0e; border: 1px solid #14532d; }
.info-box.resume .info-box-label { color: #86efac; }
.info-box.gaps { background: #1a0a0a; border: 1px solid #4c1010; }
.info-box.gaps .info-box-label { color: #fca5a5; }

/* Salary grid */
.salary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 8px;
    margin-top: 6px;
}
.salary-item {
    background: #060e07;
    border-radius: 8px;
    padding: 0.6rem 0.7rem;
    border: 1px solid #0f2a14;
}
.salary-level { font-size: 0.62rem; color: #4ade80; font-weight: 700; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.08em; }
.salary-range { font-size: 0.78rem; color: #bbf7d0; font-weight: 600; }

/* Gap + resume badges */
.gap-badge {
    display: inline-block;
    font-size: 0.68rem;
    background: #1a0808;
    color: #fca5a5;
    border: 1px solid #4c1010;
    border-radius: 6px;
    padding: 3px 10px;
    margin: 2px;
}
.add-badge {
    display: inline-block;
    font-size: 0.68rem;
    background: #071a09;
    color: #86efac;
    border: 1px solid #0f4018;
    border-radius: 6px;
    padding: 3px 10px;
    margin: 2px;
}

/* Career path dot */
.cp-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    margin-top: 5px;
    flex-shrink: 0;
}

/* Runner-up */
.runner-up {
    background: #0d0d1a;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #94a3b8;
    line-height: 1.7;
    margin-top: 0.75rem;
}
.runner-up strong { color: #fbbf24; font-family: 'Syne', sans-serif; }

/* Empty state */
.empty-state {
    text-align: center;
    padding: 5rem 2rem;
    color: #334155;
}
.empty-state h3 {
    font-family: 'Syne', sans-serif;
    color: #475569;
    font-size: 1.3rem;
    margin-bottom: 0.5rem;
}
.empty-state p { font-size: 0.875rem; line-height: 1.8; }

/* Chat */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] input {
    background: #0f0f20 !important;
    border: 1px solid #1e1e3a !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.stChatMessage { background: transparent !important; }

/* Streamlit overrides */
.stTextInput > div > div > input {
    background: #13132a !important;
    border: 1px solid #1e1e3a !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.12) !important;
}
.stTextArea > div > div > textarea {
    background: #0f0f20 !important;
    border: 1px solid #1a1a30 !important;
    color: #cbd5e1 !important;
    border-radius: 10px !important;
}
.stFileUploader {
    background: #0f0f20 !important;
    border: 1px dashed #1e1e3a !important;
    border-radius: 10px !important;
}
.stSpinner > div { border-top-color: #7c3aed !important; }

/* Back to profile button */
.back-btn-wrap {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────
for k, v in {
    "messages": [],
    "agent": None,
    "cv_text": None,
    "jd_text": "",
    "show_results": False,
    "analysis_raw": None,
    "retrieved": None,
    "prefill": None,
    "candidate_name": "",
    "name_entered": False,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Parse LLM analysis output ─────────────────────────────────────
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
        "top_role":      get("TOP_ROLE"),
        "match_pct":     get("MATCH_PCT"),
        "why_right":     get("WHY_RIGHT"),
        "next_steps":    get_list("NEXT_STEPS"),
        "skill_gaps":    get_list("SKILL_GAPS"),
        "resume_add":    get_list("RESUME_ADD"),
        "career_path":   get_list("CAREER_PATH"),
        "runner_up":     get("RUNNER_UP"),
        "runner_up_why": get("RUNNER_UP_WHY"),
    }


# ══════════════════════════════════════════════════════════════════
# SCREEN 1: WELCOME — Ask for name
# ══════════════════════════════════════════════════════════════════
if not st.session_state.name_entered:
    st.markdown("""
    <div class='welcome-wrap'>
        <div class='welcome-logo'>⚡</div>
        <div class='welcome-title'>AI <span>Career Match</span></div>
        <div class='welcome-sub'>
            Upload your CV, get matched to your ideal AI/data role.<br>
            Powered by FAISS vector search + LLM analysis.
        </div>
        <div class='pipeline-mini'>
            <span class='pm-step'>CV + JD</span>
            <span class='pm-arrow'>→</span>
            <span class='pm-step'>Embedding</span>
            <span class='pm-arrow'>→</span>
            <span class='pm-step'>FAISS</span>
            <span class='pm-arrow'>→</span>
            <span class='pm-step'>Retrieve</span>
            <span class='pm-arrow'>→</span>
            <span class='pm-step'>LLM</span>
            <span class='pm-arrow'>→</span>
            <span class='pm-step'>Advice</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Center the name input card
    col_l, col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.markdown("<div class='name-input-card'>", unsafe_allow_html=True)
        st.markdown("<div class='name-input-label'>👋 What's your name?</div>", unsafe_allow_html=True)
        name_val = st.text_input(
            "name",
            placeholder="e.g. Syed Khan",
            label_visibility="collapsed",
            key="name_field"
        )
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        if st.button("Get Started →", use_container_width=True, type="primary"):
            st.session_state.candidate_name = name_val.strip() if name_val.strip() else "there"
            st.session_state.name_entered = True
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.stop()


# ══════════════════════════════════════════════════════════════════
# SIDEBAR — shown after name entry
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "there"

    st.markdown(
        f"<div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;"
        f"color:#a855f7;margin-bottom:0.2rem;'>⚡ AI Career Match</div>"
        f"<div style='font-size:0.78rem;color:#475569;margin-bottom:1.2rem;'>Hi, {first}! 👋</div>",
        unsafe_allow_html=True
    )

    st.markdown("<div style='font-size:0.72rem;color:#475569;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;'>Quick Questions</div>", unsafe_allow_html=True)

    for action in [
        "What roles am I best suited for?",
        "What skills am I missing?",
        "What should I add to my resume?",
        "What is my career path from here?",
        "Find ML Engineer jobs in Pakistan",
        "Search for remote Data Science jobs",
        "Find AI Engineer jobs in United States",
    ]:
        if st.button(action, key=f"qa_{action}"):
            st.session_state.prefill = action

    st.markdown("---")
    st.markdown("<div style='font-size:0.72rem;color:#475569;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem;'>Upload CV</div>", unsafe_allow_html=True)
    uploaded_cv = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")

    st.markdown("<div style='font-size:0.72rem;color:#475569;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;margin:0.75rem 0 0.3rem;'>Job Description <span style='color:#334155;font-weight:400;text-transform:none;'>(optional)</span></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#334155;margin-bottom:6px;'>Paste a JD to bias FAISS toward that role.</div>", unsafe_allow_html=True)
    jd_input = st.text_area(
        "JD",
        height=100,
        placeholder="Paste JD from LinkedIn, Indeed, Glassdoor...",
        label_visibility="collapsed"
    )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    analyse_btn = st.button("⚡ Get Career Match", use_container_width=True, type="primary")

    # ── Run analysis ──────────────────────────────────────────────
    if uploaded_cv and analyse_btn:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_cv.read())
            tmp_path = tmp.name

        with st.spinner("Reading CV..."):
            from agent import extract_cv_text
            cv_text = extract_cv_text(tmp_path)
            os.unlink(tmp_path)

        st.session_state.cv_text  = cv_text
        st.session_state.jd_text  = jd_input.strip()
        st.session_state.messages = []
        st.session_state.show_results = False

        with st.spinner("CV + JD → Embedding → FAISS vector search..."):
            from rag import retrieve_context
            st.session_state.retrieved = retrieve_context(cv_text, jd_input.strip(), k=5)

        with st.spinner("Building AI agent..."):
            from agent import build_agent
            st.session_state.agent = build_agent(
                cv_text, jd_input.strip(), st.session_state.candidate_name
            )

        with st.spinner("LLM generating your career match..."):
            from agent import run_agent
            analysis_prompt = (
                "Analyse this candidate's CV and give a full career match. "
                "Follow the EXACT format with these tags:\n"
                "TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
                "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY\n"
                "Be highly specific — reference actual CV skills, tools, and experience throughout."
            )
            raw = run_agent(st.session_state.agent, analysis_prompt)
            st.session_state.analysis_raw = raw
            st.session_state.show_results = True
            st.rerun()

    elif not uploaded_cv and analyse_btn:
        st.warning("Please upload a CV PDF first.")

    if not uploaded_cv:
        st.markdown(
            "<div style='font-size:0.72rem;color:#334155;text-align:center;margin-top:0.5rem;'>Upload your CV PDF to get started</div>",
            unsafe_allow_html=True
        )

    # Back to name screen
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    if st.button("← Change Name", key="back_name"):
        st.session_state.name_entered = False
        st.session_state.cv_text = None
        st.session_state.show_results = False
        st.rerun()


# ══════════════════════════════════════════════════════════════════
# MAIN AREA
# ══════════════════════════════════════════════════════════════════
if not st.session_state.cv_text:
    name = st.session_state.candidate_name
    first = name.split()[0].capitalize() if name else "there"
    st.markdown(f"""
    <div class='empty-state'>
        <div style='font-size:3rem;margin-bottom:1rem;'>🎯</div>
        <h3>Hey {first}, ready to find your match?</h3>
        <p>
            Upload your CV in the sidebar and click <strong>⚡ Get Career Match</strong>.<br>
            Optionally paste a Job Description to bias the search toward a specific role.<br><br>
            Your CV is matched against real JD embeddings using FAISS.<br>
            The LLM then explains your match %, skill gaps, salary ranges, and exact resume tips.
        </p>
        <div style='margin-top:1.5rem;font-size:0.72rem;color:#1e1e3a;'>
            CV + JD → Embedding → FAISS Vector Search → Retrieve Roles / Skills / Paths → LLM → Personalized Advice
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    retrieved = st.session_state.retrieved or {}
    similar   = retrieved.get("similar_roles", [])
    top       = retrieved.get("top_role") or (similar[0] if similar else {})
    parsed    = parse_analysis(st.session_state.analysis_raw or "") if st.session_state.analysis_raw else {}

    # Resolve display values — prefer LLM parsed, fallback to FAISS
    top_role_name = parsed.get("top_role") or top.get("role", top.get("title", ""))
    match_pct     = parsed.get("match_pct") or str(top.get("match_pct", ""))
    why_right     = parsed.get("why_right") or top.get("why_right_for_you", top.get("description", ""))
    next_steps    = parsed.get("next_steps") or ([top.get("next_steps")] if top.get("next_steps") else [])
    resume_add    = parsed.get("resume_add") or []
    career_path   = parsed.get("career_path") or []
    runner_up     = parsed.get("runner_up") or (similar[1].get("role", similar[1].get("title", "")) if len(similar) > 1 else "")
    runner_up_pct = similar[1]["match_pct"] if len(similar) > 1 else 0
    runner_up_why = parsed.get("runner_up_why") or (similar[1].get("description", "") if len(similar) > 1 else "")

    # ── Pipeline strip ────────────────────────────────────────────
    jd_on = "on" if st.session_state.jd_text else ""
    st.markdown(f"""
    <div class='pipeline'>
        <span class='ps on'>CV</span><span class='pa'>+</span>
        <span class='ps {jd_on}'>JD</span><span class='pa'>→</span>
        <span class='ps on'>Embedding</span><span class='pa'>→</span>
        <span class='ps on'>FAISS</span><span class='pa'>→</span>
        <span class='ps on'>Roles / Skills / Paths</span><span class='pa'>→</span>
        <span class='ps on'>LLM</span><span class='pa'>→</span>
        <span class='ps on'>Career Match</span>
    </div>
    """, unsafe_allow_html=True)

    # ── Name header ───────────────────────────────────────────────
    name = st.session_state.candidate_name
    first_lower = name.split()[0].lower() + ", " if name and name != "there" else ""
    st.markdown(f"""
    <div class='name-header'>
        <div style='font-size:1.8rem;margin-bottom:0.3rem;'>🎉</div>
        <h1>{first_lower}Here's Your Career Match!</h1>
        <p>Based on your CV, here's how you align with each data career path</p>
    </div>
    """, unsafe_allow_html=True)

    # ── HERO CARD ─────────────────────────────────────────────────
    if top_role_name and match_pct:
        top_emoji = top.get("emoji", "🧠")
        skills_html = "".join(
            f"<span class='skill-chip'>{s}</span>"
            for s in (top.get("skills", [])[:6])
        )
        st.markdown(f"""
        <div class='hero-card'>
            <div class='hero-badge'>✦ Top Match</div>
            <div class='hero-emoji'>{top_emoji}</div>
            <div class='hero-title'>{top_role_name}</div>
            <div class='hero-match'>{match_pct}% Match</div>
            <div class='hero-desc'>{why_right}</div>
            <div style='margin-top:0.75rem;'>{skills_html}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── FULL BREAKDOWN ────────────────────────────────────────────
    if similar:
        st.markdown("<div class='section-header'>📊 Full Breakdown</div>", unsafe_allow_html=True)
        cols = st.columns(min(len(similar), 4))
        for i, role in enumerate(similar[:4]):
            color = role.get("color", "#6366f1")
            emoji = role.get("emoji", "⚡")
            chips = "".join(
                f"<span class='breakdown-chip'>{s}</span>"
                for s in role.get("skills", [])[:4]
            )
            with cols[i]:
                st.markdown(f"""
                <div class='breakdown-card'>
                    <div class='breakdown-card-header'>
                        <span class='breakdown-emoji'>{emoji}</span>
                        <span class='breakdown-role'>{role.get('role', role.get('title', ''))}</span>
                    </div>
                    <div class='breakdown-pct' style='color:{color};'>{role['match_pct']}%</div>
                    <div class='breakdown-chips'>{chips}</div>
                </div>
                """, unsafe_allow_html=True)

    # ── WHY RIGHT FOR YOU ─────────────────────────────────────────
    if top:
        display_role = top_role_name or top.get("role", top.get("title", "This Role"))
        st.markdown(
            f"<div class='section-header'>💡 Why {display_role} is Right for You</div>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div class='detail-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='detail-card-desc'>{why_right}</div>", unsafe_allow_html=True)

        # Next Steps
        if next_steps:
            steps_html = "".join(f"<div>• {s}</div>" for s in next_steps if s)
            st.markdown(f"""
            <div class='info-box next-steps'>
                <div class='info-box-label'>📋 Next Steps</div>
                <div class='info-box-content'>{steps_html}</div>
            </div>
            """, unsafe_allow_html=True)
        elif top.get("next_steps"):
            st.markdown(f"""
            <div class='info-box next-steps'>
                <div class='info-box-label'>📋 Next Steps</div>
                <div class='info-box-content'>{top['next_steps']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Salary
        sal = top.get("salary", {})
        if sal:
            st.markdown(f"""
            <div class='info-box salary'>
                <div class='info-box-label'>💰 Salary (BD Market | Annually)</div>
                <div class='salary-grid'>
                    <div class='salary-item'>
                        <div class='salary-level'>Junior</div>
                        <div class='salary-range'>₹{sal.get('junior','—')}</div>
                    </div>
                    <div class='salary-item'>
                        <div class='salary-level'>Mid-Level</div>
                        <div class='salary-range'>₹{sal.get('mid','—')}</div>
                    </div>
                    <div class='salary-item'>
                        <div class='salary-level'>Senior</div>
                        <div class='salary-range'>₹{sal.get('senior','—')}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Market Demand
        demand = top.get("market_demand", "")
        if demand:
            demand_color = {
                "Extremely High": "#f472b6",
                "Very High": "#4ade80",
                "High": "#818cf8",
                "Medium-High": "#fb923c",
                "Medium": "#94a3b8",
            }.get(demand, "#818cf8")
            st.markdown(f"""
            <div class='info-box demand'>
                <div class='info-box-label'>📈 Market Demand</div>
                <div class='info-box-content' style='font-weight:700;color:{demand_color};font-size:1rem;'>{demand}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── SKILL GAPS ────────────────────────────────────────────────
    gaps = retrieved.get("skill_gaps", [])
    parsed_gaps = parsed.get("skill_gaps") or []
    if gaps or parsed_gaps:
        badges = "".join(f"<span class='gap-badge'>{g}</span>" for g in gaps[:8])
        gap_items = ""
        for item in parsed_gaps:
            if ":" in item:
                p = item.split(":", 1)
                gap_items += f"<div style='font-size:0.83rem;color:#fca5a5;margin:5px 0;'>• <strong style='color:#f87171;'>{p[0]}:</strong>{p[1]}</div>"
            else:
                gap_items += f"<div style='font-size:0.83rem;color:#fca5a5;margin:5px 0;'>• {item}</div>"
        st.markdown(f"""
        <div class='detail-card'>
            <div class='detail-card-title'>🔴 Skill Gaps to Bridge</div>
            <div style='margin-bottom:0.75rem;'>{badges}</div>
            {gap_items}
        </div>
        """, unsafe_allow_html=True)

    # ── RESUME RECOMMENDATIONS ────────────────────────────────────
    rskills = retrieved.get("resume_skills", [])
    resume_items = parsed.get("resume_add") or []
    if rskills or resume_items:
        badges = "".join(f"<span class='add-badge'>+ {s}</span>" for s in rskills)
        items_html = ""
        for item in resume_items:
            if ":" in item:
                p = item.split(":", 1)
                items_html += f"<div class='info-box resume' style='margin:5px 0;'><span style='font-weight:700;color:#86efac;'>{p[0]}:</span>{p[1]}</div>"
            else:
                items_html += f"<div class='info-box resume' style='margin:5px 0;'>{item}</div>"
        st.markdown(f"""
        <div class='detail-card'>
            <div class='detail-card-title'>✅ Resume Recommendations</div>
            <div style='margin-bottom:0.75rem;'>{badges}</div>
            {items_html}
        </div>
        """, unsafe_allow_html=True)

    # ── CAREER PATH ───────────────────────────────────────────────
    cp_items = career_path or []
    if not cp_items and top.get("career_path"):
        cp_items = [top["career_path"]]
    if cp_items:
        steps_html = ""
        for step in cp_items:
            if ":" in step:
                p = step.split(":", 1)
                steps_html += f"""
                <div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:12px;'>
                    <div class='cp-dot'></div>
                    <div style='font-size:0.85rem;'>
                        <span style='color:#a5b4fc;font-weight:700;'>{p[0]}:</span>
                        <span style='color:#94a3b8;'>{p[1]}</span>
                    </div>
                </div>"""
            else:
                steps_html += f"<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:8px;display:flex;gap:10px;'><span style='color:#6366f1;'>→</span><span>{step}</span></div>"
        st.markdown(f"""
        <div class='detail-card'>
            <div class='detail-card-title'>🗺️ Your Career Path</div>
            {steps_html}
        </div>
        """, unsafe_allow_html=True)

    # ── RUNNER-UP ─────────────────────────────────────────────────
    if runner_up:
        st.markdown(f"""
        <div class='runner-up'>
            🥈 <strong>Runner-up: {runner_up} ({runner_up_pct}%)</strong><br>
            <span style='color:#64748b;font-size:0.83rem;'>{runner_up_why}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── CHAT ──────────────────────────────────────────────────────
    st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>💬 Ask a Follow-up Question</div>", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = None
    if st.session_state.prefill:
        user_input = st.session_state.prefill
        st.session_state.prefill = None
    else:
        user_input = st.chat_input(
            "Ask about jobs, skills, salaries, or your career path...",
            disabled=(st.session_state.agent is None)
        )

    if user_input:
        if st.session_state.agent is None:
            st.warning("Upload your CV and click ⚡ Get Career Match first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Searching and analysing..."):
                    try:
                        from agent import run_agent
                        resp = run_agent(st.session_state.agent, user_input)
                        st.markdown(resp)
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        err = f"❌ Error: {str(e)}"
                        st.error(err)
                        st.session_state.messages.append({"role": "assistant", "content": err})
