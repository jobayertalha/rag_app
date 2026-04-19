"""
app.py — CV Analyzer
Flow: Welcome (name) → Home → Analyze CV | Match with JD
"""

import streamlit as st
import tempfile
import os
import re

st.set_page_config(
    page_title="CV Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

/* ── RESET & BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #07070f;
    color: #e2e8f0;
}
.stApp { background-color: #07070f; }
h1, h2, h3 { font-family: 'Syne', sans-serif; }

/* Hide Streamlit default header, sidebar toggle, footer */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem; max-width: 1100px; }

/* ── TOP NAV ── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2rem;
    height: 58px;
    background: rgba(10,10,22,0.95);
    border-bottom: 1px solid #16163a;
    position: sticky;
    top: 0;
    z-index: 999;
    backdrop-filter: blur(12px);
}
.topnav-brand {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #a855f7;
    display: flex;
    align-items: center;
    gap: 7px;
    cursor: pointer;
}
.topnav-links { display: flex; align-items: center; gap: 4px; }
.topnav-btn {
    background: transparent;
    border: none;
    color: #64748b;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem;
    font-weight: 500;
    padding: 6px 14px;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.15s;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 5px;
}
.topnav-btn:hover { color: #a5b4fc; background: #13133a; }
.topnav-btn.active { color: #a5b4fc; background: #13133a; }
.topnav-user {
    font-size: 0.82rem;
    color: #94a3b8;
    background: #13132a;
    border: 1px solid #1e1e3a;
    border-radius: 20px;
    padding: 5px 14px;
    font-weight: 500;
}

/* ── WELCOME SCREEN ── */
.welcome-wrap {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(120,40,200,0.13) 0%, transparent 70%),
                radial-gradient(ellipse 60% 50% at 80% 100%, rgba(219,39,119,0.08) 0%, transparent 70%),
                #07070f;
    padding: 2rem;
}
.welcome-logo {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
    text-align: center;
}
.welcome-logo span {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.welcome-sub {
    color: #475569;
    font-size: 0.95rem;
    text-align: center;
    margin-bottom: 2.5rem;
}
.welcome-card {
    background: #0d0d1e;
    border: 1px solid #1e1e40;
    border-radius: 20px;
    padding: 2.2rem 2.2rem 1.8rem;
    width: 100%;
    max-width: 400px;
    box-shadow: 0 25px 60px rgba(0,0,0,0.5);
}
.welcome-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 1rem;
    text-align: center;
}

/* ── HOME PAGE ── */
.home-hero {
    text-align: center;
    padding: 4rem 2rem 2.5rem;
}
.home-greeting {
    font-family: 'Syne', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
}
.home-greeting span { color: #a855f7; }
.home-sub {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 0;
}

.options-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    max-width: 760px;
    margin: 0 auto;
    padding: 2rem;
}
.option-card {
    background: #0d0d1e;
    border: 1px solid #1e1e3a;
    border-radius: 20px;
    padding: 2.2rem 1.8rem 1.8rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.22s ease;
    position: relative;
    overflow: hidden;
}
.option-card::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(168,85,247,0.06), rgba(236,72,153,0.04));
    opacity: 0;
    transition: opacity 0.22s;
}
.option-card:hover::before { opacity: 1; }
.option-card:hover { border-color: #6d28d9; transform: translateY(-3px); box-shadow: 0 12px 40px rgba(109,40,217,0.18); }
.option-card.jd:hover { border-color: #db2777; box-shadow: 0 12px 40px rgba(219,39,119,0.18); }
.option-icon { font-size: 2.5rem; margin-bottom: 1rem; }
.option-title { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.5rem; }
.option-desc { font-size: 0.82rem; color: #64748b; line-height: 1.65; margin-bottom: 1rem; }
.option-tags { font-size: 0.72rem; color: #7c3aed; font-weight: 500; }
.option-tags.jd { color: #db2777; }

/* ── ANALYSIS PAGE ── */
.page-header {
    padding: 2rem 2rem 1rem;
    border-bottom: 1px solid #12122a;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.page-header h2 { font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700; color: #f1f5f9; margin: 0; }
.page-header p { font-size: 0.82rem; color: #475569; margin: 2px 0 0; }

.upload-zone {
    background: #0d0d1e;
    border: 2px dashed #1e1e3a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #7c3aed; }

/* ── RESULT CARDS ── */
.hero-card {
    background: linear-gradient(140deg, #13132a, #1a0f35, #160f2a);
    border: 1px solid #2d2060;
    border-radius: 22px;
    padding: 2.5rem 2rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #7c3aed, #db2777);
    color: #fff; font-size: 0.62rem; font-weight: 700;
    padding: 4px 14px; border-radius: 20px; margin-bottom: 1rem;
    letter-spacing: 0.12em; text-transform: uppercase;
}
.hero-emoji { font-size: 2.4rem; margin-bottom: 0.4rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; color: #f1f5f9; margin-bottom: 0.2rem; }
.hero-match {
    font-family: 'Syne', sans-serif; font-size: 3.5rem; font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.75rem; line-height: 1.1;
}
.hero-desc { font-size: 0.9rem; color: #94a3b8; max-width: 520px; margin: 0 auto 1.2rem; line-height: 1.7; }
.skill-chip { display: inline-block; font-size: 0.7rem; background: rgba(99,102,241,0.12); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.25); border-radius: 6px; padding: 3px 10px; margin: 3px; }

.section-header { font-size: 0.7rem; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.12em; margin: 1.5rem 0 0.75rem; }

.breakdown-card { background: #0f0f20; border: 1px solid #1a1a30; border-radius: 14px; padding: 1.1rem 1rem; height: 100%; }
.breakdown-card-header { display: flex; align-items: center; gap: 8px; margin-bottom: 0.6rem; }
.breakdown-role { font-size: 0.78rem; font-weight: 600; color: #94a3b8; line-height: 1.3; }
.breakdown-pct { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; line-height: 1; }
.breakdown-chips { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px; }
.breakdown-chip { font-size: 0.62rem; background: #13132a; color: #64748b; border-radius: 4px; padding: 2px 6px; border: 1px solid #1a1a30; }

.detail-card { background: #0f0f20; border: 1px solid #1a1a30; border-radius: 16px; padding: 1.5rem; margin-bottom: 1rem; }
.detail-card-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.75rem; }
.detail-card-desc { font-size: 0.875rem; color: #94a3b8; line-height: 1.7; margin-bottom: 1rem; }

.info-box { border-radius: 10px; padding: 0.85rem 1rem; margin-bottom: 0.75rem; }
.info-box-label { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem; }
.info-box-content { font-size: 0.85rem; line-height: 1.65; color: #cbd5e1; }
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

.salary-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-top: 6px; }
.salary-item { background: #060e07; border-radius: 8px; padding: 0.6rem 0.7rem; border: 1px solid #0f2a14; }
.salary-level { font-size: 0.62rem; color: #4ade80; font-weight: 700; margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.08em; }
.salary-range { font-size: 0.78rem; color: #bbf7d0; font-weight: 600; }

.gap-badge { display: inline-block; font-size: 0.68rem; background: #1a0808; color: #fca5a5; border: 1px solid #4c1010; border-radius: 6px; padding: 3px 10px; margin: 2px; }
.add-badge { display: inline-block; font-size: 0.68rem; background: #071a09; color: #86efac; border: 1px solid #0f4018; border-radius: 6px; padding: 3px 10px; margin: 2px; }

.runner-up { background: #0d0d1a; border: 1px solid #1e1e3a; border-radius: 12px; padding: 1rem 1.2rem; font-size: 0.85rem; color: #94a3b8; line-height: 1.7; margin-top: 0.75rem; }
.runner-up strong { color: #fbbf24; font-family: 'Syne', sans-serif; }

/* JD match specific */
.match-score-big {
    font-family: 'Syne', sans-serif; font-size: 5rem; font-weight: 800; line-height: 1;
    background: linear-gradient(135deg, #ec4899, #a855f7);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.jd-match-card {
    background: linear-gradient(140deg, #160d1e, #130d20);
    border: 1px solid #4a1a4a;
    border-radius: 22px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.match-label { font-size: 0.75rem; color: #94a3b8; margin-top: 0.25rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }

/* Streamlit overrides */
.stTextInput > div > div > input {
    background: #13132a !important; border: 1px solid #1e1e3a !important;
    color: #e2e8f0 !important; border-radius: 10px !important; padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus { border-color: #7c3aed !important; box-shadow: none !important; }
.stTextArea > div > div > textarea { background: #0f0f20 !important; border: 1px solid #1a1a30 !important; color: #cbd5e1 !important; border-radius: 10px !important; }
.stFileUploader { background: transparent !important; }
[data-testid="stChatInput"] textarea { background: #0f0f20 !important; border: 1px solid #1e1e3a !important; color: #e2e8f0 !important; border-radius: 10px !important; }

div.stButton > button {
    background: linear-gradient(135deg, #7c3aed, #db2777) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.88rem !important; padding: 0.55rem 1.4rem !important;
    cursor: pointer !important; transition: opacity 0.2s !important;
}
div.stButton > button:hover { opacity: 0.85 !important; }
div.stButton > button[kind="secondary"] {
    background: #13132a !important;
    color: #a5b4fc !important;
    border: 1px solid #1e1e3a !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────
for k, v in {
    "page": "welcome",        # welcome | home | analyze | jd_match
    "candidate_name": "",
    "cv_text": None,
    "jd_text": "",
    "messages": [],
    "messages_jd": [],
    "agent": None,
    "agent_jd": None,
    "retrieved": None,
    "analysis_raw": None,
    "jd_analysis_raw": None,
    "jd_retrieved": None,
    "prefill": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


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


def go(page): st.session_state.page = page; st.rerun()


# ══════════════════════════════════════════════════════════════════
# TOP NAV (shown on all pages except welcome)
# ══════════════════════════════════════════════════════════════════
def render_nav():
    page = st.session_state.page
    name = st.session_state.candidate_name
    first = name.split()[0].capitalize() if name and name != "there" else "User"

    home_active   = "active" if page == "home"     else ""
    anal_active   = "active" if page == "analyze"  else ""
    jd_active     = "active" if page == "jd_match" else ""

    st.markdown(f"""
    <div class='topnav'>
        <div class='topnav-brand'>📄 CV Analyzer</div>
        <div class='topnav-links' id='navlinks'></div>
        <div class='topnav-user'>👤 {first}</div>
    </div>
    """, unsafe_allow_html=True)

    # Use Streamlit buttons inside columns for navigation
    nav_cols = st.columns([2, 1, 1, 1, 2])
    with nav_cols[1]:
        if st.button("🏠 Home", key="nav_home"):
            go("home")
    with nav_cols[2]:
        if st.button("📊 Analysis", key="nav_analyze"):
            go("analyze")
    with nav_cols[3]:
        if st.button("🎯 JD Match", key="nav_jd"):
            go("jd_match")


# ══════════════════════════════════════════════════════════════════
# PAGE: WELCOME
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":
    st.markdown("""
    <div class='welcome-wrap'>
        <div class='welcome-logo'>Your CV <span>Analyzer</span></div>
        <div class='welcome-sub'>Powered by FAISS vector search + LLM analysis</div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
        <div class='welcome-card'>
            <div class='welcome-label'>👋 Welcome! What's your name?</div>
        </div>
        """, unsafe_allow_html=True)
        name_val = st.text_input("Your name", placeholder="e.g. Talha Jobayer",
                                  label_visibility="collapsed", key="name_field")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Get Started →", use_container_width=True):
            n = name_val.strip()
            st.session_state.candidate_name = n if n else "there"
            go("home")
    st.stop()


# ══════════════════════════════════════════════════════════════════
# NAV (all non-welcome pages)
# ══════════════════════════════════════════════════════════════════
render_nav()


# ══════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    name = st.session_state.candidate_name
    first = name.split()[0].capitalize() if name and name != "there" else "there"

    st.markdown(f"""
    <div class='home-hero'>
        <div class='home-greeting'>Hello, <span>{first}!</span> 👋</div>
        <div class='home-sub'>Ready to analyze your CV?</div>
    </div>
    """, unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 6, 1])
    with mid:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class='option-card' style='cursor:default;'>
                <div class='option-icon'>📄</div>
                <div class='option-title'>Analyze My CV</div>
                <div class='option-desc'>Get matched with AI/ML roles from our curated knowledge base using FAISS vector search.</div>
                <div class='option-tags'>Skills | Roles | Salary | Career Path</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("📄 Analyze My CV", use_container_width=True, key="home_analyze"):
                go("analyze")

        with col2:
            st.markdown("""
            <div class='option-card jd' style='cursor:default;'>
                <div class='option-icon'>🎯</div>
                <div class='option-title'>Match with Job Description</div>
                <div class='option-desc'>Paste a JD and see exactly how well your CV matches. Uncover skill gaps and alignment score.</div>
                <div class='option-tags jd'>Match % | Missing Skills | Companies</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            if st.button("🎯 Match with JD", use_container_width=True, key="home_jd"):
                go("jd_match")

    st.stop()


# ══════════════════════════════════════════════════════════════════
# PAGE: ANALYZE CV
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "analyze":
    st.markdown("""
    <div class='page-header'>
        <div>
            <h2>📊 Analyze My CV</h2>
            <p>Upload your CV and get matched to AI/data roles from our knowledge base</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Upload section (only show if no results yet)
    if not st.session_state.cv_text or not st.session_state.analysis_raw:
        _, col_mid, _ = st.columns([1, 4, 1])
        with col_mid:
            st.markdown("""
            <div class='upload-zone'>
                <div style='font-size:2rem;margin-bottom:0.5rem;'>📎</div>
                <div style='font-size:0.88rem;color:#64748b;margin-bottom:1rem;'>Upload your CV (PDF)</div>
            </div>
            """, unsafe_allow_html=True)
            uploaded_cv = st.file_uploader("Upload CV PDF", type=["pdf"],
                                            label_visibility="collapsed", key="cv_upload")
            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            if uploaded_cv:
                if st.button("⚡ Get Career Match", use_container_width=True, key="analyze_btn"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_cv.read()); tmp_path = tmp.name
                    with st.spinner("Reading CV..."):
                        from agent import extract_cv_text
                        cv_text = extract_cv_text(tmp_path)
                        os.unlink(tmp_path)
                    st.session_state.cv_text = cv_text
                    st.session_state.messages = []

                    with st.spinner("Matching CV → FAISS knowledge base..."):
                        from rag import retrieve_context
                        st.session_state.retrieved = retrieve_context(cv_text, "", k=5)

                    with st.spinner("Building AI agent..."):
                        from agent import build_agent
                        st.session_state.agent = build_agent(
                            cv_text, "", st.session_state.candidate_name)

                    with st.spinner("Generating career analysis..."):
                        from agent import run_agent
                        raw = run_agent(st.session_state.agent,
                            "Analyse this candidate's CV and give a full career match. "
                            "Follow EXACTLY these tags:\n"
                            "TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
                            "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY\n"
                            "Be specific — reference actual CV skills throughout."
                        )
                        st.session_state.analysis_raw = raw
                    st.rerun()
            else:
                st.markdown("<div style='text-align:center;color:#334155;font-size:0.82rem;padding:1rem 0;'>Upload a PDF to get started</div>", unsafe_allow_html=True)

    # ── Results ────────────────────────────────────────────────────
    if st.session_state.cv_text and st.session_state.analysis_raw:
        retrieved = st.session_state.retrieved or {}
        similar   = retrieved.get("similar_roles", [])
        top       = retrieved.get("top_role") or (similar[0] if similar else {})
        parsed    = parse_analysis(st.session_state.analysis_raw)

        top_role_name = parsed.get("top_role") or top.get("title", top.get("role", ""))
        match_pct     = parsed.get("match_pct") or str(top.get("match_pct", ""))
        why_right     = parsed.get("why_right") or top.get("description", "")
        next_steps    = parsed.get("next_steps") or []
        resume_add    = parsed.get("resume_add") or []
        career_path   = parsed.get("career_path") or []
        runner_up     = parsed.get("runner_up") or (similar[1].get("title", "") if len(similar) > 1 else "")
        runner_up_pct = similar[1]["match_pct"] if len(similar) > 1 else 0
        runner_up_why = parsed.get("runner_up_why") or ""

        # Refresh button
        col_r1, col_r2 = st.columns([5, 1])
        with col_r2:
            if st.button("↺ New CV", key="new_cv"):
                st.session_state.cv_text = None
                st.session_state.analysis_raw = None
                st.session_state.retrieved = None
                st.session_state.agent = None
                st.session_state.messages = []
                st.rerun()

        # Hero card
        top_emoji = top.get("emoji", "🧠")
        skills_html = "".join(f"<span class='skill-chip'>{s}</span>" for s in top.get("skills", [])[:6])
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

        # Breakdown
        if similar:
            st.markdown("<div class='section-header'>📊 Full Breakdown</div>", unsafe_allow_html=True)
            cols = st.columns(min(len(similar), 4))
            for i, role in enumerate(similar[:4]):
                color = role.get("color", "#6366f1")
                emoji = role.get("emoji", "⚡")
                chips = "".join(f"<span class='breakdown-chip'>{s}</span>" for s in role.get("skills", [])[:4])
                with cols[i]:
                    st.markdown(f"""
                    <div class='breakdown-card'>
                        <div class='breakdown-card-header'><span>{emoji}</span><span class='breakdown-role'>{role.get('title', role.get('role', ''))}</span></div>
                        <div class='breakdown-pct' style='color:{color};'>{role['match_pct']}%</div>
                        <div class='breakdown-chips'>{chips}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # Why right
        if top:
            display_role = top_role_name or top.get("title", "This Role")
            next_html = "".join(f"<div>• {s}</div>" for s in next_steps if s)
            st.markdown(f"""
            <div style='margin-top:1.5rem;'></div>
            <div class='detail-card'>
                <div class='detail-card-title'>💡 Why {display_role} is Right for You</div>
                <div class='detail-card-desc'>{why_right}</div>
                {f"<div class='info-box next-steps'><div class='info-box-label'>📋 Next Steps</div><div class='info-box-content'>{next_html}</div></div>" if next_html else ""}
            </div>
            """, unsafe_allow_html=True)

        # Salary
        sal = top.get("salary", {})
        if sal:
            st.markdown(f"""
            <div class='detail-card'>
                <div class='info-box salary'>
                    <div class='info-box-label'>💰 Salary (BDT | Approx. Annually)</div>
                    <div class='salary-grid'>
                        <div class='salary-item'><div class='salary-level'>Junior</div><div class='salary-range'>~৳{sal.get('junior','—')}</div></div>
                        <div class='salary-item'><div class='salary-level'>Mid-Level</div><div class='salary-range'>~৳{sal.get('mid','—')}</div></div>
                        <div class='salary-item'><div class='salary-level'>Senior</div><div class='salary-range'>~৳{sal.get('senior','—')}</div></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Market demand
        demand = top.get("market_demand", "")
        if demand:
            demand_color = {"Extremely High": "#f472b6", "Very High": "#4ade80", "High": "#818cf8", "Medium-High": "#fb923c", "Medium": "#94a3b8"}.get(demand, "#818cf8")
            st.markdown(f"""
            <div class='detail-card'>
                <div class='info-box demand'>
                    <div class='info-box-label'>📈 Market Demand</div>
                    <div class='info-box-content' style='font-weight:700;color:{demand_color};font-size:1rem;'>{demand}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Skill gaps
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

        # Resume recommendations
        rskills = retrieved.get("resume_skills", [])
        resume_items = parsed.get("resume_add") or []
        if rskills or resume_items:
            badges_r = "".join(f"<span class='add-badge'>+ {s}</span>" for s in rskills)
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
                <div style='margin-bottom:0.75rem;'>{badges_r}</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)

        # Career path
        cp_items = career_path or []
        if not cp_items and top.get("career_path"):
            cp_items = [top["career_path"]]
        if cp_items:
            steps_parts = []
            for step in cp_items:
                if ":" in step:
                    p = step.split(":", 1)
                    steps_parts.append(
                        "<div style='display:flex;gap:12px;align-items:flex-start;margin-bottom:12px;'>"
                        "<div style='width:8px;height:8px;border-radius:50%;background:#6366f1;margin-top:5px;flex-shrink:0;'></div>"
                        f"<div style='font-size:0.85rem;'><span style='color:#a5b4fc;font-weight:700;'>{p[0]}:</span><span style='color:#94a3b8;'>{p[1]}</span></div></div>"
                    )
                else:
                    steps_parts.append(f"<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:8px;display:flex;gap:10px;'><span style='color:#6366f1;'>→</span><span>{step}</span></div>")
            st.markdown(f"<div class='detail-card'><div class='detail-card-title'>🗺️ Your Career Path</div>{''.join(steps_parts)}</div>", unsafe_allow_html=True)

        # Runner-up
        if runner_up:
            st.markdown(f"""
            <div class='runner-up'>
                🥈 <strong>Runner-up: {runner_up} ({runner_up_pct}%)</strong><br>
                <span style='color:#64748b;font-size:0.83rem;'>{runner_up_why}</span>
            </div>
            """, unsafe_allow_html=True)

        # Chat
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>💬 Ask a Follow-up Question</div>", unsafe_allow_html=True)

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Ask about roles, skills, salaries, or your career path...", key="chat_analyze")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from agent import run_agent
                        resp = run_agent(st.session_state.agent, user_input)
                        st.markdown(resp)
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        err = f"❌ Error: {str(e)}"
                        st.error(err)

    st.stop()


# ══════════════════════════════════════════════════════════════════
# PAGE: MATCH WITH JD
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "jd_match":
    st.markdown("""
    <div class='page-header'>
        <div>
            <h2>🎯 Match with Job Description</h2>
            <p>Upload your CV and paste a JD to see your exact match score and skill gaps</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.jd_analysis_raw:
        _, col_mid, _ = st.columns([1, 4, 1])
        with col_mid:
            st.markdown("<div style='font-size:0.78rem;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;'>Upload Your CV (PDF)</div>", unsafe_allow_html=True)
            uploaded_cv_jd = st.file_uploader("CV PDF", type=["pdf"],
                                               label_visibility="collapsed", key="cv_upload_jd")

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:0.78rem;color:#475569;font-weight:600;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;'>Paste Job Description</div>", unsafe_allow_html=True)
            jd_input = st.text_area("Job Description", height=180,
                                     placeholder="Paste the full job description from LinkedIn, Indeed, Glassdoor...",
                                     label_visibility="collapsed", key="jd_textarea")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            if uploaded_cv_jd and jd_input.strip():
                if st.button("🎯 Analyse JD Match", use_container_width=True, key="jd_match_btn"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_cv_jd.read()); tmp_path = tmp.name
                    with st.spinner("Reading CV..."):
                        from agent import extract_cv_text
                        cv_text = extract_cv_text(tmp_path)
                        os.unlink(tmp_path)
                    st.session_state.cv_text = cv_text
                    st.session_state.jd_text = jd_input.strip()
                    st.session_state.messages_jd = []

                    with st.spinner("Matching against JD..."):
                        from rag import retrieve_context
                        st.session_state.jd_retrieved = retrieve_context(
                            cv_text, jd_input.strip(), k=5)

                    with st.spinner("Building AI agent..."):
                        from agent import build_agent
                        st.session_state.agent_jd = build_agent(
                            cv_text, jd_input.strip(), st.session_state.candidate_name)

                    with st.spinner("Generating JD match analysis..."):
                        from agent import run_agent
                        raw = run_agent(st.session_state.agent_jd,
                            "Analyse how well this candidate's CV matches the specific Job Description provided. "
                            "Give a match score focused on this JD. Follow EXACTLY these tags:\n"
                            "TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
                            "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY\n"
                            "Focus on the JD requirements specifically."
                        )
                        st.session_state.jd_analysis_raw = raw
                    st.rerun()
            else:
                if not uploaded_cv_jd:
                    st.markdown("<div style='text-align:center;color:#334155;font-size:0.82rem;padding:0.5rem 0;'>⬆ Upload your CV PDF to get started</div>", unsafe_allow_html=True)
                elif not jd_input.strip():
                    st.markdown("<div style='text-align:center;color:#334155;font-size:0.82rem;padding:0.5rem 0;'>📝 Paste a job description above</div>", unsafe_allow_html=True)

    # ── JD Results ────────────────────────────────────────────────
    if st.session_state.jd_analysis_raw:
        retrieved = st.session_state.jd_retrieved or {}
        similar   = retrieved.get("similar_roles", [])
        top       = retrieved.get("top_role") or (similar[0] if similar else {})
        parsed    = parse_analysis(st.session_state.jd_analysis_raw)

        top_role_name = parsed.get("top_role") or top.get("title", top.get("role", ""))
        match_pct     = parsed.get("match_pct") or str(top.get("match_pct", ""))
        why_right     = parsed.get("why_right") or top.get("description", "")
        next_steps    = parsed.get("next_steps") or []
        resume_add    = parsed.get("resume_add") or []
        career_path   = parsed.get("career_path") or []

        col_r1, col_r2 = st.columns([5, 1])
        with col_r2:
            if st.button("↺ New Match", key="new_jd"):
                st.session_state.jd_analysis_raw = None
                st.session_state.jd_retrieved = None
                st.session_state.agent_jd = None
                st.session_state.messages_jd = []
                st.rerun()

        # JD Match hero card
        top_emoji = top.get("emoji", "🎯")
        skills_html = "".join(f"<span class='skill-chip'>{s}</span>" for s in top.get("skills", [])[:6])
        st.markdown(f"""
        <div class='jd-match-card'>
            <div class='hero-badge' style='background:linear-gradient(135deg,#db2777,#a855f7);'>🎯 JD Match Score</div>
            <div class='hero-emoji'>{top_emoji}</div>
            <div class='hero-title'>{top_role_name}</div>
            <div class='match-score-big'>{match_pct}%</div>
            <div class='match-label'>Match with Job Description</div>
            <div class='hero-desc' style='margin-top:1rem;'>{why_right}</div>
            <div style='margin-top:0.75rem;'>{skills_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # Skill gaps from JD
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
                <div class='detail-card-title'>🔴 Missing Skills for This JD</div>
                <div style='margin-bottom:0.75rem;'>{badges}</div>
                {gap_items}
            </div>
            """, unsafe_allow_html=True)

        # Resume tips for JD
        rskills = retrieved.get("resume_skills", [])
        resume_items = parsed.get("resume_add") or []
        if rskills or resume_items:
            badges_r = "".join(f"<span class='add-badge'>+ {s}</span>" for s in rskills)
            items_html = ""
            for item in resume_items:
                if ":" in item:
                    p = item.split(":", 1)
                    items_html += f"<div class='info-box resume' style='margin:5px 0;'><span style='font-weight:700;color:#86efac;'>{p[0]}:</span>{p[1]}</div>"
                else:
                    items_html += f"<div class='info-box resume' style='margin:5px 0;'>{item}</div>"
            st.markdown(f"""
            <div class='detail-card'>
                <div class='detail-card-title'>✅ Resume Tips for This JD</div>
                <div style='margin-bottom:0.75rem;'>{badges_r}</div>
                {items_html}
            </div>
            """, unsafe_allow_html=True)

        # Next steps
        if next_steps:
            next_html = "".join(f"<div>• {s}</div>" for s in next_steps if s)
            st.markdown(f"""
            <div class='detail-card'>
                <div class='info-box next-steps'>
                    <div class='info-box-label'>📋 Steps to Land This Role</div>
                    <div class='info-box-content'>{next_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Chat for JD
        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>💬 Ask About This JD</div>", unsafe_allow_html=True)

        for msg in st.session_state.messages_jd:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input_jd = st.chat_input("Ask about the JD requirements, how to improve your match...", key="chat_jd")
        if user_input_jd:
            st.session_state.messages_jd.append({"role": "user", "content": user_input_jd})
            with st.chat_message("user"):
                st.markdown(user_input_jd)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        from agent import run_agent
                        resp = run_agent(st.session_state.agent_jd, user_input_jd)
                        st.markdown(resp)
                        st.session_state.messages_jd.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        err = f"❌ Error: {str(e)}"
                        st.error(err)

    st.stop()
