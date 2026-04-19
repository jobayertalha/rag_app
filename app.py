"""
app.py — CV Analyzer (Professional Edition)
Flow: Welcome → Home → Analyze CV | Match with JD
Nav: Pure HTML sticky header spanning full width, no Streamlit button blocks in nav
"""

import streamlit as st
import tempfile
import os
import re

st.set_page_config(
    page_title="CV Analyzer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0b0b10;
    color: #d1d5db;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: #0b0b10; }

/* Hide all Streamlit chrome */
#MainMenu, footer, header,
[data-testid="collapsedControl"],
[data-testid="stSidebar"],
[data-testid="stToolbar"],
.stDeployButton { display: none !important; visibility: hidden !important; }

/* Remove top padding completely */
section[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── NAVBAR ── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    height: 54px;
    background: rgba(11,11,16,0.96);
    border-bottom: 1px solid #18181f;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    z-index: 9999;
    display: flex;
    align-items: center;
    padding: 0 2rem;
    gap: 0;
}
.nav-brand {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f9fafb;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 8px;
    min-width: 160px;
}
.nav-brand-dot {
    width: 7px; height: 7px;
    background: #7c3aed;
    border-radius: 50%;
    display: inline-block;
}
.nav-center {
    display: flex;
    align-items: center;
    gap: 2px;
    flex: 1;
    justify-content: center;
}
.nav-link {
    font-size: 0.8rem;
    font-weight: 500;
    color: #4b5563;
    padding: 5px 14px;
    border-radius: 6px;
    cursor: pointer;
    border: none;
    background: transparent;
    font-family: 'Inter', sans-serif;
    transition: color 0.15s, background 0.15s;
    text-decoration: none;
    display: inline-block;
    line-height: 1;
}
.nav-link:hover { color: #d1d5db; background: #18181f; }
.nav-link.active { color: #a78bfa; background: #1a1a2c; }
.nav-right {
    display: flex;
    align-items: center;
    gap: 10px;
    min-width: 160px;
    justify-content: flex-end;
}
.nav-user {
    font-size: 0.78rem;
    font-weight: 500;
    color: #6b7280;
    background: #111118;
    border: 1px solid #18181f;
    border-radius: 20px;
    padding: 4px 12px;
}

/* Main content: offset for fixed nav */
.main-pad { padding-top: 54px; }

/* Inner max-width wrapper */
.cw {
    max-width: 920px;
    margin: 0 auto;
    padding: 0 2rem;
}

/* ── WELCOME ── */
.welcome-outer {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 2rem;
    background: #0b0b10;
}
.welcome-box {
    width: 100%;
    max-width: 360px;
    text-align: center;
}
.welcome-icon {
    width: 40px; height: 40px;
    background: #7c3aed;
    border-radius: 9px;
    margin: 0 auto 1.4rem;
    display: flex; align-items: center; justify-content: center;
}
.welcome-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f9fafb;
    letter-spacing: -0.03em;
    margin-bottom: 0.35rem;
}
.welcome-subtitle {
    font-size: 0.82rem;
    color: #374151;
    margin-bottom: 2rem;
    line-height: 1.6;
}
.welcome-form {
    background: #111118;
    border: 1px solid #18181f;
    border-radius: 10px;
    padding: 1.6rem;
    text-align: left;
}
.flabel {
    display: block;
    font-size: 0.72rem;
    font-weight: 600;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 7px;
}

/* ── HOME ── */
.home-hero {
    padding: 3.5rem 0 2.5rem;
    text-align: center;
}
.home-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f9fafb;
    letter-spacing: -0.03em;
    margin-bottom: 0.4rem;
}
.home-title em { font-style: normal; color: #7c3aed; }
.home-sub { font-size: 0.85rem; color: #374151; }

.opt-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 2rem; }
.opt-card {
    background: #111118;
    border: 1px solid #18181f;
    border-radius: 10px;
    padding: 1.8rem;
    text-align: left;
    transition: border-color 0.18s;
}
.opt-card:hover { border-color: #4c1d95; }
.opt-card.jd:hover { border-color: #9d174d; }
.opt-badge {
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #7c3aed;
    background: rgba(124,58,237,0.1);
    border-radius: 4px;
    padding: 3px 8px;
    margin-bottom: 0.9rem;
}
.opt-badge.jd { color: #db2777; background: rgba(219,39,119,0.1); }
.opt-title {
    font-size: 1rem;
    font-weight: 600;
    color: #f9fafb;
    margin-bottom: 0.45rem;
    letter-spacing: -0.01em;
}
.opt-desc { font-size: 0.8rem; color: #374151; line-height: 1.65; margin-bottom: 1.1rem; }
.opt-meta { font-size: 0.68rem; color: #1f2937; font-family: 'JetBrains Mono', monospace; }

/* ── PAGE HEADER ── */
.ph {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #111118;
    margin-bottom: 2rem;
}
.ph h2 { font-size: 1.2rem; font-weight: 600; color: #f9fafb; letter-spacing: -0.02em; margin-bottom: 0.3rem; }
.ph p { font-size: 0.8rem; color: #374151; }

/* ── RESULTS ── */
.hero-card {
    background: #111118;
    border: 1px solid #18181f;
    border-radius: 10px;
    padding: 2.4rem 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-block;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #7c3aed;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 4px;
    padding: 3px 10px;
    margin-bottom: 1rem;
}
.hero-role { font-size: 1.5rem; font-weight: 700; color: #f9fafb; letter-spacing: -0.03em; margin-bottom: 0.3rem; }
.hero-pct { font-size: 3.8rem; font-weight: 700; color: #7c3aed; letter-spacing: -0.04em; line-height: 1; margin-bottom: 0.2rem; }
.hero-pct-lbl { font-size: 0.68rem; color: #374151; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 1rem; }
.hero-desc { font-size: 0.83rem; color: #4b5563; max-width: 480px; margin: 0 auto; line-height: 1.7; }

.sk-chip {
    display: inline-block;
    font-size: 0.68rem;
    font-family: 'JetBrains Mono', monospace;
    background: #0e0e18;
    color: #4b5563;
    border: 1px solid #18181f;
    border-radius: 4px;
    padding: 2px 7px;
    margin: 2px;
}

.section-lbl {
    font-size: 0.65rem;
    font-weight: 700;
    color: #1f2937;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 1.8rem 0 0.9rem;
}

.bd-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.7rem; margin-bottom: 1.5rem; }
.bd-card { background: #111118; border: 1px solid #18181f; border-radius: 8px; padding: 1rem; }
.bd-role { font-size: 0.72rem; font-weight: 500; color: #4b5563; margin-bottom: 0.4rem; line-height: 1.4; }
.bd-pct { font-size: 1.7rem; font-weight: 700; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.4rem; }
.bd-chip { display: inline-block; font-size: 0.58rem; background: #0e0e18; color: #374151; border-radius: 3px; padding: 2px 5px; margin: 1px; border: 1px solid #1a1a26; font-family: 'JetBrains Mono', monospace; }

.ic {
    background: #111118;
    border: 1px solid #18181f;
    border-radius: 8px;
    padding: 1.3rem;
    margin-bottom: 0.7rem;
}
.ic-title { font-size: 0.78rem; font-weight: 600; color: #6b7280; margin-bottom: 0.9rem; letter-spacing: -0.01em; }
.ic-body { font-size: 0.82rem; color: #4b5563; line-height: 1.7; }

.sal-row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 0.6rem; }
.sal-item { background: #0e0e18; border: 1px solid #13131e; border-radius: 6px; padding: 0.7rem; }
.sal-level { font-size: 0.6rem; font-weight: 700; color: #1f2937; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 4px; }
.sal-val { font-size: 0.78rem; font-weight: 600; color: #9ca3af; font-family: 'JetBrains Mono', monospace; }

.gap-tag { display: inline-block; font-size: 0.66rem; font-family: 'JetBrains Mono', monospace; background: rgba(239,68,68,0.06); color: #f87171; border: 1px solid rgba(239,68,68,0.14); border-radius: 4px; padding: 2px 8px; margin: 2px; }
.add-tag { display: inline-block; font-size: 0.66rem; font-family: 'JetBrains Mono', monospace; background: rgba(16,185,129,0.06); color: #34d399; border: 1px solid rgba(16,185,129,0.14); border-radius: 4px; padding: 2px 8px; margin: 2px; }

.path-step { display: flex; gap: 10px; align-items: flex-start; margin-bottom: 10px; }
.path-dot { width: 5px; height: 5px; border-radius: 50%; background: #7c3aed; margin-top: 8px; flex-shrink: 0; }
.path-txt { font-size: 0.81rem; color: #4b5563; line-height: 1.6; }
.path-txt strong { color: #6b7280; font-weight: 600; }

.runnerup { background: #0e0e18; border: 1px solid #13131e; border-radius: 8px; padding: 1rem 1.2rem; font-size: 0.8rem; color: #4b5563; margin-top: 1rem; line-height: 1.6; }
.runnerup strong { color: #f59e0b; font-weight: 600; }

.jd-pct { font-size: 4rem; font-weight: 700; color: #db2777; letter-spacing: -0.04em; line-height: 1; margin-bottom: 0.2rem; }

.fhint { font-size: 0.72rem; color: #1f2937; margin-top: 0.4rem; }

/* ── STREAMLIT OVERRIDES ── */
div.stButton > button {
    background: #7c3aed !important;
    color: #fff !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    padding: 0.52rem 1.1rem !important;
    transition: background 0.15s !important;
    letter-spacing: -0.01em !important;
}
div.stButton > button:hover { background: #6d28d9 !important; }
div.stButton > button[kind="secondary"] {
    background: #111118 !important;
    color: #4b5563 !important;
    border: 1px solid #18181f !important;
}
div.stButton > button[kind="secondary"]:hover { color: #9ca3af !important; border-color: #374151 !important; }
div.stButton > button:disabled { background: #111118 !important; color: #1f2937 !important; cursor: not-allowed !important; }

.stTextInput > div > div > input {
    background: #0e0e18 !important; border: 1px solid #18181f !important;
    color: #f9fafb !important; border-radius: 7px !important;
    padding: 0.52rem 0.8rem !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.83rem !important;
}
.stTextInput > div > div > input::placeholder { color: #1f2937 !important; }
.stTextInput > div > div > input:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important; outline: none !important; }

.stTextArea > div > div > textarea {
    background: #0e0e18 !important; border: 1px solid #18181f !important;
    color: #d1d5db !important; border-radius: 7px !important;
    font-family: 'Inter', sans-serif !important; font-size: 0.82rem !important;
}
.stTextArea > div > div > textarea:focus { border-color: #7c3aed !important; box-shadow: 0 0 0 3px rgba(124,58,237,0.1) !important; }

[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stChatInput"] textarea {
    background: #111118 !important; border: 1px solid #18181f !important;
    color: #d1d5db !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────
for k, v in {
    "page": "welcome",
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
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def go(page):
    st.session_state.page = page
    st.rerun()


def sign_out():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


def parse_analysis(text: str) -> dict:
    def get(tag):
        m = re.search(rf"{tag}:\s*(.+?)(?=\n[A-Z_]+:|$)", text, re.DOTALL)
        return m.group(1).strip() if m else ""
    def get_list(tag):
        m = re.search(rf"{tag}:\s*((?:\n- .+)+)", text)
        if not m: return []
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
# WELCOME
# ══════════════════════════════════════════════════════════════════
if st.session_state.page == "welcome":
    # Enter key submits via JS
    st.markdown("""
    <script>
    setTimeout(function() {
        const inputs = window.parent.document.querySelectorAll('input[type="text"]');
        inputs.forEach(function(inp) {
            inp.addEventListener('keydown', function(e) {
                if (e.key === 'Enter') {
                    const btn = window.parent.document.querySelector('button[kind="primary"]');
                    if (btn) btn.click();
                }
            });
        });
    }, 800);
    </script>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='welcome-outer'>
        <div class='welcome-box'>
            <div class='welcome-icon'>
                <svg width='20' height='20' viewBox='0 0 24 24' fill='none' xmlns='http://www.w3.org/2000/svg'>
                    <path d='M9 12h6M9 16h6M7 3H5a2 2 0 00-2 2v14a2 2 0 002 2h14a2 2 0 002-2V8l-5-5H7z'
                        stroke='white' stroke-width='1.5' stroke-linecap='round' stroke-linejoin='round'/>
                </svg>
            </div>
            <div class='welcome-title'>CV Analyzer</div>
            <div class='welcome-subtitle'>Match your CV to AI and data roles using<br>FAISS vector search and LLM analysis.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 1.1, 1])
    with col:
        st.markdown("<div class='welcome-form'><span class='flabel'>Your name</span></div>", unsafe_allow_html=True)
        name_val = st.text_input(
            "name", placeholder="e.g. Talha Jobayer",
            label_visibility="collapsed", key="name_field"
        )
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        if st.button("Continue", use_container_width=True, key="welcome_btn"):
            n = (name_val or "").strip()
            st.session_state.candidate_name = n if n else "there"
            go("home")

    st.stop()


# ══════════════════════════════════════════════════════════════════
# NAVBAR — rendered via query param navigation
# ══════════════════════════════════════════════════════════════════
# Handle nav param from href clicks
nav_param = st.query_params.get("nav", None)
if nav_param in ("home", "analyze", "jd_match"):
    st.query_params.clear()
    go(nav_param)

page = st.session_state.page
name = st.session_state.candidate_name
first = name.split()[0].capitalize() if name and name != "there" else "User"

h = "active" if page == "home"     else ""
a = "active" if page == "analyze"  else ""
j = "active" if page == "jd_match" else ""

st.markdown(f"""
<div class='navbar'>
    <div class='nav-brand'><span class='nav-brand-dot'></span>CV Analyzer</div>
    <div class='nav-center'>
        <a class='nav-link {h}' href='?nav=home'>Home</a>
        <a class='nav-link {a}' href='?nav=analyze'>Analysis</a>
        <a class='nav-link {j}' href='?nav=jd_match'>JD Match</a>
    </div>
    <div class='nav-right'>
        <div class='nav-user'>{first}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sign out — inline with the fixed navbar via absolute positioning trick
# We put a Streamlit button that is visually hidden behind the nav using CSS
st.markdown("""
<style>
/* Sign out button: fixed top-right inside the navbar area */
div[data-testid="stHorizontalBlock"]:has(> div > div[data-testid="column"]:last-child > div.stButton > button#signout_btn) {
    position: fixed !important;
    top: 10px !important;
    right: 1.5rem !important;
    z-index: 10000 !important;
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)

_, _, so_col = st.columns([10, 2, 1])
with so_col:
    if st.button("Sign out", key="signout_btn", type="secondary"):
        sign_out()


# ══════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown("<div class='main-pad'><div class='cw'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='home-hero'>
        <div class='home-title'>Hello, <em>{first}</em></div>
        <div class='home-sub'>Ready to analyze your CV?</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='opt-card'>
            <div class='opt-badge'>Knowledge Base</div>
            <div class='opt-title'>Analyze My CV</div>
            <div class='opt-desc'>Match your CV against curated AI and data role descriptions using FAISS vector search. Get role fit scores, salary ranges, and career path guidance.</div>
            <div class='opt-meta'>Skills · Roles · Salary · Career Path</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Analyze My CV", use_container_width=True, key="btn_analyze"):
            go("analyze")

    with col2:
        st.markdown("""
        <div class='opt-card jd'>
            <div class='opt-badge jd'>JD Matching</div>
            <div class='opt-title'>Match with Job Description</div>
            <div class='opt-desc'>Paste any job description and see your exact match score. Identify missing skills, get resume tailoring tips, and understand your gap to the role.</div>
            <div class='opt-meta'>Match % · Skill Gaps · Resume Tips</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Match with JD", use_container_width=True, key="btn_jd"):
            go("jd_match")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════
# ANALYZE CV
# ══════════════════════════════════════════════════════════════════
if page == "analyze":
    st.markdown("<div class='main-pad'><div class='cw'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='ph'>
        <h2>CV Analysis</h2>
        <p>Upload your CV to get matched against our AI and data roles knowledge base</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.analysis_raw:
        st.markdown("<span class='flabel'>Upload CV (PDF)</span>", unsafe_allow_html=True)
        uploaded_cv = st.file_uploader("CV PDF", type=["pdf"],
                                        label_visibility="collapsed", key="cv_upload")
        if uploaded_cv:
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            if st.button("Run Analysis", key="analyze_btn"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_cv.read()); tmp_path = tmp.name
                with st.spinner("Extracting CV text..."):
                    from agent import extract_cv_text
                    cv_text = extract_cv_text(tmp_path); os.unlink(tmp_path)
                st.session_state.cv_text = cv_text
                st.session_state.messages = []
                with st.spinner("Running FAISS vector search..."):
                    from rag import retrieve_context
                    st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
                with st.spinner("Initializing agent..."):
                    from agent import build_agent
                    st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
                with st.spinner("Generating career analysis..."):
                    from agent import run_agent
                    raw = run_agent(st.session_state.agent,
                        "Analyse this candidate's CV and give a full career match. "
                        "Follow EXACTLY these tags:\n"
                        "TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
                        "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY\n"
                        "Be specific — reference actual CV skills throughout.")
                    st.session_state.analysis_raw = raw
                st.rerun()
        else:
            st.markdown("<div class='fhint'>Accepts PDF format.</div>", unsafe_allow_html=True)

    if st.session_state.analysis_raw:
        retrieved = st.session_state.retrieved or {}
        similar   = retrieved.get("similar_roles", [])
        top       = retrieved.get("top_role") or (similar[0] if similar else {})
        parsed    = parse_analysis(st.session_state.analysis_raw)

        top_role_name = parsed.get("top_role") or top.get("title", top.get("role", ""))
        match_pct     = parsed.get("match_pct") or str(top.get("match_pct", ""))
        why_right     = parsed.get("why_right") or top.get("description", "")
        next_steps    = parsed.get("next_steps") or []
        career_path   = parsed.get("career_path") or []
        runner_up     = parsed.get("runner_up") or (similar[1].get("title", "") if len(similar) > 1 else "")
        runner_up_pct = similar[1]["match_pct"] if len(similar) > 1 else 0
        runner_up_why = parsed.get("runner_up_why") or ""

        _, btn_col = st.columns([8, 1])
        with btn_col:
            if st.button("New CV", key="new_cv", type="secondary"):
                st.session_state.cv_text = None; st.session_state.analysis_raw = None
                st.session_state.retrieved = None; st.session_state.agent = None
                st.session_state.messages = []; st.rerun()

        skills_html = "".join(f"<span class='sk-chip'>{s}</span>" for s in top.get("skills", [])[:6])
        st.markdown(f"""
        <div class='hero-card'>
            <div class='hero-badge'>Top Match</div>
            <div class='hero-role'>{top_role_name}</div>
            <div class='hero-pct'>{match_pct}%</div>
            <div class='hero-pct-lbl'>Match Score</div>
            <div class='hero-desc'>{why_right}</div>
            <div style='margin-top:1rem;'>{skills_html}</div>
        </div>
        """, unsafe_allow_html=True)

        if similar:
            st.markdown("<div class='section-lbl'>All Role Matches</div>", unsafe_allow_html=True)
            bd = "<div class='bd-grid'>"
            for role in similar[:4]:
                color = role.get("color", "#7c3aed")
                chips = "".join(f"<span class='bd-chip'>{s}</span>" for s in role.get("skills", [])[:3])
                bd += f"<div class='bd-card'><div class='bd-role'>{role.get('title',role.get('role',''))}</div><div class='bd-pct' style='color:{color};'>{role['match_pct']}%</div><div>{chips}</div></div>"
            bd += "</div>"
            st.markdown(bd, unsafe_allow_html=True)

        if why_right:
            next_html = "".join(f"<div class='path-step'><div class='path-dot'></div><div class='path-txt'>{s}</div></div>" for s in next_steps if s)
            st.markdown(f"""
            <div class='ic'>
                <div class='ic-title'>Why {top_role_name} fits you</div>
                <div class='ic-body'>{why_right}</div>
                {f"<div style='margin-top:1rem;font-size:0.68rem;font-weight:700;color:#1f2937;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:8px;'>Next Steps</div>{next_html}" if next_html else ""}
            </div>
            """, unsafe_allow_html=True)

        sal = top.get("salary", {})
        if sal:
            st.markdown(f"""
            <div class='ic'>
                <div class='ic-title'>Salary Ranges — BDT, Annual</div>
                <div class='sal-row'>
                    <div class='sal-item'><div class='sal-level'>Junior</div><div class='sal-val'>৳{sal.get('junior','—')}</div></div>
                    <div class='sal-item'><div class='sal-level'>Mid</div><div class='sal-val'>৳{sal.get('mid','—')}</div></div>
                    <div class='sal-item'><div class='sal-level'>Senior</div><div class='sal-val'>৳{sal.get('senior','—')}</div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        demand = top.get("market_demand", "")
        if demand:
            st.markdown(f"""
            <div class='ic'>
                <div class='ic-title'>Market Demand</div>
                <div style='font-size:0.95rem;font-weight:700;color:#a78bfa;'>{demand}</div>
            </div>
            """, unsafe_allow_html=True)

        gaps = retrieved.get("skill_gaps", [])
        parsed_gaps = parsed.get("skill_gaps") or []
        if gaps or parsed_gaps:
            badges = "".join(f"<span class='gap-tag'>{g}</span>" for g in gaps[:8])
            gi = ""
            for item in parsed_gaps:
                if ":" in item:
                    p = item.split(":", 1)
                    gi += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'><span style='color:#f87171;font-weight:600;'>{p[0]}</span> —{p[1]}</div>"
                else:
                    gi += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'>— {item}</div>"
            st.markdown(f"<div class='ic'><div class='ic-title'>Skill Gaps</div><div style='margin-bottom:0.7rem;'>{badges}</div>{gi}</div>", unsafe_allow_html=True)

        rskills = retrieved.get("resume_skills", [])
        resume_items = parsed.get("resume_add") or []
        if rskills or resume_items:
            add_b = "".join(f"<span class='add-tag'>{s}</span>" for s in rskills)
            ri = ""
            for item in resume_items:
                if ":" in item:
                    p = item.split(":", 1)
                    ri += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'><span style='color:#34d399;font-weight:600;'>{p[0]}</span> —{p[1]}</div>"
                else:
                    ri += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'>— {item}</div>"
            st.markdown(f"<div class='ic'><div class='ic-title'>Resume Recommendations</div><div style='margin-bottom:0.7rem;'>{add_b}</div>{ri}</div>", unsafe_allow_html=True)

        cp_items = career_path or []
        if not cp_items and top.get("career_path"):
            cp_items = [top["career_path"]]
        if cp_items:
            sh = ""
            for step in cp_items:
                if ":" in step:
                    p = step.split(":", 1)
                    sh += f"<div class='path-step'><div class='path-dot'></div><div class='path-txt'><strong>{p[0]}:</strong>{p[1]}</div></div>"
                else:
                    sh += f"<div class='path-step'><div class='path-dot'></div><div class='path-txt'>{step}</div></div>"
            st.markdown(f"<div class='ic'><div class='ic-title'>Career Path</div>{sh}</div>", unsafe_allow_html=True)

        if runner_up:
            st.markdown(f"<div class='runnerup'><strong>Runner-up: {runner_up} ({runner_up_pct}%)</strong><br><span style='font-size:0.78rem;'>{runner_up_why}</span></div>", unsafe_allow_html=True)

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-lbl'>Follow-up Questions</div>", unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        user_input = st.chat_input("Ask about roles, skills, salaries, or career path...", key="chat_analyze")
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"): st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    try:
                        from agent import run_agent
                        resp = run_agent(st.session_state.agent, user_input)
                        st.markdown(resp)
                        st.session_state.messages.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════════
# JD MATCH
# ══════════════════════════════════════════════════════════════════
if page == "jd_match":
    st.markdown("<div class='main-pad'><div class='cw'>", unsafe_allow_html=True)

    st.markdown("""
    <div class='ph'>
        <h2>JD Match Analysis</h2>
        <p>Upload your CV and paste a job description to get a precise match score</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.jd_analysis_raw:
        st.markdown("<span class='flabel'>Upload CV (PDF)</span>", unsafe_allow_html=True)
        uploaded_cv_jd = st.file_uploader("CV PDF", type=["pdf"],
                                           label_visibility="collapsed", key="cv_upload_jd")
        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("<span class='flabel'>Job Description</span>", unsafe_allow_html=True)
        jd_input = st.text_area("JD", height=200,
            placeholder="Paste the full job description from LinkedIn, Indeed, or any job board...",
            label_visibility="collapsed", key="jd_textarea")
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        ready = uploaded_cv_jd is not None and bool((jd_input or "").strip())
        if st.button("Analyse JD Match", disabled=not ready, key="jd_match_btn"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_cv_jd.read()); tmp_path = tmp.name
            with st.spinner("Extracting CV text..."):
                from agent import extract_cv_text
                cv_text = extract_cv_text(tmp_path); os.unlink(tmp_path)
            st.session_state.cv_text = cv_text
            st.session_state.jd_text = jd_input.strip()
            st.session_state.messages_jd = []
            with st.spinner("Running FAISS match against JD..."):
                from rag import retrieve_context
                st.session_state.jd_retrieved = retrieve_context(cv_text, jd_input.strip(), k=5)
            with st.spinner("Initializing agent..."):
                from agent import build_agent
                st.session_state.agent_jd = build_agent(cv_text, jd_input.strip(), st.session_state.candidate_name)
            with st.spinner("Generating JD match analysis..."):
                from agent import run_agent
                raw = run_agent(st.session_state.agent_jd,
                    "Analyse how well this candidate's CV matches the specific Job Description provided. "
                    "Focus your match score on THIS JD specifically. Follow EXACTLY these tags:\n"
                    "TOP_ROLE, MATCH_PCT, WHY_RIGHT, NEXT_STEPS, "
                    "SKILL_GAPS, RESUME_ADD, CAREER_PATH, RUNNER_UP, RUNNER_UP_WHY\n"
                    "Be specific about JD requirements vs candidate skills.")
                st.session_state.jd_analysis_raw = raw
            st.rerun()

        if not ready:
            st.markdown("<div class='fhint'>Both CV and job description are required.</div>", unsafe_allow_html=True)

    if st.session_state.jd_analysis_raw:
        retrieved = st.session_state.jd_retrieved or {}
        similar   = retrieved.get("similar_roles", [])
        top       = retrieved.get("top_role") or (similar[0] if similar else {})
        parsed    = parse_analysis(st.session_state.jd_analysis_raw)

        top_role_name = parsed.get("top_role") or top.get("title", top.get("role", ""))
        match_pct     = parsed.get("match_pct") or str(top.get("match_pct", ""))
        why_right     = parsed.get("why_right") or top.get("description", "")
        next_steps    = parsed.get("next_steps") or []

        _, btn_col = st.columns([8, 1])
        with btn_col:
            if st.button("New Match", key="new_jd", type="secondary"):
                st.session_state.jd_analysis_raw = None; st.session_state.jd_retrieved = None
                st.session_state.agent_jd = None; st.session_state.messages_jd = []; st.rerun()

        skills_html = "".join(f"<span class='sk-chip'>{s}</span>" for s in top.get("skills", [])[:6])
        st.markdown(f"""
        <div class='hero-card'>
            <div class='hero-badge' style='color:#db2777;background:rgba(219,39,119,0.08);border-color:rgba(219,39,119,0.16);'>JD Match Score</div>
            <div class='hero-role'>{top_role_name}</div>
            <div class='jd-pct'>{match_pct}%</div>
            <div class='hero-pct-lbl'>Match with Job Description</div>
            <div class='hero-desc' style='margin-top:0.75rem;'>{why_right}</div>
            <div style='margin-top:1rem;'>{skills_html}</div>
        </div>
        """, unsafe_allow_html=True)

        gaps = retrieved.get("skill_gaps", [])
        parsed_gaps = parsed.get("skill_gaps") or []
        if gaps or parsed_gaps:
            badges = "".join(f"<span class='gap-tag'>{g}</span>" for g in gaps[:8])
            gi = ""
            for item in parsed_gaps:
                if ":" in item:
                    p = item.split(":", 1)
                    gi += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'><span style='color:#f87171;font-weight:600;'>{p[0]}</span> —{p[1]}</div>"
                else:
                    gi += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'>— {item}</div>"
            st.markdown(f"<div class='ic'><div class='ic-title'>Skills Missing for This JD</div><div style='margin-bottom:0.7rem;'>{badges}</div>{gi}</div>", unsafe_allow_html=True)

        rskills = retrieved.get("resume_skills", [])
        resume_items = parsed.get("resume_add") or []
        if rskills or resume_items:
            add_b = "".join(f"<span class='add-tag'>{s}</span>" for s in rskills)
            ri = ""
            for item in resume_items:
                if ":" in item:
                    p = item.split(":", 1)
                    ri += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'><span style='color:#34d399;font-weight:600;'>{p[0]}</span> —{p[1]}</div>"
                else:
                    ri += f"<div style='font-size:0.8rem;color:#4b5563;margin:5px 0;'>— {item}</div>"
            st.markdown(f"<div class='ic'><div class='ic-title'>Resume Tailoring for This JD</div><div style='margin-bottom:0.7rem;'>{add_b}</div>{ri}</div>", unsafe_allow_html=True)

        if next_steps:
            sh = "".join(f"<div class='path-step'><div class='path-dot'></div><div class='path-txt'>{s}</div></div>" for s in next_steps if s)
            st.markdown(f"<div class='ic'><div class='ic-title'>Steps to Strengthen Your Application</div>{sh}</div>", unsafe_allow_html=True)

        st.markdown("<div style='height:2rem'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-lbl'>Ask About This JD</div>", unsafe_allow_html=True)
        for msg in st.session_state.messages_jd:
            with st.chat_message(msg["role"]): st.markdown(msg["content"])
        user_input_jd = st.chat_input("Ask about JD requirements or how to improve your match...", key="chat_jd")
        if user_input_jd:
            st.session_state.messages_jd.append({"role": "user", "content": user_input_jd})
            with st.chat_message("user"): st.markdown(user_input_jd)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    try:
                        from agent import run_agent
                        resp = run_agent(st.session_state.agent_jd, user_input_jd)
                        st.markdown(resp)
                        st.session_state.messages_jd.append({"role": "assistant", "content": resp})
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()
