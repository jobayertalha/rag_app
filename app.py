"""
app.py — AI Career Platform
Extends existing CV Analyzer with: Home Dashboard, CV Analysis,
JD Matching, AI/ML Interest Quiz, and About page.
"""

import streamlit as st
import tempfile
import os
import re

from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context, load_roles
from quiz import QUESTIONS, calculate_quiz_score, MAX_SCORE

st.set_page_config(
    page_title="AI Career Platform",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# CSS — Unified dark theme, extended for new pages
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

/* Navbar */
.navbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(15, 15, 32, 0.98);
    border-bottom: 1px solid #2d2d5a;
    padding: 0.6rem 2rem;
    margin-bottom: 2rem;
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Cards */
.card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 1.2rem;
    margin-bottom: 1rem;
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
.hero-match {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}

/* Mode / Feature Cards */
.mode-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s;
    cursor: pointer;
}
.mode-card:hover {
    border-color: #7c3aed;
    transform: translateY(-3px);
    box-shadow: 0 8px 32px rgba(168, 85, 247, 0.15);
}
.mode-icon { font-size: 2.2rem; margin-bottom: 0.6rem; }
.mode-title { font-size: 1.05rem; font-weight: 700; color: #f1f5f9; }
.mode-desc { font-size: 0.72rem; color: #64748b; margin-top: 0.3rem; line-height: 1.4; }

/* JD Match */
.jd-match-card {
    background: linear-gradient(135deg, #0f0f20 0%, #1a0f35 100%);
    border: 1px solid #2d2060;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.jd-match-score { font-size: 3rem; font-weight: 800; }

/* Welcome */
.welcome-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 24px;
    padding: 2rem;
    text-align: center;
    max-width: 420px;
    margin: 3rem auto;
}
.welcome-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    text-align: center;
    margin-top: 2rem;
}
.welcome-gradient {
    background: linear-gradient(135deg, #a855f7, #ec4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Quiz-specific */
.quiz-question-card {
    background: #0f0f20;
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.6rem;
}
.quiz-category-badge {
    display: inline-block;
    background: rgba(168, 85, 247, 0.15);
    border: 1px solid rgba(168, 85, 247, 0.3);
    color: #c084fc;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    margin-bottom: 0.6rem;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.quiz-verdict-card {
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.quiz-progress-bar-bg {
    background: #1e1e3a;
    border-radius: 8px;
    height: 8px;
    margin: 0.5rem 0;
}
.result-correct {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
}
.result-wrong {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    border-radius: 10px;
    padding: 0.6rem 1rem;
    margin-bottom: 0.4rem;
}

/* About page */
.about-hero {
    background: linear-gradient(135deg, #0f0f20 0%, #1a0a2e 100%);
    border: 1px solid #2d1a5a;
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    margin-bottom: 1.5rem;
}
.contact-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1a1a30;
    color: #cbd5e1;
    font-size: 0.9rem;
}
.contact-icon { font-size: 1.1rem; width: 28px; text-align: center; }
.tech-pill {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    color: #a5b4fc;
    font-size: 0.72rem;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    margin: 0.2rem;
}

/* Nav active state helper */
.nav-active {
    background: rgba(168, 85, 247, 0.15) !important;
    color: #a855f7 !important;
}

/* Streamlit button overrides for nav */
div[data-testid="stHorizontalBlock"] .stButton button {
    background: transparent;
    border: 1px solid transparent;
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
    border-radius: 8px;
    transition: all 0.2s;
}
div[data-testid="stHorizontalBlock"] .stButton button:hover {
    background: rgba(168, 85, 247, 0.1);
    color: #a855f7;
    border-color: rgba(168, 85, 247, 0.3);
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE — Initialize all keys once
# ============================================================
_defaults = {
    "page": "home",
    "mode": None,
    "tab": "overview",
    "messages": [],
    "agent": None,
    "cv_text": None,
    "analysis_raw": None,
    "retrieved": None,
    "candidate_name": "",
    "name_entered": False,
    "matched_companies": [],
    "jd_result": None,
    "show_dropdown": False,
    # Quiz state
    "quiz_started": False,
    "quiz_answers": {},
    "quiz_submitted": False,
    "quiz_score_result": None,
    "quiz_current_q": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# HELPERS — unchanged from original
# ============================================================
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
        "skill_gaps": get_list("SKILL_GAPS"),
        "resume_add": get_list("RESUME_ADD"),
        "career_path": get_list("CAREER_PATH"),
    }


def match_companies(cv_text: str) -> list:
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
                    "salary": salary_str,
                    "location": role.get("location", "Dhaka"),
                })
        companies.sort(key=lambda x: x["match_score"], reverse=True)
        return companies[:6]
    except Exception:
        return []


def calculate_jd_match(cv_text: str, jd_text: str) -> dict:
    cv_lower = cv_text.lower()
    jd_lower = jd_text.lower()
    keywords = set()
    for skill in ['python', 'sql', 'tensorflow', 'pytorch', 'langchain', 'rag', 'llm', 'nlp',
                  'docker', 'kubernetes', 'aws', 'gcp', 'pandas', 'numpy', 'scikit-learn', 'git']:
        if skill in jd_lower:
            keywords.add(skill)
    found = [kw for kw in keywords if kw in cv_lower]
    missing = [kw for kw in keywords if kw not in cv_lower]
    pct = int((len(found) / max(len(keywords), 1)) * 100)
    return {
        "match_pct": min(95, pct),
        "matched": found[:15],
        "missing": missing[:15],
        "found_count": len(found),
        "total": len(keywords),
    }


def nav_goto(page, extra_state: dict = None):
    st.session_state.page = page
    if extra_state:
        for k, v in extra_state.items():
            st.session_state[k] = v
    st.rerun()


# ============================================================
# NAVBAR
# ============================================================
def render_navbar():
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    page = st.session_state.page

    col_logo, c1, c2, c3, c4, c5, col_user = st.columns([1.4, 0.7, 0.75, 0.75, 0.65, 0.65, 0.9])

    with col_logo:
        if st.button("🚀 AI Career Platform", key="nav_logo", use_container_width=True):
            nav_goto("home", {"mode": None})

    with c1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True):
            nav_goto("home", {"mode": None})

    with c2:
        if st.button("📄 Analyze", key="nav_analyze", use_container_width=True):
            nav_goto("home", {"mode": "cv"})

    with c3:
        if st.button("🎯 JD Match", key="nav_jd", use_container_width=True):
            nav_goto("home", {"mode": "jd"})

    with c4:
        if st.button("🧠 Quiz", key="nav_quiz", use_container_width=True):
            nav_goto("quiz")

    with c5:
        if st.button("ℹ️ About", key="nav_about", use_container_width=True):
            nav_goto("about")

    with col_user:
        if st.button(f"👤 {first} ▼", key="nav_user", use_container_width=True):
            st.session_state.show_dropdown = not st.session_state.show_dropdown
            st.rerun()

    if st.session_state.show_dropdown:
        st.markdown(f"""
        <div style="background:#0f0f20;border:1px solid #2d2d5a;border-radius:12px;
                    padding:0.8rem;max-width:200px;margin-left:auto;margin-right:0.5rem;">
            <div style="font-size:0.7rem;color:#64748b;">Signed in as</div>
            <div style="font-size:0.85rem;font-weight:600;color:#cbd5e1;margin-bottom:0.5rem;">{name}</div>
            <hr style="border-color:#2d2d5a;margin:0.3rem 0;">
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Sign Out / Change Name", key="signout_confirm", use_container_width=True):
            for k in ["name_entered", "cv_text", "analysis_raw", "retrieved", "agent",
                      "messages", "show_dropdown", "jd_result", "matched_companies",
                      "quiz_started", "quiz_answers", "quiz_submitted", "quiz_score_result"]:
                st.session_state[k] = _defaults.get(k, False if "entered" in k or "submitted" in k or "started" in k else None)
            st.session_state.page = "home"
            st.rerun()

    st.markdown("<hr style='margin:0.4rem 0 1.5rem 0;border-color:#1a1a30;'>", unsafe_allow_html=True)


# ============================================================
# WELCOME SCREEN
# ============================================================
def render_welcome():
    st.markdown("""
    <div class="welcome-title">
        AI <span class="welcome-gradient">Career Platform</span>
    </div>
    <p style="text-align:center;color:#64748b;margin-bottom:2rem;font-size:1rem;">
        CV Analysis · JD Matching · AI/ML Interest Quiz
    </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<div class="welcome-card">', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.15rem;font-weight:600;margin-bottom:1rem;">👋 Welcome! What\'s your name?</div>', unsafe_allow_html=True)
        with st.form(key="name_form"):
            name = st.text_input("Name", placeholder="e.g. Talha Jobayer", label_visibility="collapsed")
            if st.form_submit_button("✨ Get Started →", use_container_width=True, type="primary"):
                if name and name.strip():
                    st.session_state.candidate_name = name.strip()
                    st.session_state.name_entered = True
                    st.rerun()
                else:
                    st.error("Please enter your name to continue")
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# HOME PAGE — Dashboard with 3 feature cards
# ============================================================
def render_home():
    name = st.session_state.candidate_name
    first = name.split()[0]

    st.markdown(f"<h2 style='text-align:center;margin-bottom:0.2rem;'>Hello, {first}! 👋</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748b;margin-bottom:2rem;'>Your AI-powered career companion. What would you like to do today?</p>", unsafe_allow_html=True)

    # ── 3 clickable feature cards ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>📄</div>
            <div class='mode-title'>Analyze My CV</div>
            <div class='mode-desc'>Upload your CV and get matched with the best AI/ML roles from our knowledge base.</div>
            <div class='mode-desc' style='color:#a855f7;margin-top:0.5rem;'>✨ Role Match · Skills · Salary · Career Path</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📄 Analyze My CV", key="home_cv", use_container_width=True, type="primary"):
            st.session_state.mode = "cv"
            st.rerun()

    with col2:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>🎯</div>
            <div class='mode-title'>Match with Job Description</div>
            <div class='mode-desc'>Paste a JD and see exactly how well your CV aligns with that specific role.</div>
            <div class='mode-desc' style='color:#10b981;margin-top:0.5rem;'>✨ Match % · Missing Skills · Target Companies</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Match with JD", key="home_jd", use_container_width=True):
            st.session_state.mode = "jd"
            st.rerun()

    with col3:
        st.markdown("""
        <div class='mode-card'>
            <div class='mode-icon'>🧠</div>
            <div class='mode-title'>AI/ML Interest Quiz</div>
            <div class='mode-desc'>Take a 15-question quiz covering Python, math, logic, and AI concepts to gauge your fit.</div>
            <div class='mode-desc' style='color:#f59e0b;margin-top:0.5rem;'>✨ Aptitude Score · Role Suggestions · Learning Path</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧠 Take the Quiz", key="home_quiz", use_container_width=True):
            nav_goto("quiz")

    # ── CV Analysis inline UI ──
    if st.session_state.mode == "cv":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'><h4 style='margin-bottom:1rem;'>📄 Upload your CV</h4>", unsafe_allow_html=True)
        uploaded = st.file_uploader("CV PDF", type=["pdf"], key="cv_upload", label_visibility="collapsed")
        if uploaded and st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            _process_cv_analysis(uploaded)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── JD Match inline UI ──
    elif st.session_state.mode == "jd":
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='card'><h4 style='margin-bottom:1rem;'>🎯 Match Your CV Against a Job Description</h4>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            uploaded = st.file_uploader("Your CV (PDF)", type=["pdf"], key="jd_upload")
        with col_b:
            jd_text = st.text_area("Job Description", height=160, placeholder="Paste the full job description here...")
        if uploaded and jd_text and st.button("🎯 Calculate Match", type="primary", use_container_width=True):
            _process_jd_match(uploaded, jd_text)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Chat shortcut if agent ready ──
    if st.session_state.agent:
        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns(2)
        with col_l:
            if st.button("💬 Continue Career Chat", use_container_width=True):
                nav_goto("chat")
        with col_r:
            if st.button("📊 View Full Analysis", use_container_width=True):
                nav_goto("analyze")


# ============================================================
# CV ANALYSIS PROCESSING — unchanged logic
# ============================================================
def _process_cv_analysis(uploaded):
    with st.spinner("Analyzing your CV…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            cv_text = extract_cv_text(tmp.name)
            os.unlink(tmp.name)
        st.session_state.cv_text = cv_text
        st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
        st.session_state.matched_companies = match_companies(cv_text)
        st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
        raw = run_agent(
            st.session_state.agent,
            "Analyse this CV. Follow tags: TOP_ROLE, MATCH_PCT, WHY_RIGHT, SKILL_GAPS, RESUME_ADD, CAREER_PATH"
        )
        st.session_state.analysis_raw = raw
    nav_goto("analyze")


def _process_jd_match(uploaded, jd_text):
    with st.spinner("Calculating match…"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded.read())
            cv_text = extract_cv_text(tmp.name)
            os.unlink(tmp.name)
        st.session_state.cv_text = cv_text
        st.session_state.jd_result = calculate_jd_match(cv_text, jd_text)
        st.session_state.matched_companies = match_companies(cv_text)
    nav_goto("jd_result")


# ============================================================
# JD RESULT PAGE — unchanged logic
# ============================================================
def render_jd_result():
    r = st.session_state.jd_result
    if not r:
        st.warning("No result found. Please run a JD match first.")
        if st.button("← Back to Home"):
            nav_goto("home", {"mode": "jd"})
        return

    pct = r["match_pct"]
    if pct < 30:
        color, status = "#ef4444", "Low Match"
    elif pct < 60:
        color, status = "#f59e0b", "Partial Match"
    elif pct < 80:
        color, status = "#10b981", "Good Match"
    else:
        color, status = "#06b6d4", "Excellent Match!"

    st.markdown(f"""
    <div class='jd-match-card'>
        <div style='color:{color};font-size:0.9rem;font-weight:600;letter-spacing:0.05em;'>JD MATCH SCORE</div>
        <div class='jd-match-score' style='color:{color};'>{pct}%</div>
        <div style='color:#f1f5f9;font-size:1.1rem;font-weight:600;'>{status}</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Keywords Matched", f"{r['found_count']} / {r['total']}")
        if r.get("matched"):
            st.markdown("**✅ Skills found in your CV**")
            st.markdown(" ".join(f"<span class='skill-chip'>{kw}</span>" for kw in r["matched"][:10]), unsafe_allow_html=True)
    with col2:
        if r.get("missing"):
            st.markdown("**❌ Missing skills — add these to your CV**")
            st.markdown(" ".join(f"<span class='skill-chip gap-chip'>{kw}</span>" for kw in r["missing"][:8]), unsafe_allow_html=True)
        if st.session_state.matched_companies:
            st.markdown("**🏢 Companies to target**")
            for c in st.session_state.matched_companies[:3]:
                st.markdown(f"- **{c['name']}** ({c['match_score']}% match)")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("← Run New Match", use_container_width=True):
            nav_goto("home", {"mode": "jd"})
    with c2:
        if st.button("🏠 Home", use_container_width=True):
            nav_goto("home", {"mode": None})


# ============================================================
# ANALYSIS PAGE — unchanged logic
# ============================================================
def render_analysis():
    if not st.session_state.cv_text:
        st.warning("Please upload and analyze your CV first.")
        if st.button("← Go to Home"):
            nav_goto("home", {"mode": "cv"})
        return

    r = st.session_state.retrieved or {}
    matches = r.get("all_matches", [])
    top = r.get("top_match", {})
    readiness = r.get("readiness", {})
    parsed = parse_analysis(st.session_state.analysis_raw or "") if st.session_state.analysis_raw else {}

    t1, t2, t3, t4, t5 = st.tabs(["📊 Overview", "🏢 Companies", "💰 Salary", "🔧 Skills", "💬 Chat"])

    with t1:
        st.markdown(f"<div class='hero-match'>{parsed.get('match_pct', top.get('match_pct', 0))}% Match</div>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>{parsed.get('top_role', top.get('title', 'AI Professional'))}</h3>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align:center;color:#94a3b8;'>{parsed.get('why_right', 'Great alignment with your skills!')}</p>", unsafe_allow_html=True)

        score = readiness.get("total_score", 0)
        if score < 30:
            st.warning(f"🔴 Readiness Score: {score}% — Focus on fundamentals")
        elif score < 60:
            st.warning(f"🟡 Readiness Score: {score}% — Keep building your skills")
        else:
            st.success(f"🟢 Readiness Score: {score}% — Ready to apply!")

        path = parsed.get("career_path", [])
        if path:
            st.markdown("### 🗺️ Career Path")
            for i, p in enumerate(path[:5]):
                st.markdown(f"**{i+1}.** {p}")

    with t2:
        if st.session_state.matched_companies:
            for c in st.session_state.matched_companies[:5]:
                st.markdown(f"**{c['name']}** — {c['role']} ({c['match_score']}% match)  \n📍 {c['location']}")
                if c.get("salary"):
                    st.caption(f"💰 {c['salary']}/month")
                st.markdown("---")
        else:
            st.info("No company data available for your CV.")

    with t3:
        for role in matches[:4]:
            if role.get("salary_min"):
                st.metric(role.get("title", "Role"), f"৳{role['salary_min']:,} – ৳{role['salary_max']:,}")

    with t4:
        gaps = parsed.get("skill_gaps") or r.get("skill_gaps", [])
        if gaps:
            st.markdown("**❌ Skill Gaps**")
            st.markdown(" ".join(f"<span class='skill-chip gap-chip'>{g}</span>" for g in gaps[:8]), unsafe_allow_html=True)
        recs = parsed.get("resume_add") or r.get("resume_skills", [])
        if recs:
            st.markdown("**➕ Recommended Additions**")
            st.markdown(" ".join(f"<span class='skill-chip'>+ {sk}</span>" for sk in recs[:8]), unsafe_allow_html=True)

    with t5:
        render_chat_inline()


# ============================================================
# CHAT PAGE (standalone) + inline version
# ============================================================
def render_chat_inline():
    if not st.session_state.agent:
        st.warning("Please analyze your CV first to unlock chat.")
        return
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    if prompt := st.chat_input("Ask about your career…", key="chat_inline_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                resp = run_agent(st.session_state.agent, prompt)
                st.markdown(resp)
                st.session_state.messages.append({"role": "assistant", "content": resp})
        st.rerun()


def render_chat():
    if not st.session_state.agent:
        st.warning("Please analyze your CV first.")
        if st.button("← Go to Home"):
            nav_goto("home", {"mode": "cv"})
        return
    st.markdown("### 💬 Career Chat")
    render_chat_inline()


# ============================================================
# CAREER RECOMMENDATIONS — unchanged logic
# ============================================================
def render_career_rec():
    if not st.session_state.cv_text:
        st.warning("Please analyze your CV first.")
        return
    r = st.session_state.retrieved or {}
    score = r.get("readiness", {}).get("total_score", 0)
    st.markdown("### 🎯 Career Recommendations")
    if score < 30:
        st.warning("🔴 **Focus on Fundamentals**")
        st.markdown("- Complete Python basics\n- Take machine learning courses\n- Build 2–3 small projects\n- Get relevant certifications")
    elif score < 60:
        st.warning("🟡 **Building Momentum**")
        st.markdown("- Take advanced ML courses\n- Build portfolio projects\n- Get cloud certifications (AWS/GCP)\n- Contribute to open source")
    else:
        st.success("🟢 **Ready for Job Search!**")
        st.markdown("- Update LinkedIn profile\n- Start applying to matching companies\n- Prepare for technical interviews\n- Network with industry professionals")


# ============================================================
# QUIZ PAGE — New module
# ============================================================
def render_quiz():
    st.markdown("<h2 style='text-align:center;'>🧠 AI/ML Interest & Aptitude Quiz</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#64748b;'>15 questions · Python · Math · Logic · AI/ML Concepts · Interest</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Results screen ──
    if st.session_state.quiz_submitted and st.session_state.quiz_score_result:
        _render_quiz_results()
        return

    # ── Start screen ──
    if not st.session_state.quiz_started:
        col_l, col_c, col_r = st.columns([1, 2, 1])
        with col_c:
            st.markdown("""
            <div class='card' style='text-align:center;padding:2rem;'>
                <div style='font-size:2.5rem;margin-bottom:0.8rem;'>🧠</div>
                <div style='font-size:1.2rem;font-weight:700;color:#f1f5f9;margin-bottom:0.5rem;'>Ready to test your AI/ML aptitude?</div>
                <div style='color:#64748b;font-size:0.85rem;line-height:1.6;'>
                    This quiz covers <strong style='color:#a5b4fc;'>Python basics</strong>,
                    <strong style='color:#a5b4fc;'>math & statistics</strong>,
                    <strong style='color:#a5b4fc;'>logical reasoning</strong>,
                    <strong style='color:#a5b4fc;'>AI/ML concepts</strong>,
                    and <strong style='color:#a5b4fc;'>interest & motivation</strong>.<br><br>
                    Earn points per correct answer. Score ≥ 50% = AI/ML suitable.
                </div>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Start Quiz", type="primary", use_container_width=True):
                st.session_state.quiz_started = True
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score_result = None
                st.rerun()
        return

    # ── Question form ──
    st.markdown(f"<p style='color:#64748b;text-align:center;'>Answer all questions, then click <strong>Submit Quiz</strong>.</p>", unsafe_allow_html=True)

    with st.form("quiz_form"):
        for q in QUESTIONS:
            qid = q["id"]
            st.markdown(f"""
            <div class='quiz-question-card'>
                <span class='quiz-category-badge'>{q['category']}</span>
                <div style='color:#f1f5f9;font-size:0.95rem;font-weight:500;margin-bottom:0.3rem;'>
                    Q{qid}. {q['question']}
                    <span style='color:#64748b;font-size:0.72rem;margin-left:0.5rem;'>[{q['points']} pt{"s" if q['points'] > 1 else ""}]</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            selected = st.radio(
                f"q{qid}",
                options=q["options"],
                key=f"quiz_q_{qid}",
                label_visibility="collapsed",
                index=st.session_state.quiz_answers.get(qid, 0),
            )
            # Store zero-based index of selected option
            st.session_state.quiz_answers[qid] = q["options"].index(selected)
            st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button("📊 Submit Quiz", use_container_width=True, type="primary")

    if submitted:
        result = calculate_quiz_score(st.session_state.quiz_answers)
        st.session_state.quiz_score_result = result
        st.session_state.quiz_submitted = True
        st.rerun()


def _render_quiz_results():
    res = st.session_state.quiz_score_result
    pct = res["pct"]
    color = res["color"]

    # ── Verdict card ──
    st.markdown(f"""
    <div class='quiz-verdict-card' style='background:linear-gradient(135deg,#0f0f20,#1a0a2e);border:1px solid {color}40;'>
        <div style='font-size:3rem;font-weight:800;color:{color};'>{pct}%</div>
        <div style='font-size:1.3rem;font-weight:700;color:#f1f5f9;margin:0.4rem 0;'>{res["verdict_msg"]}</div>
        <div style='color:#94a3b8;font-size:0.88rem;max-width:500px;margin:0 auto;line-height:1.6;'>{res["verdict_detail"]}</div>
        <div style='color:#64748b;font-size:0.8rem;margin-top:0.8rem;'>
            Score: {res["earned"]} / {res["max_score"]} points
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Category breakdown ──
    with col1:
        st.markdown("#### 📊 Category Breakdown")
        for cat, data in res["categories"].items():
            cat_pct = int((data["earned"] / max(data["possible"], 1)) * 100)
            bar_color = "#10b981" if cat_pct >= 70 else "#f59e0b" if cat_pct >= 40 else "#ef4444"
            st.markdown(f"""
            <div style='margin-bottom:0.8rem;'>
                <div style='display:flex;justify-content:space-between;color:#cbd5e1;font-size:0.82rem;margin-bottom:0.3rem;'>
                    <span>{cat}</span>
                    <span style='color:{bar_color};font-weight:600;'>{data["correct"]}/{data["total"]} correct</span>
                </div>
                <div class='quiz-progress-bar-bg'>
                    <div style='height:8px;border-radius:8px;background:{bar_color};width:{cat_pct}%;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Roles / Suggestions ──
    with col2:
        if res["recommended_roles"]:
            st.markdown("#### 🎯 Recommended Roles")
            for role in res["recommended_roles"]:
                st.markdown(f"""
                <div style='background:rgba(168,85,247,0.08);border:1px solid rgba(168,85,247,0.2);
                            border-radius:10px;padding:0.5rem 0.9rem;margin-bottom:0.4rem;
                            color:#e2d9f3;font-size:0.88rem;'>
                    {role}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("#### 💡 Next Steps")
            for sug in res["suggestions"]:
                st.markdown(f"""
                <div style='background:#0f0f20;border:1px solid #1e1e3a;border-radius:10px;
                            padding:0.5rem 0.9rem;margin-bottom:0.4rem;color:#94a3b8;font-size:0.85rem;'>
                    {sug}
                </div>
                """, unsafe_allow_html=True)

    if res["suggestions"] and res["recommended_roles"]:
        st.markdown("#### 💡 Action Steps")
        for sug in res["suggestions"]:
            st.markdown(f"- {sug}")

    # ── Detailed Q&A review ──
    with st.expander("📋 Review Your Answers"):
        for r in res["results"]:
            css_class = "result-correct" if r["is_correct"] else "result-wrong"
            icon = "✅" if r["is_correct"] else "❌"
            st.markdown(f"""
            <div class='{css_class}'>
                <div style='font-size:0.82rem;font-weight:600;color:#cbd5e1;'>
                    {icon} Q{r["id"]}. {r["question"]}
                </div>
                <div style='font-size:0.78rem;color:#94a3b8;margin-top:0.3rem;'>
                    Your answer: <span style='color:{"#10b981" if r["is_correct"] else "#fca5a5"};'>{r["selected_answer"]}</span>
                    {"" if r["is_correct"] else f" · Correct: <span style='color:#10b981;'>{r['correct_answer']}</span>"}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Retake Quiz", use_container_width=True):
            st.session_state.quiz_started = False
            st.session_state.quiz_submitted = False
            st.session_state.quiz_answers = {}
            st.session_state.quiz_score_result = None
            st.rerun()
    with c2:
        if st.button("📄 Analyze My CV", use_container_width=True):
            nav_goto("home", {"mode": "cv"})
    with c3:
        if st.button("🏠 Home", use_container_width=True):
            nav_goto("home", {"mode": None})


# ============================================================
# ABOUT PAGE
# ============================================================
def render_about():
    col_l, col_c, col_r = st.columns([0.5, 3, 0.5])
    with col_c:
        st.markdown("""
        <div class='about-hero'>
            <div style='font-size:3rem;margin-bottom:0.6rem;'>🚀</div>
            <div style='font-family:"Syne",sans-serif;font-size:2rem;font-weight:800;
                        background:linear-gradient(135deg,#a855f7,#ec4899);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        margin-bottom:0.3rem;'>AI Career Platform</div>
            <div style='color:#64748b;font-size:0.9rem;'>
                AI-powered career matching for data science &amp; AI/ML roles
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Developer card ──
        st.markdown("""
        <div class='card' style='padding:1.6rem;'>
            <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;'>
                Built by
            </div>
            <div style='font-size:1.35rem;font-weight:700;color:#f1f5f9;margin-bottom:0.2rem;'>
                Talha Jobayer Zihan
            </div>
            <div style='font-size:0.85rem;color:#a855f7;margin-bottom:1.2rem;'>
                Researcher &amp; AI/ML Engineer
            </div>

            <div class='contact-row'>
                <span class='contact-icon'>🏛️</span>
                <span>Department of Computer Science &amp; Engineering, RUET</span>
            </div>
            <div class='contact-row'>
                <span class='contact-icon'>📞</span>
                <a href='tel:01721577792' style='color:#cbd5e1;text-decoration:none;'>01721577792</a>
            </div>
            <div class='contact-row'>
                <span class='contact-icon'>✉️</span>
                <a href='mailto:jobayertalha2020@gmail.com' style='color:#cbd5e1;text-decoration:none;'>jobayertalha2020@gmail.com</a>
            </div>
            <div class='contact-row' style='border-bottom:none;'>
                <span class='contact-icon'>🔗</span>
                <a href='https://linkedin.com' target='_blank' style='color:#a5b4fc;text-decoration:none;'>LinkedIn Profile</a>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Tech stack ──
        st.markdown("""
        <div class='card' style='padding:1.6rem;margin-top:0;'>
            <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;'>
                Technology Stack
            </div>
        """, unsafe_allow_html=True)

        tech = ["Streamlit", "LangChain", "Groq LLaMA-3.3-70b", "FAISS Vector Search",
                "HuggingFace Embeddings", "Python 3.10", "pypdf", "sentence-transformers"]
        chips = "".join(f"<span class='tech-pill'>{t}</span>" for t in tech)
        st.markdown(chips + "</div>", unsafe_allow_html=True)

        # ── Pipeline ──
        st.markdown("""
        <div class='card' style='padding:1.6rem;margin-top:0;'>
            <div style='font-size:0.7rem;color:#64748b;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:1rem;'>
                How It Works
            </div>
            <div style='display:flex;flex-direction:column;gap:0.6rem;'>
        """, unsafe_allow_html=True)

        steps = [
            ("1", "#a855f7", "Upload CV (PDF)", "Extracted and preprocessed"),
            ("2", "#ec4899", "FAISS Vector Search", "Cosine similarity against JD knowledge base"),
            ("3", "#f59e0b", "Role Matching", "Top roles with % match scores and salary data"),
            ("4", "#10b981", "LLM Analysis", "Groq LLaMA generates personalized career advice"),
            ("5", "#06b6d4", "Interactive Chat", "Ask follow-up questions about your career"),
        ]
        for num, col, title, desc in steps:
            st.markdown(f"""
            <div style='display:flex;align-items:center;gap:0.8rem;padding:0.5rem 0;border-bottom:1px solid #1a1a30;'>
                <div style='background:{col}20;border:1px solid {col}40;color:{col};
                            width:28px;height:28px;border-radius:50%;display:flex;
                            align-items:center;justify-content:center;font-size:0.75rem;
                            font-weight:700;flex-shrink:0;'>{num}</div>
                <div>
                    <div style='color:#f1f5f9;font-size:0.88rem;font-weight:600;'>{title}</div>
                    <div style='color:#64748b;font-size:0.75rem;'>{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div></div>", unsafe_allow_html=True)

        if st.button("🚀 Go to Home Dashboard", use_container_width=True, type="primary"):
            nav_goto("home", {"mode": None})


# ============================================================
# MAIN ROUTER
# ============================================================
def main():
    # Step 1: Require name entry
    if not st.session_state.name_entered:
        render_welcome()
        return

    # Step 2: Navbar (always shown after login)
    render_navbar()

    # Step 3: Route to page
    page = st.session_state.page
    if page == "home":
        render_home()
    elif page == "analyze":
        render_analysis()
    elif page == "chat":
        render_chat()
    elif page == "jd_result":
        render_jd_result()
    elif page == "career":
        render_career_rec()
    elif page == "quiz":
        render_quiz()
    elif page == "about":
        render_about()
    else:
        # Fallback to home for any unknown page
        st.session_state.page = "home"
        render_home()


if __name__ == "__main__":
    main()
