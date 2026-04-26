"""
app.py — AI Career Platform
Enhanced Theme: Blue-accent branding + Dark/Light mode toggle
"""

import streamlit as st
import tempfile
import os
import re

from agent import extract_cv_text, build_agent, run_agent
from rag import retrieve_context, match_cv_with_jd, score_ai_ml_readiness
from quiz import calculate_interest_score, get_shuffled_questions, reset_quiz

st.set_page_config(
    page_title="AI Career Platform",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    "dark_mode": True,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ============================================================
# THEME VARIABLES
# ============================================================
def get_theme():
    if st.session_state.dark_mode:
        return {
            "bg_primary": "#080c18",
            "bg_secondary": "#0d1117",
            "bg_card": "#111827",
            "bg_card2": "#0f172a",
            "border": "#1e2d45",
            "border_glow": "#1d4ed8",
            "text_primary": "#f0f6ff",
            "text_secondary": "#8ba3c7",
            "text_muted": "#4d6a8a",
            "accent_blue": "#3b82f6",
            "accent_blue_bright": "#60a5fa",
            "accent_blue_dark": "#1d4ed8",
            "accent_cyan": "#06b6d4",
            "accent_green": "#10b981",
            "accent_amber": "#f59e0b",
            "accent_purple": "#8b5cf6",
            "accent_pink": "#ec4899",
            "input_bg": "#0d1117",
            "sidebar_bg": "linear-gradient(180deg, #060a14 0%, #080c18 60%, #060912 100%)",
            "glow_blue": "rgba(59, 130, 246, 0.25)",
            "glow_cyan": "rgba(6, 182, 212, 0.15)",
            "hero_gradient": "linear-gradient(135deg, #0d1117 0%, #0a1628 50%, #050d1a 100%)",
            "card_hover_shadow": "0 8px 40px rgba(59, 130, 246, 0.2), 0 2px 12px rgba(0,0,0,0.5)",
            "mode_icon": "☀️",
            "mode_label": "Light Mode",
        }
    else:
        return {
            "bg_primary": "#eef2fb",
            "bg_secondary": "#e2e9f8",
            "bg_card": "#ffffff",
            "bg_card2": "#f4f7ff",
            "border": "#b8cdf0",
            "border_glow": "#2563eb",
            "text_primary": "#0c1526",
            "text_secondary": "#253d63",
            "text_muted": "#5a749e",
            "accent_blue": "#2563eb",
            "accent_blue_bright": "#1d4ed8",
            "accent_blue_dark": "#1e3a8a",
            "accent_cyan": "#0e7490",
            "accent_green": "#047857",
            "accent_amber": "#b45309",
            "accent_purple": "#6d28d9",
            "accent_pink": "#be185d",
            "input_bg": "#ffffff",
            "sidebar_bg": "linear-gradient(180deg, #ffffff 0%, #f8faff 100%)",
            "glow_blue": "rgba(37, 99, 235, 0.14)",
            "glow_cyan": "rgba(14, 116, 144, 0.1)",
            "hero_gradient": "linear-gradient(135deg, #dbeafe 0%, #eff6ff 50%, #e0ecff 100%)",
            "card_hover_shadow": "0 8px 40px rgba(37, 99, 235, 0.18), 0 2px 12px rgba(0,0,0,0.08)",
            "mode_icon": "🌙",
            "mode_label": "Dark Mode",
        }

T = get_theme()

# ============================================================
# INJECT CSS
# ============================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@400;500;600;700;800&display=swap');

:root {{
    --bg-primary: {T['bg_primary']};
    --bg-secondary: {T['bg_secondary']};
    --bg-card: {T['bg_card']};
    --bg-card2: {T['bg_card2']};
    --border: {T['border']};
    --border-glow: {T['border_glow']};
    --text-primary: {T['text_primary']};
    --text-secondary: {T['text_secondary']};
    --text-muted: {T['text_muted']};
    --accent-blue: {T['accent_blue']};
    --accent-blue-bright: {T['accent_blue_bright']};
    --accent-blue-dark: {T['accent_blue_dark']};
    --accent-cyan: {T['accent_cyan']};
    --accent-green: {T['accent_green']};
    --accent-amber: {T['accent_amber']};
    --accent-purple: {T['accent_purple']};
    --accent-pink: {T['accent_pink']};
    --input-bg: {T['input_bg']};
    --glow-blue: {T['glow_blue']};
    --glow-cyan: {T['glow_cyan']};
    --card-hover-shadow: {T['card_hover_shadow']};
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}
html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif !important; }}
.stApp {{ background: var(--bg-primary) !important; min-height: 100vh; }}
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.stDeployButton {{display: none;}}

[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}}
[data-testid="stSidebarCollapseButton"] {{
    display: flex !important;
    background: {T['accent_blue']} !important;
    border: 2px solid {T['accent_blue']} !important;
    border-radius: 8px !important;
    margin: 0.5rem !important;
    z-index: 999999 !important;
}}
[data-testid="stSidebarCollapseButton"] svg {{ fill: #ffffff !important; }}

.sidebar-brand {{
    text-align: center; padding: 1rem 0.5rem 0.6rem;
    border-bottom: 1px solid var(--border); margin-bottom: 0.6rem; position: relative;
}}
.sidebar-brand-logo {{ font-size: 2rem; display: block; margin-bottom: 0.2rem; filter: drop-shadow(0 0 8px var(--glow-blue)); }}
.sidebar-brand-name {{ font-family: 'Syne', sans-serif !important; font-size: 0.72rem; font-weight: 700; color: var(--text-primary); letter-spacing: 0.05em; text-transform: uppercase; }}
.sidebar-brand-name span {{ color: var(--accent-blue); }}

.user-chip {{
    background: var(--bg-card2); border: 1px solid var(--border); border-radius: 30px;
    padding: 0.3rem 0.7rem; margin-bottom: 0.7rem; text-align: center;
    font-size: 0.7rem; color: var(--text-secondary); font-weight: 500;
    box-shadow: inset 0 0 10px var(--glow-blue);
}}

[data-testid="stSidebar"] .stButton > button {{
    border-radius: 10px !important; padding: 0.6rem 1rem !important; margin-bottom: 0.28rem !important;
    transition: all 0.22s ease !important; white-space: normal !important; line-height: 1.4 !important;
    height: auto !important; min-height: 44px !important; font-weight: 700 !important;
    font-size: 0.73rem !important; font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important; width: 100% !important;
    background: var(--bg-card2) !important; border: 2px solid var(--accent-blue) !important; color: var(--accent-blue) !important;
}}
[data-testid="stSidebar"] .stButton > button *, [data-testid="stSidebar"] .stButton > button p,
[data-testid="stSidebar"] .stButton > button span, [data-testid="stSidebar"] .stButton > button div {{
    color: inherit !important; font-size: inherit !important; font-weight: inherit !important; background: transparent !important;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    transform: translateX(4px) !important; background: var(--accent-blue) !important;
    color: #ffffff !important; border-color: var(--accent-blue) !important; box-shadow: 0 4px 20px var(--glow-blue) !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--accent-blue-dark), var(--accent-blue)) !important;
    color: #ffffff !important; border: 2px solid transparent !important; box-shadow: 0 4px 18px var(--glow-blue) !important;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    color: #ffffff !important; transform: translateX(2px) !important;
}}

.main-content {{ padding: 0.5rem 2rem 2rem 2rem; }}
.main-header {{ margin-bottom: 1.5rem; padding-bottom: 0.8rem; position: relative; }}
.main-header::after {{
    content: ''; display: block; margin-top: 0.7rem; width: 100%; height: 1px;
    background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-cyan) 8%, var(--border) 30%, transparent 100%);
    border-radius: 2px;
}}
.main-header h1 {{ font-family: 'Syne', sans-serif !important; font-size: 1.85rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.3rem; letter-spacing: -0.02em; }}
.main-header h1 .hl {{ background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
.main-header p {{ color: var(--text-secondary); font-size: 0.88rem; }}

.feature-card {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 18px;
    padding: 1.5rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); height: 100%; position: relative; overflow: hidden;
}}
.feature-card::before {{
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-blue)); opacity: 0; transition: opacity 0.3s ease;
}}
.feature-card:hover {{ transform: translateY(-6px); border-color: var(--border-glow); box-shadow: var(--card-hover-shadow); }}
.feature-card:hover::before {{ opacity: 1; }}
.feature-icon {{ font-size: 2.2rem; margin-bottom: 0.8rem; display: block; filter: drop-shadow(0 0 6px var(--glow-blue)); }}
.feature-title {{ font-family: 'Syne', sans-serif !important; font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.5rem; letter-spacing: -0.01em; }}
.feature-desc {{ color: var(--text-secondary); font-size: 0.8rem; line-height: 1.55; }}
.feature-tags {{ margin-top: 0.85rem; display: flex; gap: 0.4rem; flex-wrap: wrap; }}
.feature-tag {{ background: var(--glow-blue); border: 1px solid var(--border-glow); border-radius: 20px; padding: 0.2rem 0.65rem; font-size: 0.62rem; color: var(--accent-blue-bright); font-weight: 600; letter-spacing: 0.03em; }}

.result-card {{
    background: var(--bg-card); border: 1px solid var(--border); border-radius: 18px;
    padding: 1.6rem; margin-bottom: 1.5rem; position: relative; overflow: hidden;
}}
.result-card::after {{
    content: ''; position: absolute; top: -40px; right: -40px; width: 120px; height: 120px;
    background: radial-gradient(circle, var(--glow-blue) 0%, transparent 70%); pointer-events: none;
}}

.skill-chip {{ display: inline-block; background: var(--bg-card2); border: 1px solid var(--border); border-radius: 20px; padding: 0.22rem 0.65rem; font-size: 0.68rem; color: var(--text-secondary); margin: 0.2rem; font-weight: 500; }}
.gap-chip {{ border-color: #7f1d1d; color: #fca5a5; background: rgba(127,29,29,0.15); }}

.stTextInput > div > div > input, .stTextArea > div > div > textarea {{
    background: var(--input-bg) !important; border: 1px solid var(--border) !important; border-radius: 12px !important;
    color: var(--text-primary) !important; font-family: 'Space Grotesk', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}
.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {{
    border-color: var(--accent-blue) !important; box-shadow: 0 0 0 3px var(--glow-blue) !important;
}}

.contact-card, .about-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 18px; padding: 1.6rem; }}
.contact-item {{ display: flex; align-items: center; gap: 1rem; padding: 0.85rem 0; border-bottom: 1px solid var(--border); }}
.contact-item:last-child {{ border-bottom: none; }}
.contact-icon {{ font-size: 1.2rem; min-width: 40px; color: var(--accent-blue-bright); }}
.contact-label {{ font-weight: 600; color: var(--text-primary); min-width: 90px; font-size: 0.84rem; }}
.contact-value {{ color: var(--text-secondary); font-size: 0.84rem; }}
.contact-link {{ color: var(--accent-blue-bright); text-decoration: none; }}
.contact-link:hover {{ color: var(--accent-cyan); text-decoration: underline; }}
.social-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-top: 1rem; }}
.social-card {{ background: var(--bg-card2); border: 1px solid var(--border); border-radius: 14px; padding: 0.9rem; text-align: center; transition: all 0.3s ease; text-decoration: none; }}
.social-card:hover {{ background: var(--glow-blue); border-color: var(--accent-blue); transform: translateY(-3px); box-shadow: 0 8px 24px var(--glow-blue); }}
.social-icon {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
.social-name {{ color: var(--text-secondary); font-size: 0.75rem; font-weight: 600; }}
.interest-tag {{ display: inline-block; background: var(--glow-blue); border: 1px solid var(--border-glow); border-radius: 20px; padding: 0.25rem 0.85rem; font-size: 0.7rem; color: var(--accent-blue-bright); margin: 0.2rem; font-weight: 600; }}
.tech-stack {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.8rem; }}
.tech-pill {{ background: var(--bg-card2); border: 1px solid var(--border); border-radius: 20px; padding: 0.22rem 0.65rem; font-size: 0.65rem; color: var(--text-muted); font-weight: 500; }}
.profile-header {{ text-align: center; margin-bottom: 1.5rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }}
.profile-icon {{ font-size: 3.2rem; margin-bottom: 0.4rem; }}
.profile-name {{ font-family: 'Syne', sans-serif !important; font-size: 1.35rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.2rem; }}
.profile-title {{ color: var(--accent-blue-bright); font-size: 0.8rem; font-weight: 600; }}
.section-header {{ font-family: 'Syne', sans-serif !important; font-size: 0.88rem; font-weight: 700; color: var(--text-primary); margin: 1rem 0 0.5rem 0; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); letter-spacing: 0.02em; }}

.welcome-container {{ max-width: 400px; margin: 50px auto; text-align: center; }}
.welcome-badge {{ display: inline-block; background: var(--glow-blue); border: 1px solid var(--border-glow); border-radius: 30px; padding: 0.3rem 1rem; font-size: 0.65rem; color: var(--accent-blue-bright); font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.8rem; }}
.welcome-title {{ font-family: 'Syne', sans-serif !important; font-size: 2rem; font-weight: 800; margin-bottom: 0.4rem; color: var(--text-primary); line-height: 1.1; letter-spacing: -0.03em; }}
.welcome-gradient {{ background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }}
.welcome-subtitle {{ color: var(--text-secondary); margin-bottom: 1.5rem; font-size: 0.82rem; line-height: 1.5; }}
.welcome-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 20px; padding: 1.4rem; box-shadow: 0 20px 60px var(--glow-blue); position: relative; overflow: hidden; }}
.welcome-card::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan)); }}

.quiz-start-container {{ max-width: 400px; margin: 30px auto; text-align: center; }}
.quiz-start-card {{ background: var(--bg-card); border: 1px solid var(--border); border-radius: 18px; padding: 1.4rem; box-shadow: 0 8px 32px var(--glow-blue); }}
.quiz-start-icon {{ font-size: 2rem; margin-bottom: 0.5rem; }}
.quiz-start-title {{ font-family: 'Syne', sans-serif !important; font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.3rem; }}
.quiz-start-desc {{ color: var(--text-secondary); font-size: 0.72rem; margin-bottom: 0.8rem; }}
.quiz-question {{ background: var(--bg-card); border: 1px solid var(--border); border-left: 3px solid var(--accent-blue); border-radius: 14px; padding: 1.1rem 1.2rem; margin-bottom: 0.9rem; transition: border-color 0.2s ease; }}
.quiz-question:hover {{ border-color: var(--accent-blue-bright); border-left-color: var(--accent-cyan); }}
.quiz-question-text {{ font-weight: 600; color: var(--text-primary); margin-bottom: 0.8rem; font-size: 0.88rem; line-height: 1.4; }}

.stButton > button {{ font-family: 'Space Grotesk', sans-serif !important; font-weight: 600 !important; border-radius: 10px !important; transition: all 0.25s ease !important; }}
.stButton > button[kind="primary"] {{ background: linear-gradient(135deg, var(--accent-blue-dark), var(--accent-blue)) !important; border: none !important; color: #ffffff !important; box-shadow: 0 4px 20px var(--glow-blue) !important; }}
.stButton > button[kind="primary"]:hover {{ background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important; box-shadow: 0 8px 32px var(--glow-blue) !important; transform: translateY(-2px) !important; }}

.stMarkdown, .stMarkdown p, .element-container .stMarkdown p {{ color: var(--text-secondary) !important; }}
.stMarkdown strong, .stMarkdown b {{ color: var(--text-primary) !important; }}
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5 {{ color: var(--text-primary) !important; font-family: 'Syne', sans-serif !important; }}
.stMarkdown li {{ color: var(--text-secondary) !important; }}
.stMarkdown code {{ background: var(--bg-card2) !important; color: var(--accent-blue) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; padding: 0.1rem 0.3rem !important; }}
label {{ color: var(--text-secondary) !important; font-family: 'Space Grotesk', sans-serif !important; }}
.stRadio label {{ color: var(--text-primary) !important; }}
hr {{ border-color: var(--border) !important; }}
.stCaption {{ color: var(--text-muted) !important; font-size: 0.68rem !important; }}

[data-testid="stFileUploader"] {{ background: var(--bg-card) !important; border: 2px dashed var(--border-glow) !important; border-radius: 14px !important; transition: border-color 0.2s ease !important; }}
[data-testid="stFileUploader"]:hover {{ border-color: var(--accent-blue) !important; }}
[data-testid="stFileUploader"] > div {{ background: var(--bg-card) !important; color: var(--text-primary) !important; }}
[data-testid="stFileUploaderDropzone"] {{ background: var(--bg-card2) !important; border: none !important; border-radius: 12px !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] {{ color: var(--text-secondary) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] svg {{ fill: var(--accent-blue) !important; stroke: var(--accent-blue) !important; }}
[data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] p, [data-testid="stFileUploaderDropzoneInstructions"] small {{ color: var(--text-secondary) !important; }}
[data-testid="stFileUploaderDropzone"] button {{ background: var(--accent-blue) !important; color: #ffffff !important; border: none !important; border-radius: 8px !important; }}

.stSpinner > div {{ border-top-color: var(--accent-blue) !important; }}
.stSuccess {{ background: rgba(16,185,129,0.1) !important; border-color: var(--accent-green) !important; color: #34d399 !important; }}
.stWarning {{ background: rgba(245,158,11,0.1) !important; border-color: var(--accent-amber) !important; }}
.stError {{ background: rgba(239,68,68,0.1) !important; border-color: #ef4444 !important; }}

.welcome-card .stButton > button, .welcome-card button[kind="primaryFormSubmit"],
[data-testid="stForm"] button[kind="primaryFormSubmit"], [data-testid="stForm"] .stButton > button {{
    background: linear-gradient(135deg, var(--accent-blue-dark), var(--accent-blue)) !important;
    border: none !important; color: #ffffff !important; box-shadow: 0 4px 20px var(--glow-blue) !important;
    border-radius: 10px !important; font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important;
}}
[data-testid="stForm"] button[kind="primaryFormSubmit"]:hover, [data-testid="stForm"] .stButton > button:hover {{
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    box-shadow: 0 8px 32px var(--glow-blue) !important; transform: translateY(-2px) !important;
}}

.stButton > button[kind="secondary"] {{ background: var(--bg-card2) !important; border: 2px solid var(--accent-blue) !important; color: var(--accent-blue) !important; font-weight: 600 !important; }}
.stButton > button[kind="secondary"]:hover {{ background: var(--accent-blue) !important; color: #ffffff !important; border-color: var(--accent-blue) !important; box-shadow: 0 4px 16px var(--glow-blue) !important; transform: translateY(-2px) !important; }}

@keyframes glowPulse {{
    0%, 100% {{ box-shadow: 0 0 10px var(--glow-blue); }}
    50% {{ box-shadow: 0 0 24px var(--glow-blue), 0 0 40px var(--glow-cyan); }}
}}
.hero-badge {{ animation: glowPulse 3s ease-in-out infinite; }}
</style>
""", unsafe_allow_html=True)


# ============================================================
# NAVIGATION
# ============================================================
def nav_goto(page):
    if st.session_state.page != page:
        st.session_state.page = page
        st.rerun()


def sign_out():
    for key in ["candidate_name", "name_entered", "cv_text", "agent",
                "analysis_raw", "retrieved", "jd_match_result", "quiz_responses", "quiz_result"]:
        st.session_state[key] = _defaults[key]
    st.session_state.page = "home"
    st.rerun()


def render_sidebar():
    T = get_theme()
    name = st.session_state.candidate_name
    first = name.split()[0] if name else "Guest"
    current_page = st.session_state.page

    sidebar_bg = T['sidebar_bg']
    sidebar_border = T['border']
    sb_extra = f"""
    <style>
    [data-testid="stSidebar"] > div:first-child {{
        background: {sidebar_bg} !important;
        border-right: 1px solid {sidebar_border} !important;
    }}
    .mode-toggle-wrap .stButton > button {{
        background: {T['bg_card']} !important; border: 1.5px solid {T['border']} !important;
        color: {T['text_secondary']} !important; min-height: 36px !important; font-size: 0.68rem !important;
        text-align: center !important; justify-content: center !important; margin-bottom: 0.5rem !important;
    }}
    .mode-toggle-wrap .stButton > button:hover {{
        background: {T['glow_blue']} !important; border-color: {T['accent_blue']} !important;
        color: {T['accent_blue']} !important; transform: none !important; box-shadow: none !important;
    }}
    .signout-wrap .stButton > button {{
        background: transparent !important; border: 1.5px solid {T['text_muted']} !important;
        color: {T['text_muted']} !important; margin-top: 0.3rem !important;
    }}
    .signout-wrap .stButton > button:hover {{
        background: #ef4444 !important; border-color: #ef4444 !important; color: #ffffff !important;
        transform: none !important; box-shadow: 0 4px 14px rgba(239,68,68,0.3) !important;
    }}
    </style>
    """
    st.markdown(sb_extra, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown(f"""
        <div class="sidebar-brand">
            <span class="sidebar-brand-logo">🎯</span>
            <div class="sidebar-brand-name">AI <span>Career</span> Platform</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="user-chip">👤 &nbsp;{first}</div>', unsafe_allow_html=True)

        st.markdown('<div class="mode-toggle-wrap">', unsafe_allow_html=True)
        mode_label = f"{T['mode_icon']}  {T['mode_label']}"
        if st.button(mode_label, key="dark_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='margin-bottom:0.25rem'></div>", unsafe_allow_html=True)

        nav_items = [
            ("🏠  Home", "home"),
            ("📄  Analyze CV", "analyze"),
            ("🎯  JD Match", "jd_match"),
            ("🧠  Quiz", "quiz"),
            ("ℹ️  About", "about"),
            ("📞  Contact", "contact")
        ]

        for label, page_key in nav_items:
            if current_page == page_key:
                st.button(label, key=f"nav_{page_key}", use_container_width=True, type="primary")
            else:
                if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                    nav_goto(page_key)

        st.markdown('<div class="signout-wrap">', unsafe_allow_html=True)
        if st.button("⏻  Sign Out", key="signout_btn", use_container_width=True):
            sign_out()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="text-align:center; margin-top:0.6rem; font-size:0.6rem; color: {T['text_muted']};">
            © 2025 AI Career Platform
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# WELCOME SCREEN
# ============================================================
def render_welcome():
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-badge hero-badge">✦ Powered by AI &amp; FAISS</div>
            <div class="welcome-title">AI <span class="welcome-gradient">Career</span><br>Platform</div>
            <p class="welcome-subtitle">Your AI-powered career companion.<br>Match your CV to the best AI/ML roles in seconds.</p>
            <div class="welcome-card">
                <div style="font-size:0.82rem; font-weight:600; margin-bottom:0.9rem; color:var(--text-primary);">
                    👋 Welcome! What's your name?
                </div>
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

        st.markdown("</div></div>", unsafe_allow_html=True)


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
            <span class="feature-icon">📄</span>
            <div class="feature-title">Analyze My CV</div>
            <div class="feature-desc">Upload your CV and get matched with the best AI/ML roles using FAISS vector search.</div>
            <div class="feature-tags">
                <span class="feature-tag">Role Match</span>
                <span class="feature-tag">Skills</span>
                <span class="feature-tag">Salary</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📄 Analyze CV", key="home_cv", use_container_width=True, type="primary"):
            nav_goto("analyze")

    with col2:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🎯</span>
            <div class="feature-title">Match with JD</div>
            <div class="feature-desc">Paste a job description and see how well your CV aligns with real market data.</div>
            <div class="feature-tags">
                <span class="feature-tag">Match %</span>
                <span class="feature-tag">Skill Gaps</span>
                <span class="feature-tag">Companies</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 JD Match", key="home_jd", use_container_width=True, type="primary"):
            nav_goto("jd_match")

    with col3:
        st.markdown("""
        <div class="feature-card">
            <span class="feature-icon">🧠</span>
            <div class="feature-title">Interest Quiz</div>
            <div class="feature-desc">Take a quiz to discover which AI/ML roles match your thinking style and interests.</div>
            <div class="feature-tags">
                <span class="feature-tag">Interest Score</span>
                <span class="feature-tag">Role Fit</span>
                <span class="feature-tag">Career Path</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🧠 Take Quiz", key="home_quiz", use_container_width=True, type="primary"):
            nav_goto("quiz")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# ANALYZE PAGE
# ============================================================
def render_analyze():
    st.markdown("""
    <div class="main-content">
        <div class="main-header">
            <h1>📄 <span class="hl">CV</span> Analysis</h1>
            <p>Upload your CV to get personalized career recommendations based on real job market data.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded = st.file_uploader("Upload CV (PDF)", type=["pdf"], label_visibility="collapsed")
        if uploaded and st.button("🚀 Start Analysis", use_container_width=True, type="primary"):
            with st.spinner("Analyzing your CV..."):
                try:
                    tmp_path = None
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    cv_text = extract_cv_text(tmp_path)
                    os.unlink(tmp_path)

                    if not cv_text or len(cv_text.strip()) < 20:
                        st.error("⚠️ Could not extract text from your PDF. Please ensure it's a text-based PDF (not scanned image).")
                    else:
                        st.session_state.cv_text = cv_text
                        st.session_state.retrieved = retrieve_context(cv_text, "", k=5)
                        st.session_state.agent = build_agent(cv_text, "", st.session_state.candidate_name)
                        st.session_state.chat_history = []
                        st.session_state.analysis_raw = run_agent(
                            st.session_state.agent,
                            "Analyse this CV. Follow tags: TOP_ROLE, MATCH_PCT, WHY_RIGHT, SKILL_GAPS, RESUME_ADD, CAREER_PATH"
                        )
                        st.rerun()
                except Exception as e:
                    if tmp_path and os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    st.error(f"⚠️ Error reading PDF: {str(e)[:200]}. Please try a different PDF file.")

    if st.session_state.retrieved:
        render_analysis_results()

    st.markdown("---")
    if st.button("← Back to Home", key="back_home_analyze"):
        nav_goto("home")

    st.markdown("</div>", unsafe_allow_html=True)


def render_analysis_results():
    retrieved = st.session_state.retrieved
    top_match = retrieved.get("top_match", {})
    readiness = retrieved.get("readiness", {})
    analysis_raw = st.session_state.get("analysis_raw", "")
    has_ai_exp = retrieved.get("has_ai_experience", False)
    T = get_theme()

    faiss_pct   = top_match.get("match_pct", 0)
    ready_score = readiness.get("total_score", 0)
    unified_score = round(faiss_pct * 0.55 + ready_score * 0.45)
    unified_score = max(5, min(97, unified_score))
    level         = readiness.get("level", "Developing")
    rec           = readiness.get("recommendation", "")
    role_title    = top_match.get("title", top_match.get("role", "AI Professional"))
    company       = top_match.get("company", "")
    location      = top_match.get("location", "Dhaka")

    if unified_score >= 80:
        tier_color  = "#10b981"
        tier_label  = "Very Strong — AI/ML Field Ready"
        tier_icon   = "🚀"
        tier_bg     = "rgba(16,185,129,0.10)"
        tier_border = "rgba(16,185,129,0.35)"
        bar_gradient= "linear-gradient(90deg,#059669,#10b981,#34d399)"
        verdict     = "Your profile is highly competitive for AI/ML roles in Bangladesh's market."
    elif unified_score >= 60:
        tier_color  = "#3b82f6"
        tier_label  = "Strong — Needs More Polishing"
        tier_icon   = "💪"
        tier_bg     = "rgba(59,130,246,0.10)"
        tier_border = "rgba(59,130,246,0.30)"
        bar_gradient= "linear-gradient(90deg,#1d4ed8,#3b82f6,#60a5fa)"
        verdict     = "Solid foundation with targeted gaps. Closing 2–3 skill areas will unlock mid-level roles."
    elif unified_score >= 40:
        tier_color  = "#f59e0b"
        tier_label  = "Developing — Explore & Validate Interest"
        tier_icon   = "🔍"
        tier_bg     = "rgba(245,158,11,0.10)"
        tier_border = "rgba(245,158,11,0.30)"
        bar_gradient= "linear-gradient(90deg,#d97706,#f59e0b,#fbbf24)"
        verdict     = "You have foundational interest. Build 2–3 focused projects and earn a recognized certificate to progress."
    else:
        tier_color  = "#ef4444"
        tier_label  = "Beginner — Consider Broader Exploration"
        tier_icon   = "⚡"
        tier_bg     = "rgba(239,68,68,0.10)"
        tier_border = "rgba(239,68,68,0.25)"
        bar_gradient= "linear-gradient(90deg,#b91c1c,#ef4444,#f87171)"
        verdict     = "Limited AI/ML signal detected in your CV. Consider whether this field aligns with your core interests before investing heavily."

    bar_pct = unified_score

    # ── HERO: Unified Score Card ──
    st.markdown(f"""
    <div class="result-card" style="text-align:center; padding:2rem 1.5rem 1.5rem;">
        <div style="font-size:0.68rem; font-weight:700; color:{tier_color}; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;">
            {tier_icon} AI/ML Profile Score
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:4rem; font-weight:900; line-height:1;
                    background:linear-gradient(135deg,{tier_color},{tier_color}99);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
                    filter:drop-shadow(0 0 20px {tier_color}40); margin-bottom:0.5rem;">
            {unified_score}%
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:var(--text-primary); margin-bottom:0.2rem;">
            {role_title}
        </div>
        <div style="font-size:0.75rem; color:var(--text-muted); margin-bottom:1.2rem;">
            {company}{' · ' + location if company else location}
        </div>
        <div style="background:var(--bg-card2); border:1px solid var(--border); border-radius:30px; height:10px; overflow:hidden; margin:0 2rem 0.8rem;">
            <div style="width:{bar_pct}%; height:100%; background:{bar_gradient}; border-radius:30px;
                        transition:width 0.8s ease; box-shadow:0 0 8px {tier_color}60;"></div>
        </div>
        <div style="display:inline-block; padding:0.35rem 1.1rem; background:{tier_bg}; border:1px solid {tier_border};
                    border-radius:30px; font-size:0.72rem; font-weight:700; color:{tier_color}; letter-spacing:0.04em;">
            {tier_label}
        </div>
        <div style="margin-top:0.7rem; font-size:0.78rem; color:var(--text-secondary); max-width:480px; margin-left:auto; margin-right:auto; line-height:1.5;">
            {verdict}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scoreboard / Field Fit Scale ──
    st.markdown(f"""
    <div class="result-card">
        <div style="font-weight:700; color:var(--text-primary); margin-bottom:1rem; font-size:0.88rem; font-family:'Syne',sans-serif;">
            📊 AI/ML Field Readiness Scale
        </div>
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:0.5rem;">
            <div style="padding:0.7rem 0.5rem; border-radius:10px; text-align:center;
                        background:{'rgba(239,68,68,0.15)' if unified_score < 40 else 'var(--bg-card2)'};
                        border:1.5px solid {'#ef4444' if unified_score < 40 else 'var(--border)'};
                        opacity:{'1' if unified_score < 40 else '0.55'};">
                <div style="font-size:1.1rem; margin-bottom:0.2rem;">⚡</div>
                <div style="font-size:0.62rem; font-weight:700; color:{'#ef4444' if unified_score < 40 else 'var(--text-muted)'}; margin-bottom:0.2rem;">0–39%</div>
                <div style="font-size:0.6rem; color:var(--text-muted); line-height:1.3;">Consider other fields first</div>
            </div>
            <div style="padding:0.7rem 0.5rem; border-radius:10px; text-align:center;
                        background:{'rgba(245,158,11,0.15)' if 40 <= unified_score < 60 else 'var(--bg-card2)'};
                        border:1.5px solid {'#f59e0b' if 40 <= unified_score < 60 else 'var(--border)'};
                        opacity:{'1' if 40 <= unified_score < 60 else '0.55'};">
                <div style="font-size:1.1rem; margin-bottom:0.2rem;">🔍</div>
                <div style="font-size:0.62rem; font-weight:700; color:{'#f59e0b' if 40 <= unified_score < 60 else 'var(--text-muted)'}; margin-bottom:0.2rem;">40–59%</div>
                <div style="font-size:0.6rem; color:var(--text-muted); line-height:1.3;">Interested — explore &amp; validate</div>
            </div>
            <div style="padding:0.7rem 0.5rem; border-radius:10px; text-align:center;
                        background:{'rgba(59,130,246,0.15)' if 60 <= unified_score < 80 else 'var(--bg-card2)'};
                        border:1.5px solid {'#3b82f6' if 60 <= unified_score < 80 else 'var(--border)'};
                        opacity:{'1' if 60 <= unified_score < 80 else '0.55'};">
                <div style="font-size:1.1rem; margin-bottom:0.2rem;">💪</div>
                <div style="font-size:0.62rem; font-weight:700; color:{'#3b82f6' if 60 <= unified_score < 80 else 'var(--text-muted)'}; margin-bottom:0.2rem;">60–79%</div>
                <div style="font-size:0.6rem; color:var(--text-muted); line-height:1.3;">Strong — needs polishing</div>
            </div>
            <div style="padding:0.7rem 0.5rem; border-radius:10px; text-align:center;
                        background:{'rgba(16,185,129,0.15)' if unified_score >= 80 else 'var(--bg-card2)'};
                        border:1.5px solid {'#10b981' if unified_score >= 80 else 'var(--border)'};
                        opacity:{'1' if unified_score >= 80 else '0.55'};">
                <div style="font-size:1.1rem; margin-bottom:0.2rem;">🚀</div>
                <div style="font-size:0.62rem; font-weight:700; color:{'#10b981' if unified_score >= 80 else 'var(--text-muted)'}; margin-bottom:0.2rem;">80–100%</div>
                <div style="font-size:0.6rem; color:var(--text-muted); line-height:1.3;">Very strong — AI/ML ready</div>
            </div>
        </div>
        <div style="margin-top:1rem; padding-top:0.8rem; border-top:1px solid var(--border);">
            <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
                <span style="font-size:0.68rem; color:var(--text-muted); font-weight:600;">Score Breakdown</span>
                <span style="font-size:0.68rem; color:var(--text-muted);">Role Match {faiss_pct}% · Readiness {round(ready_score)}%</span>
            </div>
            <div style="display:flex; gap:0.3rem; height:6px; border-radius:4px; overflow:hidden;">
                <div style="width:{faiss_pct*0.55}%; background:var(--accent-blue); border-radius:4px;"></div>
                <div style="width:{ready_score*0.45}%; background:var(--accent-cyan); border-radius:4px;"></div>
            </div>
            <div style="display:flex; gap:1rem; margin-top:0.35rem;">
                <div style="display:flex; align-items:center; gap:0.3rem;">
                    <div style="width:8px; height:8px; background:var(--accent-blue); border-radius:2px;"></div>
                    <span style="font-size:0.62rem; color:var(--text-muted);">Role Match (55%)</span>
                </div>
                <div style="display:flex; align-items:center; gap:0.3rem;">
                    <div style="width:8px; height:8px; background:var(--accent-cyan); border-radius:2px;"></div>
                    <span style="font-size:0.62rem; color:var(--text-muted);">CV Readiness (45%)</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if rec:
        st.markdown(f"""
        <div style="padding:0.8rem 1rem; background:{tier_bg}; border:1px solid {tier_border};
                    border-left:4px solid {tier_color}; border-radius:10px; margin-bottom:1rem;">
            <div style="display:flex; align-items:flex-start; gap:0.6rem;">
                <span style="font-size:1rem; line-height:1;">{tier_icon}</span>
                <div>
                    <div style="font-size:0.72rem; font-weight:700; color:{tier_color}; margin-bottom:0.2rem; letter-spacing:0.04em; text-transform:uppercase;">
                        {level} · Recommendation
                    </div>
                    <div style="font-size:0.82rem; color:var(--text-primary); line-height:1.5;">{rec}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if not has_ai_exp:
        st.markdown("""
        <div style="padding:0.7rem 1rem; background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.25);
                    border-left:4px solid var(--accent-blue); border-radius:10px; margin-bottom:1rem;">
            <span style="font-size:0.82rem; color:var(--text-primary);">
                💡 <strong>Tip:</strong> No AI/ML work experience detected. Adding internships or research projects will significantly boost your profile score.
            </span>
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        gaps = retrieved.get("skill_gaps", [])
        if gaps:
            chips = "".join(f"<span class='skill-chip gap-chip'>{g}</span>" for g in gaps[:6])
            st.markdown(f"""
            <div class="result-card">
                <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.6rem; font-size:0.85rem;">❌ Skill Gaps</div>
                {chips}
            </div>
            """, unsafe_allow_html=True)

    with col2:
        recs = retrieved.get("resume_skills", [])
        if recs:
            chips = "".join(f"<span class='skill-chip'>+ {sk}</span>" for sk in recs[:6])
            st.markdown(f"""
            <div class="result-card">
                <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.6rem; font-size:0.85rem;">➕ Add to Resume</div>
                {chips}
            </div>
            """, unsafe_allow_html=True)

    roles_html_parts = []
    for r in retrieved.get("all_matches", [])[:4]:
        sal_min = r.get("salary_min", 0)
        sal_max = r.get("salary_max", 0)
        sal_str = f"৳{sal_min:,}–৳{sal_max:,}/mo" if sal_min else ""
        rtitle  = r.get("title", r.get("role", "Role"))
        rcomp   = r.get("company", "Various")
        rpct    = r.get("match_pct", 0)
        roles_html_parts.append(
            f'<div style="margin-bottom:0.7rem; padding:0.7rem 1rem; background:var(--bg-card2); border:1px solid var(--border);'
            f'border-left:3px solid var(--accent-blue); border-radius:10px; display:flex; justify-content:space-between; align-items:center;">'
            f'<div><strong style="color:var(--text-primary);">{rtitle}</strong>'
            f'<span style="color:var(--text-muted); font-size:0.75rem;"> · {rcomp}</span></div>'
            f'<div style="text-align:right;">'
            f'<span style="color:var(--accent-blue-bright); font-weight:700; font-size:0.85rem;">{rpct}%</span>'
            f'<div style="color:var(--text-muted); font-size:0.7rem;">{sal_str}</div>'
            f'</div></div>'
        )
    roles_inner = "".join(roles_html_parts)
    st.markdown(f"""
    <div class="result-card">
        <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.8rem; font-size:0.9rem; font-family:'Syne',sans-serif;">🗺️ Matched Roles</div>
        {roles_inner}
    </div>
    """, unsafe_allow_html=True)

    if analysis_raw:
        def parse_tag(text, tag):
            pattern = rf'{tag}:\s*(.*?)(?=\n[A-Z_]+:|$)'
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""

        def parse_list_tag(text, tag):
            pattern = rf'{tag}:\s*\n(.*?)(?=\n[A-Z_]+:|$)'
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if not m:
                return []
            raw = m.group(1).strip()
            return [re.sub(r'^[-•*]\s*', '', l.strip())
                    for l in raw.split('\n')
                    if l.strip() and re.match(r'^[-•*]', l.strip())]

        top_role    = parse_tag(analysis_raw, "TOP_ROLE")
        match_pct   = parse_tag(analysis_raw, "MATCH_PCT")
        why_right   = parse_tag(analysis_raw, "WHY_RIGHT")
        runner_up   = parse_tag(analysis_raw, "RUNNER_UP")
        runner_why  = parse_tag(analysis_raw, "RUNNER_UP_WHY")
        next_steps  = parse_list_tag(analysis_raw, "NEXT_STEPS")
        skill_gaps  = parse_list_tag(analysis_raw, "SKILL_GAPS")
        resume_add  = parse_list_tag(analysis_raw, "RESUME_ADD")
        career_path = parse_list_tag(analysis_raw, "CAREER_PATH")

        if top_role or why_right:
            hero_title = ""
            if top_role:
                pct_badge = (f"&nbsp;<span style='color:var(--accent-blue);font-size:0.85rem;'>({match_pct}% match)</span>"
                             if match_pct else "")
                hero_title = (f"<div style='font-family:Syne,sans-serif; font-size:1.05rem; font-weight:800;"
                              f"color:var(--text-primary); margin-bottom:0.4rem;'>🏆 {top_role}{pct_badge}</div>")
            why_html = (f"<div style='color:var(--text-secondary); font-size:0.84rem; line-height:1.6;'>{why_right}</div>"
                        if why_right else "")
            st.markdown(f"""
            <div class="result-card" style="border-left:4px solid var(--accent-blue);">
                <div style="font-size:0.68rem; font-weight:700; color:var(--text-muted); letter-spacing:0.1em; text-transform:uppercase; margin-bottom:0.5rem;">🤖 AI Career Analysis</div>
                {hero_title}
                {why_html}
            </div>
            """, unsafe_allow_html=True)

        # ── FIX 1: Next Steps — build all HTML in one string, render once ──
        if next_steps:
            _ns_items = "".join(
                '<div style="display:flex; gap:0.8rem; align-items:flex-start; margin-bottom:0.6rem;'
                'padding:0.5rem 0.8rem; background:var(--bg-card2); border-radius:8px; border:1px solid var(--border);">'
                f'<span style="background:var(--accent-blue); color:#fff; border-radius:50%; width:22px; height:22px;'
                'min-width:22px; display:flex; align-items:center; justify-content:center;'
                f'font-size:0.65rem; font-weight:700;">{_i + 1}</span>'
                f'<span style="color:var(--text-primary); font-size:0.82rem; line-height:1.5;">{_s}</span>'
                '</div>'
                for _i, _s in enumerate(next_steps[:4])
            )
            st.markdown(
                f'<div class="result-card">'
                f'<div style="font-weight:700; color:var(--text-primary); margin-bottom:0.7rem; font-size:0.88rem; font-family:\'Syne\',sans-serif;">🚀 Next Steps</div>'
                f'{_ns_items}</div>',
                unsafe_allow_html=True
            )

        # ── FIX 2: Skill Gaps — build all HTML in one string, render once ──
        if skill_gaps:
            def _gap_inner(s):
                if ':' in s:
                    gl, gc = s.split(':', 1)
                    return f"<strong style='color:#ef4444;'>⚡ {gl.strip()}</strong>: {gc.strip()}"
                return f"<strong style='color:#ef4444;'>⚡ {s}</strong>"

            _gap_items = "".join(
                '<div style="padding:0.5rem 0.8rem; background:rgba(239,68,68,0.06); border:1px solid rgba(239,68,68,0.2);'
                'border-left:3px solid #ef4444; border-radius:8px; margin-bottom:0.5rem;'
                f'color:var(--text-primary); font-size:0.82rem; line-height:1.5;">{_gap_inner(_s)}</div>'
                for _s in skill_gaps[:5]
            )
            st.markdown(
                f'<div class="result-card">'
                f'<div style="font-weight:700; color:var(--text-primary); margin-bottom:0.7rem; font-size:0.88rem; font-family:\'Syne\',sans-serif;">🔍 Skill Gaps to Close</div>'
                f'{_gap_items}</div>',
                unsafe_allow_html=True
            )

        # ── FIX 3: Resume Additions — build all HTML in one string, render once ──
        if resume_add:
            _ra_items = "".join(
                '<div style="padding:0.5rem 0.8rem; background:rgba(16,185,129,0.06); border:1px solid rgba(16,185,129,0.2);'
                'border-left:3px solid var(--accent-green); border-radius:8px; margin-bottom:0.5rem;'
                f'color:var(--text-primary); font-size:0.82rem; line-height:1.5;">✅ {_s}</div>'
                for _s in resume_add[:5]
            )
            st.markdown(
                f'<div class="result-card">'
                f'<div style="font-weight:700; color:var(--text-primary); margin-bottom:0.7rem; font-size:0.88rem; font-family:\'Syne\',sans-serif;">📝 Resume Additions</div>'
                f'{_ra_items}</div>',
                unsafe_allow_html=True
            )

        # ── FIX 4: Career Path — build all HTML in one string, render once ──
        if career_path:
            n = min(len(career_path), 3)
            step_meta = [
                ("#3b82f6", "🎯", "Short-term"),
                ("#8b5cf6", "📈", "Mid-term"),
                ("#10b981", "🏆", "Long-term"),
            ]
            _cp_steps = []
            for _i, _s in enumerate(career_path[:3]):
                _dot_color, _step_icon, _ = step_meta[_i] if _i < len(step_meta) else ("#3b82f6", "•", "Step")
                _connector = (
                    f'<div style="width:2px; height:24px; background:var(--border); margin:2px auto;"></div>'
                    if _i < n - 1 else ""
                )
                if ':' in _s:
                    _raw_label, _raw_content = _s.split(':', 1)
                    _label   = _raw_label.strip()
                    _content = _raw_content.strip()
                else:
                    _label   = f"Step {_i + 1}"
                    _content = _s.strip()

                _cp_steps.append(
                    f'<div style="display:flex; gap:0.8rem; align-items:flex-start; margin-bottom:0.2rem;">'
                    f'<div style="min-width:14px; display:flex; flex-direction:column; align-items:center; padding-top:4px;">'
                    f'<div style="width:14px; height:14px; background:{_dot_color}; border-radius:50%;'
                    f'box-shadow:0 0 6px {_dot_color}60; flex-shrink:0;"></div>'
                    f'{_connector}'
                    f'</div>'
                    f'<div style="flex:1; padding:0.6rem 0.9rem; background:var(--bg-card2);'
                    f'border:1px solid var(--border); border-left:3px solid {_dot_color};'
                    f'border-radius:8px; margin-bottom:0.5rem;">'
                    f'<div style="font-size:0.7rem; font-weight:700; color:{_dot_color};'
                    f'letter-spacing:0.05em; text-transform:uppercase; margin-bottom:0.25rem;">'
                    f'{_step_icon} {_label}</div>'
                    f'<div style="color:var(--text-primary); font-size:0.82rem; line-height:1.55;">{_content}</div>'
                    f'</div></div>'
                )
            _cp_html = "".join(_cp_steps)
            st.markdown(
                f'<div class="result-card">'
                f'<div style="font-weight:700; color:var(--text-primary); margin-bottom:0.9rem;'
                f'font-size:0.88rem; font-family:\'Syne\',sans-serif;">🗺️ Career Path</div>'
                f'{_cp_html}</div>',
                unsafe_allow_html=True
            )

        if runner_up:
            ru_why_html = (f"<div style='color:var(--text-secondary); font-size:0.82rem; line-height:1.5;'>{runner_why}</div>"
                           if runner_why else "")
            st.markdown(f"""
            <div class="result-card" style="border-left:4px solid var(--accent-purple);">
                <div style="font-size:0.68rem; font-weight:700; color:var(--text-muted); letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:0.4rem;">🥈 Runner-Up Role</div>
                <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:700;
                            color:var(--text-primary); margin-bottom:0.3rem;">{runner_up}</div>
                {ru_why_html}
            </div>
            """, unsafe_allow_html=True)

    # AI Chat
    st.markdown("""
    <div class="result-card">
        <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.6rem; font-size:0.9rem; font-family:'Syne',sans-serif;">💬 Ask Your Career Advisor</div>
        <div style="color:var(--text-secondary); font-size:0.78rem; margin-bottom:0.8rem;">Ask anything about your CV, career path, skill gaps, or job search strategy.</div>
    """, unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history[-6:]:
        role_icon = "🧑" if msg["role"] == "user" else "🤖"
        bg = "var(--glow-blue)" if msg["role"] == "user" else "var(--bg-card2)"
        st.markdown(f"""
        <div style="padding:0.6rem 0.9rem; background:{bg}; border:1px solid var(--border); border-radius:10px; margin-bottom:0.4rem; font-size:0.82rem; color:var(--text-primary);">
            <strong>{role_icon}</strong> {msg["content"]}
        </div>
        """, unsafe_allow_html=True)

    chat_input = st.text_input("Your question", placeholder="e.g. What skills should I learn first?", key="chat_input", label_visibility="collapsed")
    if st.button("Send →", key="chat_send", type="primary"):
        if chat_input and st.session_state.agent:
            with st.spinner("Thinking..."):
                reply = run_agent(st.session_state.agent, chat_input)
            st.session_state.chat_history.append({"role": "user", "content": chat_input})
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# JD MATCH PAGE
# ============================================================

def render_jd_match():
    st.markdown("""
    <div class="main-content">
        <div class="main-header">
            <h1>🎯 JD <span class="hl">Matching</span></h1>
            <p>See how well your CV matches a specific job description.</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        uploaded = st.file_uploader("Your CV (PDF)", type=["pdf"], key="jd_cv_upload")
    with col2:
        jd_text = st.text_area("Job Description", height=200, placeholder="Paste the full job description here...")

    if uploaded and jd_text and st.button("🎯 Calculate Match", use_container_width=True, type="primary"):
        with st.spinner("Calculating match and analyzing..."):
            try:
                tmp_path = None
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded.read())
                    tmp_path = tmp.name
                cv_text = extract_cv_text(tmp_path)
                os.unlink(tmp_path)

                if not cv_text or len(cv_text.strip()) < 20:
                    st.error("⚠️ Could not extract text from your PDF. Please ensure it's a text-based PDF (not scanned image).")
                else:
                    st.session_state.cv_text = cv_text
                    st.session_state.jd_text_for_match = jd_text
                    # Initialize agent if not already done
                    if not st.session_state.agent:
                        st.session_state.agent = build_agent(cv_text, jd_text, st.session_state.candidate_name)
                    st.session_state.jd_match_result = match_cv_with_jd(cv_text, jd_text)
                    st.rerun()
            except Exception as e:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                st.error(f"⚠️ Error reading PDF: {str(e)[:200]}. Please try a different PDF file.")

    if st.session_state.jd_match_result:
        render_jd_match_results()

    st.markdown("---")
    if st.button("← Back to Home", key="back_home_jd"):
        nav_goto("home")

    st.markdown("</div>", unsafe_allow_html=True)

def render_jd_match_results():
    result = st.session_state.jd_match_result
    pct = result.get("match_pct", 0)
    cv_text = st.session_state.get("cv_text", "")
    jd_text = st.session_state.get("jd_text_for_match", "")
    
    # Get detailed analysis from the agent if available
    if cv_text and jd_text and st.session_state.agent:
        with st.spinner("Analyzing JD match in detail..."):
            jd_analysis = run_agent(
                st.session_state.agent,
                f"""Analyze this job description match. My CV matches this JD at {pct}%. 
                JD: {jd_text[:1500]}
                
                Provide structured analysis with these tags:
                JD_MATCH_STRENGTH: [strengths of my CV for this role]
                JD_GAPS: [what's missing]
                JD_ACTION_PLAN: [3 specific things to do]
                JD_VERDICT: [Should I apply? 1-2 sentences]
                """
            )
    else:
        jd_analysis = ""

    # Determine match quality colors and messages
    if pct < 30:
        color, status, icon, message = "#ef4444", "Low Match", "⚠️", "This role may require significant skill development before applying."
        bg_intensity = "rgba(239,68,68,0.08)"
        border_color = "rgba(239,68,68,0.3)"
    elif pct < 60:
        color, status, icon, message = "#f59e0b", "Partial Match", "📌", "You have some relevant skills. Focus on closing the identified gaps."
        bg_intensity = "rgba(245,158,11,0.08)"
        border_color = "rgba(245,158,11,0.3)"
    elif pct < 80:
        color, status, icon, message = "#10b981", "Good Match", "✅", "Strong alignment! Tailor your application to highlight matching skills."
        bg_intensity = "rgba(16,185,129,0.08)"
        border_color = "rgba(16,185,129,0.3)"
    else:
        color, status, icon, message = "#3b82f6", "Excellent Match!", "🎯", "You're highly qualified. Apply confidently and customize your cover letter."
        bg_intensity = "rgba(59,130,246,0.08)"
        border_color = "rgba(59,130,246,0.3)"

    # Main score card with detailed context
    st.markdown(f"""
    <div class="result-card" style="text-align:center; padding:2rem 1.5rem;">
        <div style="font-size:0.68rem; font-weight:700; color:{color}; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;">
            {icon} JD Match Analysis
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:4rem; font-weight:900; line-height:1;
                    background:linear-gradient(135deg,{color},{color}99);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
                    filter:drop-shadow(0 0 20px {color}40); margin-bottom:0.5rem;">
            {pct}%
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:var(--text-primary); margin-bottom:0.5rem;">
            {status}
        </div>
        <div style="color:var(--text-secondary); font-size:0.85rem; margin-bottom:1rem; max-width:500px; margin-left:auto; margin-right:auto;">
            {message}
        </div>
        <div style="background:var(--bg-card2); border:1px solid var(--border); border-radius:30px; height:8px; overflow:hidden; margin:0 2rem 1rem;">
            <div style="width:{pct}%; height:100%; background:linear-gradient(90deg, {color}, {color}cc); border-radius:30px;
                        transition:width 0.8s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Skill Match Breakdown
    if cv_text and result.get("similar_roles"):
        cv_lower = cv_text.lower()
        all_required_skills = []
        for role in result["similar_roles"][:3]:
            all_required_skills.extend(role.get("skills", []))
        unique_skills = list(dict.fromkeys(all_required_skills))[:12]
        
        matched_skills = []
        missing_skills = []
        partial_skills = []
        
        for skill in unique_skills:
            skill_lower = skill.lower()
            if skill_lower in cv_lower:
                matched_skills.append(skill)
            elif any(word in cv_lower for word in skill_lower.split() if len(word) > 3):
                partial_skills.append(skill)
            else:
                missing_skills.append(skill)
        
        match_ratio = len(matched_skills) / max(len(unique_skills), 1)
        
        st.markdown(f"""
        <div class="result-card">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:1rem; font-size:0.9rem; font-family:'Syne',sans-serif;">
                🔍 Skill Match Breakdown
            </div>
            <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:1rem; margin-bottom:1rem;">
                <div style="text-align:center; padding:0.5rem; background:rgba(16,185,129,0.1); border-radius:10px;">
                    <div style="font-size:1.3rem; font-weight:800; color:#10b981;">{len(matched_skills)}</div>
                    <div style="font-size:0.68rem; color:var(--text-muted);">Matched</div>
                </div>
                <div style="text-align:center; padding:0.5rem; background:rgba(245,158,11,0.1); border-radius:10px;">
                    <div style="font-size:1.3rem; font-weight:800; color:#f59e0b;">{len(partial_skills)}</div>
                    <div style="font-size:0.68rem; color:var(--text-muted);">Partial</div>
                </div>
                <div style="text-align:center; padding:0.5rem; background:rgba(239,68,68,0.1); border-radius:10px;">
                    <div style="font-size:1.3rem; font-weight:800; color:#ef4444;">{len(missing_skills)}</div>
                    <div style="font-size:0.68rem; color:var(--text-muted);">Missing</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        if matched_skills:
            skills_chips = "".join(f"<span class='skill-chip' style='background:rgba(16,185,129,0.15); border-color:#10b981;'>✓ {s}</span>" for s in matched_skills[:8])
            st.markdown(f"""
            <div style="margin-top:0.8rem;">
                <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); margin-bottom:0.4rem;">✅ Skills You Have</div>
                <div style="display:flex; flex-wrap:wrap; gap:0.3rem;">{skills_chips}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if missing_skills:
            missing_chips = "".join(f"<span class='skill-chip gap-chip'>✗ {s}</span>" for s in missing_skills[:8])
            st.markdown(f"""
            <div style="margin-top:0.8rem;">
                <div style="font-size:0.72rem; font-weight:600; color:var(--text-secondary); margin-bottom:0.4rem;">❌ Skills to Develop</div>
                <div style="display:flex; flex-wrap:wrap; gap:0.3rem;">{missing_chips}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Similar Roles with context
    if result.get("similar_roles"):
        st.markdown("""
        <div class="result-card">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.8rem; font-size:0.9rem; font-family:'Syne',sans-serif;">
                📊 Similar Roles You May Qualify For
            </div>
        """, unsafe_allow_html=True)
        
        for i, role in enumerate(result["similar_roles"][:4]):
            role_title = role.get('title', role.get('role', 'Role'))
            role_company = role.get('company', 'Various')
            role_score = role.get('match_pct', 0)
            sal_min = role.get('salary_min', 0)
            sal_max = role.get('salary_max', 0)
            sal_str = f"৳{sal_min:,}–৳{sal_max:,}/mo" if sal_min else "Salary not specified"
            
            # Determine if this role is better match than current JD
            comparison = ""
            if role_score > pct + 10:
                comparison = f"<span style='color:#10b981; font-size:0.65rem;'>▲ Better fit than this JD</span>"
            elif role_score > pct:
                comparison = f"<span style='color:#f59e0b; font-size:0.65rem;'>▲ Slightly better fit</span>"
            elif role_score < pct - 10:
                comparison = f"<span style='color:#ef4444; font-size:0.65rem;'>▼ This JD fits you better</span>"
            
            st.markdown(f"""
            <div style="margin-bottom:0.7rem; padding:0.8rem 1rem; background:var(--bg-card2); border:1px solid var(--border); 
                        border-left:3px solid {'#10b981' if role_score > pct else '#3b82f6'}; border-radius:10px;">
                <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;">
                    <div>
                        <strong style="color:var(--text-primary); font-size:0.88rem;">{role_title}</strong>
                        <span style="color:var(--text-muted); font-size:0.72rem;"> · {role_company}</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="color:{'#10b981' if role_score > pct else '#3b82f6'}; font-weight:700; font-size:0.85rem;">{role_score}% match</span>
                        <div style="color:var(--text-muted); font-size:0.65rem;">{sal_str}</div>
                    </div>
                </div>
                {f"<div style='margin-top:0.4rem;'>{comparison}</div>" if comparison else ""}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    # Actionable Recommendations
    st.markdown(f"""
    <div class="result-card" style="border-left:4px solid {color};">
        <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.6rem; font-size:0.88rem; font-family:'Syne',sans-serif;">
            💡 Actionable Recommendations
        </div>
        <div style="color:var(--text-primary); font-size:0.85rem; line-height:1.6; margin-bottom:0.8rem;">
            {_generate_jd_recommendation(pct, result)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # LLM Analysis if available
    if jd_analysis:
        st.markdown("""
        <div class="result-card">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.7rem; font-size:0.88rem; font-family:'Syne',sans-serif;">
                🤖 AI Career Advisor Insights
            </div>
        """, unsafe_allow_html=True)
        
        # Parse and display LLM analysis
        def parse_jd_tag(text, tag):
            pattern = rf'{tag}:\s*(.*?)(?=\n[A-Z_]+:|$)'
            m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            return m.group(1).strip() if m else ""
        
        strengths = parse_jd_tag(jd_analysis, "JD_MATCH_STRENGTH")
        gaps = parse_jd_tag(jd_analysis, "JD_GAPS")
        action_plan = parse_jd_tag(jd_analysis, "JD_ACTION_PLAN")
        verdict = parse_jd_tag(jd_analysis, "JD_VERDICT")
        
        if strengths:
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="font-size:0.72rem; font-weight:700; color:var(--accent-green); margin-bottom:0.3rem;">✨ Your Strengths</div>
                <div style="color:var(--text-secondary); font-size:0.82rem; line-height:1.5;">{strengths}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if gaps:
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="font-size:0.72rem; font-weight:700; color:var(--accent-amber); margin-bottom:0.3rem;">📋 Gaps to Address</div>
                <div style="color:var(--text-secondary); font-size:0.82rem; line-height:1.5;">{gaps}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if action_plan:
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="font-size:0.72rem; font-weight:700; color:var(--accent-blue); margin-bottom:0.3rem;">🚀 Next Steps</div>
                <div style="color:var(--text-secondary); font-size:0.82rem; line-height:1.5;">{action_plan}</div>
            </div>
            """, unsafe_allow_html=True)
        
        if verdict:
            st.markdown(f"""
            <div style="margin-top:0.5rem; padding-top:0.5rem; border-top:1px solid var(--border);">
                <div style="font-size:0.72rem; font-weight:700; color:{color}; margin-bottom:0.3rem;">🎯 Verdict</div>
                <div style="color:var(--text-primary); font-size:0.85rem; line-height:1.5;">{verdict}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Back button
    st.markdown("---")
    if st.button("← Back to Home", key="back_home_jd"):
        nav_goto("home")

def _generate_jd_recommendation(pct: int, result: dict) -> str:
    """Generate contextual recommendations based on match score."""
    if pct < 30:
        return """**Hold off on applying** — Focus on building foundational skills first. 
        Complete 2-3 relevant projects and earn certifications in the required technologies. 
        Consider applying for internships or junior roles to gain experience."""
    elif pct < 60:
        return """**Apply with tailored approach** — Your application needs customization. 
        Highlight the skills you do have prominently. Create a portfolio project addressing their specific needs. 
        Use the skill gaps above as a learning roadmap for the next 2-3 months."""
    elif pct < 80:
        return """**Strong candidate — Apply now!** Customize your CV to emphasize matching skills. 
        Write a targeted cover letter addressing how your experience solves their problems. 
        Prepare specific examples from your past work that relate to their requirements."""
    else:
        return """**Excellent fit — Priority application!** You're highly qualified. 
        Apply immediately and follow up within a week. Prepare for interviews by reviewing their tech stack. 
        Consider reaching out to current employees for referral — you have strong alignment.""" 



# ============================================================
# QUIZ PAGE
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
    "quiz_started": False,  # ADD THIS LINE
    "dark_mode": True,
}


def render_quiz_results():
    result = st.session_state.quiz_result
    score = result["score"]
    max_score = result["max_score"]
    pct = result["pct"]
    level = result["level"]
    alignment = result["alignment"]
    color = result["color"]
    icon = result["icon"]
    rec = result["recommendation"]

    # Score bar display
    bar_width = (score / max_score) * 100
    
    st.markdown(f"""
    <div class="result-card" style="text-align:center;">
        <div style="font-size:0.68rem; font-weight:700; color:{color}; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.5rem;">
            {icon} AI/ML Interest Alignment
        </div>
        <div style="font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:900; line-height:1;
                    background:linear-gradient(135deg,{color},{color}99);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
                    filter:drop-shadow(0 0 20px {color}40); margin-bottom:0.3rem;">
            {score}/{max_score}
        </div>
        <div style="font-size:0.9rem; font-weight:700; color:{color}; margin-bottom:0.5rem;">{pct}% · {level}</div>
        <div style="background:var(--bg-card2); border:1px solid var(--border); border-radius:30px; height:8px; overflow:hidden; margin:0.5rem 2rem;">
            <div style="width:{bar_width}%; height:100%; background:linear-gradient(90deg, {color}, {color}cc); border-radius:30px;"></div>
        </div>
        <div style="margin-top:1rem; padding:1rem; background:{color}10; border-radius:12px;">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.3rem;">{rec["verdict"]}</div>
            <div style="color:var(--text-secondary); font-size:0.85rem;">{rec["message"]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="result-card">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.6rem;">🎯 Recommended Roles</div>
        """, unsafe_allow_html=True)
        for role in rec["roles"]:
            st.markdown(f"- {role}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="result-card">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.6rem;">🚀 Next Steps</div>
        """, unsafe_allow_html=True)
        for step in rec["next_steps"]:
            st.markdown(f"- {step}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Category breakdown
    if result.get("category_scores"):
        st.markdown("""
        <div class="result-card">
            <div style="font-weight:700; color:var(--text-primary); margin-bottom:0.8rem;">📊 Interest Breakdown</div>
        """, unsafe_allow_html=True)
        
        for cat, data in result["category_scores"].items():
            cat_pct = int((data["score"] / max(data["max_possible"], 1)) * 100)
            st.markdown(f"""
            <div style="margin-bottom:0.8rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.2rem;">
                    <span style="font-size:0.72rem; color:var(--text-secondary);">{cat}</span>
                    <span style="font-size:0.72rem; color:var(--text-secondary);">{cat_pct}%</span>
                </div>
                <div style="background:var(--bg-card2); border-radius:10px; height:4px; overflow:hidden;">
                    <div style="width:{cat_pct}%; height:100%; background:linear-gradient(90deg, {result['color']}, {result['color']}99); border-radius:10px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)




# ============================================================
# ABOUT PAGE
# ============================================================
def render_about():
    st.markdown("""
    <div class="main-content">
        <div class="main-header">
            <h1>ℹ️ <span class="hl">About</span></h1>
            <p>Learn more about the developer and this platform.</p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-card">
        <div class="profile-header">
            <div class="profile-icon">👨‍💻</div>
            <div class="profile-name">Talha Jobayer Zihan</div>
            <div class="profile-title">Researcher &amp; AI/ML Engineer</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🔬 Research Interests</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="margin-bottom:1rem;">
        <span class="interest-tag">Natural Language Processing (NLP)</span>
        <span class="interest-tag">Computer Vision</span>
        <span class="interest-tag">Cyber Security</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎓 Academic Affiliation</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:var(--text-secondary); margin-bottom:1rem; font-size:0.85rem; line-height:1.6;">
        Department of Computer Science &amp; Engineering<br>
        Rajshahi University of Engineering &amp; Technology (RUET)
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🚀 About This Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color:var(--text-secondary); line-height:1.6; margin-bottom:1rem; font-size:0.85rem;">
        AI Career Platform is an intelligent career matching system designed to help job seekers in Bangladesh
        find the best AI/ML roles based on their CV content, skills, and career preferences.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">✨ Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <ul style="color:var(--text-secondary); line-height:1.8; margin-bottom:1rem; font-size:0.82rem; padding-left:1rem;">
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
    if st.button("← Back to Home", key="back_home_about"):
        nav_goto("home")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# CONTACT PAGE
# ============================================================
def render_contact():
    st.markdown("""
    <div class="main-content">
        <div class="main-header">
            <h1>📞 <span class="hl">Contact</span></h1>
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

    <div class="section-header" style="margin-top:1.5rem; text-align:center;">Connect With Me</div>
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
    if st.button("← Back to Home", key="back_home_contact"):
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
