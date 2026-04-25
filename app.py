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
from quiz import QUESTIONS, calculate_interest_score

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

html, body, [class*="css"] {{
    font-family: 'Space Grotesk', sans-serif !important;
}}

.stApp {{
    background: var(--bg-primary) !important;
    min-height: 100vh;
}}

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
.stDeployButton {{display: none;}}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {{
    background: {T['sidebar_bg']} !important;
    border-right: 1px solid var(--border) !important;
    padding-top: 0 !important;
}}

[data-testid="stSidebarCollapseButton"] {{
    display: flex !important;
    background: var(--bg-card) !important;
    border-radius: 8px !important;
    margin: 0.5rem !important;
    z-index: 999999 !important;
}}
[data-testid="stSidebarCollapseButton"] svg {{
    fill: var(--accent-blue) !important;
}}

/* Brand header in sidebar */
.sidebar-brand {{
    text-align: center;
    padding: 1rem 0.5rem 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.6rem;
    position: relative;
}}

.sidebar-brand-logo {{
    font-size: 2rem;
    display: block;
    margin-bottom: 0.2rem;
    filter: drop-shadow(0 0 8px var(--glow-blue));
}}

.sidebar-brand-name {{
    font-family: 'Syne', sans-serif !important;
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}}

.sidebar-brand-name span {{
    color: var(--accent-blue);
}}

/* Sidebar user chip */
.user-chip {{
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 30px;
    padding: 0.3rem 0.7rem;
    margin-bottom: 0.7rem;
    text-align: center;
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-weight: 500;
    box-shadow: inset 0 0 10px var(--glow-blue);
}}

/* NAV BUTTONS */
[data-testid="stSidebar"] .stButton > button {{
    border-radius: 10px !important;
    padding: 0.55rem 0.4rem !important;
    margin-bottom: 0.3rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    white-space: normal !important;
    line-height: 1.3 !important;
    height: auto !important;
    min-height: 52px !important;
    font-weight: 700 !important;
    font-size: 0.7rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: 0.02em !important;
    background: var(--bg-card2) !important;
    position: relative;
    overflow: hidden;
    width: 100% !important;
}}

[data-testid="stSidebar"] .stButton > button p {{
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
}}

[data-testid="stSidebar"] .stButton > button:hover {{
    transform: translateX(3px) !important;
    box-shadow: 0 4px 16px var(--glow-blue) !important;
}}

/* Nav button color overrides */
[data-testid="stSidebar"] .stButton > button[key="nav_home"] {{
    border: 2px solid var(--accent-blue) !important;
    color: var(--accent-blue) !important;
}}
[data-testid="stSidebar"] .stButton > button[key="nav_home"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"][key="nav_home"] {{
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    border: 2px solid var(--accent-blue) !important;
    box-shadow: 0 0 16px rgba(37,99,235,0.35) !important;
}}

[data-testid="stSidebar"] .stButton > button[key="nav_analyze"] {{
    border: 2px solid var(--accent-green) !important;
    color: var(--accent-green) !important;
}}
[data-testid="stSidebar"] .stButton > button[key="nav_analyze"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"][key="nav_analyze"] {{
    background: var(--accent-green) !important;
    color: #ffffff !important;
    border: 2px solid var(--accent-green) !important;
    box-shadow: 0 0 16px rgba(16,185,129,0.3) !important;
}}

[data-testid="stSidebar"] .stButton > button[key="nav_jd_match"] {{
    border: 2px solid var(--accent-amber) !important;
    color: var(--accent-amber) !important;
}}
[data-testid="stSidebar"] .stButton > button[key="nav_jd_match"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"][key="nav_jd_match"] {{
    background: var(--accent-amber) !important; color: #ffffff !important;
    border: 2px solid var(--accent-amber) !important;
    box-shadow: 0 0 16px rgba(245,158,11,0.3) !important;
}}

[data-testid="stSidebar"] .stButton > button[key="nav_quiz"] {{
    border: 2px solid var(--accent-purple) !important; color: var(--accent-purple) !important;
}}
[data-testid="stSidebar"] .stButton > button[key="nav_quiz"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"][key="nav_quiz"] {{
    background: var(--accent-purple) !important; color: #ffffff !important;
    border: 2px solid var(--accent-purple) !important;
    box-shadow: 0 0 16px rgba(139,92,246,0.3) !important;
}}

[data-testid="stSidebar"] .stButton > button[key="nav_about"] {{
    border: 2px solid var(--accent-pink) !important; color: var(--accent-pink) !important;
}}
[data-testid="stSidebar"] .stButton > button[key="nav_about"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"][key="nav_about"] {{
    background: var(--accent-pink) !important; color: #ffffff !important;
    border: 2px solid var(--accent-pink) !important;
    box-shadow: 0 0 16px rgba(236,72,153,0.3) !important;
}}

[data-testid="stSidebar"] .stButton > button[key="nav_contact"] {{
    border: 2px solid var(--accent-cyan) !important; color: var(--accent-cyan) !important;
}}
[data-testid="stSidebar"] .stButton > button[key="nav_contact"]:hover,
[data-testid="stSidebar"] .stButton > button[kind="primary"][key="nav_contact"] {{
    background: var(--accent-cyan) !important; color: #ffffff !important;
    border: 2px solid var(--accent-cyan) !important;
    box-shadow: 0 0 16px rgba(6,182,212,0.3) !important;
}}

[data-testid="stSidebar"] .stButton > button[key="signout_btn"] {{
    border: 2px solid var(--text-muted) !important; color: var(--text-muted) !important;
    background: transparent !important;
}}
[data-testid="stSidebar"] .stButton > button[key="signout_btn"]:hover {{
    background: #ef4444 !important; color: #ffffff !important;
    border: 2px solid #ef4444 !important;
    box-shadow: 0 0 14px rgba(239,68,68,0.3) !important;
}}

/* Dark mode toggle button */
[data-testid="stSidebar"] .stButton > button[key="dark_toggle"] {{
    border: 2px solid var(--border) !important;
    color: var(--text-secondary) !important;
    background: var(--bg-card) !important;
    font-size: 0.68rem !important;
    min-height: 38px !important;
    margin-bottom: 0.5rem !important;
    font-weight: 600 !important;
}}
[data-testid="stSidebar"] .stButton > button[key="dark_toggle"]:hover {{
    border-color: var(--accent-blue) !important;
    color: var(--accent-blue) !important;
    background: var(--glow-blue) !important;
    box-shadow: 0 0 12px var(--glow-blue) !important;
    transform: none !important;
}}

/* ─── MAIN CONTENT ─── */
.main-content {{
    padding: 0.5rem 2rem 2rem 2rem;
}}

/* Page header with gradient underline */
.main-header {{
    margin-bottom: 1.5rem;
    padding-bottom: 0.8rem;
    position: relative;
}}

.main-header::after {{
    content: '';
    display: block;
    margin-top: 0.7rem;
    width: 100%;
    height: 1px;
    background: linear-gradient(90deg, var(--accent-blue) 0%, var(--accent-cyan) 8%, var(--border) 30%, transparent 100%);
    border-radius: 2px;
}}

.main-header h1 {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.85rem;
    font-weight: 800;
    color: var(--text-primary);
    margin-bottom: 0.3rem;
    letter-spacing: -0.02em;
}}

.main-header h1 .hl {{
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.main-header p {{
    color: var(--text-secondary);
    font-size: 0.88rem;
}}

/* ─── FEATURE CARDS ─── */
.feature-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    position: relative;
    overflow: hidden;
}}

.feature-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan), var(--accent-blue));
    opacity: 0;
    transition: opacity 0.3s ease;
}}

.feature-card:hover {{
    transform: translateY(-6px);
    border-color: var(--border-glow);
    box-shadow: var(--card-hover-shadow);
}}

.feature-card:hover::before {{
    opacity: 1;
}}

.feature-icon {{
    font-size: 2.2rem;
    margin-bottom: 0.8rem;
    display: block;
    filter: drop-shadow(0 0 6px var(--glow-blue));
}}

.feature-title {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -0.01em;
}}

.feature-desc {{
    color: var(--text-secondary);
    font-size: 0.8rem;
    line-height: 1.55;
}}

.feature-tags {{
    margin-top: 0.85rem;
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
}}

.feature-tag {{
    background: var(--glow-blue);
    border: 1px solid var(--border-glow);
    border-radius: 20px;
    padding: 0.2rem 0.65rem;
    font-size: 0.62rem;
    color: var(--accent-blue-bright);
    font-weight: 600;
    letter-spacing: 0.03em;
}}

/* ─── RESULT CARDS ─── */
.result-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}}

.result-card::after {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 120px; height: 120px;
    background: radial-gradient(circle, var(--glow-blue) 0%, transparent 70%);
    pointer-events: none;
}}

.match-score {{
    text-align: center;
    padding: 1rem;
}}

.match-percentage {{
    font-family: 'Syne', sans-serif !important;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    filter: drop-shadow(0 0 20px var(--glow-blue));
}}

/* ─── SKILL CHIPS ─── */
.skill-chip {{
    display: inline-block;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.22rem 0.65rem;
    font-size: 0.68rem;
    color: var(--text-secondary);
    margin: 0.2rem;
    font-weight: 500;
}}

.gap-chip {{
    border-color: #7f1d1d;
    color: #fca5a5;
    background: rgba(127,29,29,0.15);
}}

/* ─── FORM ELEMENTS ─── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
    background: var(--input-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
}}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 3px var(--glow-blue) !important;
}}

/* ─── CONTACT & ABOUT ─── */
.contact-card, .about-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.6rem;
}}

.contact-item {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.85rem 0;
    border-bottom: 1px solid var(--border);
}}
.contact-item:last-child {{ border-bottom: none; }}

.contact-icon {{ font-size: 1.2rem; min-width: 40px; color: var(--accent-blue-bright); }}
.contact-label {{ font-weight: 600; color: var(--text-primary); min-width: 90px; font-size: 0.84rem; }}
.contact-value {{ color: var(--text-secondary); font-size: 0.84rem; }}
.contact-link {{ color: var(--accent-blue-bright); text-decoration: none; }}
.contact-link:hover {{ color: var(--accent-cyan); text-decoration: underline; }}

.social-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-top: 1rem;
}}

.social-card {{
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem;
    text-align: center;
    transition: all 0.3s ease;
    text-decoration: none;
}}

.social-card:hover {{
    background: var(--glow-blue);
    border-color: var(--accent-blue);
    transform: translateY(-3px);
    box-shadow: 0 8px 24px var(--glow-blue);
}}

.social-icon {{ font-size: 1.4rem; margin-bottom: 0.3rem; }}
.social-name {{ color: var(--text-secondary); font-size: 0.75rem; font-weight: 600; }}

.interest-tag {{
    display: inline-block;
    background: var(--glow-blue);
    border: 1px solid var(--border-glow);
    border-radius: 20px;
    padding: 0.25rem 0.85rem;
    font-size: 0.7rem;
    color: var(--accent-blue-bright);
    margin: 0.2rem;
    font-weight: 600;
}}

.tech-stack {{ display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.8rem; }}

.tech-pill {{
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.22rem 0.65rem;
    font-size: 0.65rem;
    color: var(--text-muted);
    font-weight: 500;
}}

.profile-header {{
    text-align: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border);
}}
.profile-icon {{ font-size: 3.2rem; margin-bottom: 0.4rem; }}
.profile-name {{
    font-family: 'Syne', sans-serif !important;
    font-size: 1.35rem; font-weight: 800; color: var(--text-primary); margin-bottom: 0.2rem;
}}
.profile-title {{ color: var(--accent-blue-bright); font-size: 0.8rem; font-weight: 600; }}

.section-header {{
    font-family: 'Syne', sans-serif !important;
    font-size: 0.88rem; font-weight: 700; color: var(--text-primary);
    margin: 1rem 0 0.5rem 0; padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
    letter-spacing: 0.02em;
}}

/* ─── WELCOME SCREEN ─── */
.welcome-container {{
    max-width: 400px;
    margin: 50px auto;
    text-align: center;
}}

.welcome-badge {{
    display: inline-block;
    background: var(--glow-blue);
    border: 1px solid var(--border-glow);
    border-radius: 30px;
    padding: 0.3rem 1rem;
    font-size: 0.65rem;
    color: var(--accent-blue-bright);
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}}

.welcome-title {{
    font-family: 'Syne', sans-serif !important;
    font-size: 2rem;
    font-weight: 800;
    margin-bottom: 0.4rem;
    color: var(--text-primary);
    line-height: 1.1;
    letter-spacing: -0.03em;
}}

.welcome-gradient {{
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}

.welcome-subtitle {{
    color: var(--text-secondary);
    margin-bottom: 1.5rem;
    font-size: 0.82rem;
    line-height: 1.5;
}}

.welcome-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.4rem;
    box-shadow: 0 20px 60px var(--glow-blue);
    position: relative;
    overflow: hidden;
}}

.welcome-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-cyan));
}}

/* ─── QUIZ ─── */
.quiz-start-container {{
    max-width: 400px; margin: 30px auto; text-align: center;
}}

.quiz-start-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.4rem;
    box-shadow: 0 8px 32px var(--glow-blue);
}}

.quiz-start-icon {{ font-size: 2rem; margin-bottom: 0.5rem; }}
.quiz-start-title {{ font-family: 'Syne', sans-serif !important; font-size: 0.95rem; font-weight: 700; color: var(--text-primary); margin-bottom: 0.3rem; }}
.quiz-start-desc {{ color: var(--text-secondary); font-size: 0.72rem; margin-bottom: 0.8rem; }}

.quiz-question {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent-blue);
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.9rem;
    transition: border-color 0.2s ease;
}}

.quiz-question:hover {{ border-color: var(--accent-blue-bright); border-left-color: var(--accent-cyan); }}

.quiz-question-text {{ font-weight: 600; color: var(--text-primary); margin-bottom: 0.8rem; font-size: 0.88rem; line-height: 1.4; }}

/* ─── GLOBAL BUTTON THEME ─── */
.stButton > button {{
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    transition: all 0.25s ease !important;
}}

.stButton > button[kind="primary"] {{
    background: linear-gradient(135deg, var(--accent-blue-dark), var(--accent-blue)) !important;
    border: none !important;
    color: #ffffff !important;
    box-shadow: 0 4px 20px var(--glow-blue) !important;
}}

.stButton > button[kind="primary"]:hover {{
    background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan)) !important;
    box-shadow: 0 8px 32px var(--glow-blue) !important;
    transform: translateY(-2px) !important;
}}

/* ─── STREAMLIT NATIVE OVERRIDES ─── */
.stMarkdown p {{ color: var(--text-secondary) !important; }}
.stMarkdown strong {{ color: var(--text-primary) !important; }}
label {{ color: var(--text-secondary) !important; font-family: 'Space Grotesk', sans-serif !important; }}
.stRadio label {{ color: var(--text-primary) !important; }}

/* Divider */
hr {{ border-color: var(--border) !important; }}

/* Caption */
.stCaption {{ color: var(--text-muted) !important; font-size: 0.68rem !important; }}

/* File uploader — override Streamlit dark theme completely */
[data-testid="stFileUploader"] {{
    background: var(--bg-card) !important;
    border: 2px dashed var(--border-glow) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s ease !important;
}}
[data-testid="stFileUploader"]:hover {{ border-color: var(--accent-blue) !important; }}

[data-testid="stFileUploader"] > div {{
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: var(--bg-card2) !important;
    border: none !important;
    border-radius: 12px !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] {{
    color: var(--text-secondary) !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] svg {{
    fill: var(--accent-blue) !important;
    stroke: var(--accent-blue) !important;
}}

[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] small {{
    color: var(--text-secondary) !important;
}}

/* Upload button inside dropzone */
[data-testid="stFileUploaderDropzone"] button {{
    background: var(--accent-blue) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
}}

/* Spinner */
.stSpinner > div {{ border-top-color: var(--accent-blue) !important; }}

/* Success/Warning/Error */
.stSuccess {{ background: rgba(16,185,129,0.1) !important; border-color: var(--accent-green) !important; color: #34d399 !important; }}
.stWarning {{ background: rgba(245,158,11,0.1) !important; border-color: var(--accent-amber) !important; }}
.stError {{ background: rgba(239,68,68,0.1) !important; border-color: #ef4444 !important; }}

/* Glow pulse for hero badge */
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

    with st.sidebar:
        # Brand
        st.markdown(f"""
        <div class="sidebar-brand">
            <span class="sidebar-brand-logo">🎯</span>
            <div class="sidebar-brand-name">AI <span>Career</span> Platform</div>
        </div>
        """, unsafe_allow_html=True)

        # User chip
        st.markdown(f"""
        <div class="user-chip">👤 &nbsp;{first}</div>
        """, unsafe_allow_html=True)

        # Dark/Light toggle
        mode_label = f"{T['mode_icon']}  {T['mode_label']}"
        if st.button(mode_label, key="dark_toggle", use_container_width=True):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.markdown("<div style='margin-bottom:0.4rem'></div>", unsafe_allow_html=True)

        # Navigation buttons
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

        # Sign out
        if st.button("⏻  Sign Out", key="signout_btn", use_container_width=True):
            sign_out()

        st.markdown(f"""
        <div style="text-align:center; margin-top:0.6rem; font-size:0.6rem; color: var(--text-muted);">
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
            <div class="welcome-title">
                AI <span class="welcome-gradient">Career</span><br>Platform
            </div>
            <p class="welcome-subtitle">Your AI-powered career companion.<br>Match your CV to the best AI/ML roles in seconds.</p>
            <div class="welcome-card">
                <div style="font-size: 0.82rem; font-weight: 600; margin-bottom: 0.9rem; color: var(--text-primary);">
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
    if st.button("← Back to Home", key="back_home_analyze"):
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
            <div style="font-size: 0.72rem; color: var(--text-muted); margin-bottom: 0.3rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase;">Match Score</div>
            <div class="match-percentage">{top_match.get('match_pct', 0)}%</div>
            <h3 style="color: var(--text-primary); margin-top: 0.4rem; font-size: 1rem; font-family:'Syne',sans-serif; font-weight:700;">{top_match.get('title', top_match.get('role', 'AI Professional'))}</h3>
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
        <div style="margin-bottom: 0.8rem; padding: 0.7rem 1rem; background: var(--bg-card2); border: 1px solid var(--border); border-radius: 10px;">
            <strong style="color:var(--text-primary);">{r.get('title', r.get('role', 'Role'))}</strong> — {r.get('company', 'Various')} 
            <span style="color: var(--accent-blue-bright); font-weight:600;">({r.get('match_pct', 0)}% match)</span>
        </div>
        """, unsafe_allow_html=True)
        if r.get("salary_min"):
            st.caption(f"💰 ৳{r['salary_min']:,} – ৳{r['salary_max']:,}/month")

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
    if st.button("← Back to Home", key="back_home_jd"):
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
            <div style="font-size: 0.72rem; color: var(--text-muted); font-weight:600; letter-spacing:0.1em; text-transform:uppercase;">Match Score</div>
            <div style="font-family:'Syne',sans-serif; font-size: 3rem; font-weight: 800; color: {color}; line-height:1.1;">{pct}%</div>
            <div style="font-size: 0.9rem; font-weight: 600; color: {color}; margin-top:0.3rem;">{status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if result.get("similar_roles"):
        st.markdown("### 📊 Similar Roles from Database")
        for role in result["similar_roles"][:3]:
            st.markdown(f"""
            <div style="padding: 0.7rem 1rem; background: var(--bg-card); border: 1px solid var(--border); border-radius:10px; margin-bottom:0.5rem; color:var(--text-primary);">
                <strong>{role.get('title', role.get('role', 'Role'))}</strong>
                <span style="color:var(--text-muted);"> at {role.get('company', 'Various')}</span>
            </div>
            """, unsafe_allow_html=True)


# ============================================================
# QUIZ PAGE
# ============================================================
def render_quiz():
    st.markdown("""
    <div class="main-content">
        <div class="main-header">
            <h1>🧠 Career <span class="hl">Interest Quiz</span></h1>
            <p>Discover which AI/ML roles match your thinking style and interests.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.session_state.quiz_result:
        render_quiz_results()
        st.markdown("---")
        if st.button("← Back to Home", key="back_home_quiz"):
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
            if st.button("🚀 Start Quiz", use_container_width=True, type="primary"):
                st.session_state.quiz_responses = {q["id"]: None for q in QUESTIONS}
                st.rerun()

        st.markdown("---")
        if st.button("← Back to Home", key="back_home_quiz_start"):
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
            <div style="font-family:'Syne',sans-serif; font-size: 3rem; font-weight: 800; color: {color}; line-height:1.1;">{pct}%</div>
            <div style="font-size: 1rem; font-weight: 700; margin: 0.4rem 0; color: var(--text-primary); font-family:'Syne',sans-serif;">Interest Level: {level}</div>
            <div style="color: var(--text-secondary); font-size: 0.85rem;">{result["message"]}</div>
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
    <div style="margin-bottom: 1rem;">
        <span class="interest-tag">Natural Language Processing (NLP)</span>
        <span class="interest-tag">Computer Vision</span>
        <span class="interest-tag">Cyber Security</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🎓 Academic Affiliation</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: var(--text-secondary); margin-bottom: 1rem; font-size: 0.85rem; line-height:1.6;">
        Department of Computer Science &amp; Engineering<br>
        Rajshahi University of Engineering &amp; Technology (RUET)
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">🚀 About This Platform</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="color: var(--text-secondary); line-height: 1.6; margin-bottom: 1rem; font-size: 0.85rem;">
        AI Career Platform is an intelligent career matching system designed to help job seekers in Bangladesh
        find the best AI/ML roles based on their CV content, skills, and career preferences.
    </p>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">✨ Features</div>', unsafe_allow_html=True)
    st.markdown("""
    <ul style="color: var(--text-secondary); line-height: 1.8; margin-bottom: 1rem; font-size: 0.82rem; padding-left: 1rem;">
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
