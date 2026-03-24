"""
agent.py — compatible with langchain >= 0.2 / 1.x (modern API)
Uses: ChatGroq directly + tool binding, no deprecated agent helpers.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import Tool
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3,
    )


def get_job_search_tool():
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        def no_search(q):
            return "Job search unavailable — add SERPAPI_API_KEY to Streamlit secrets to enable real-time job listings."
        return Tool(name="job_search", func=no_search,
                    description="Search real-time job listings.")
    search = SerpAPIWrapper(
        serpapi_api_key=serpapi_key,
        params={"engine": "google_jobs"}
    )
    return Tool(
        name="job_search",
        func=search.run,
        description=(
            "Search real-time job listings from Google Jobs. "
            "Input: job title + location. "
            "Examples: 'ML Engineer Pakistan', 'Data Scientist Remote'"
        )
    )


def extract_cv_text(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "".join(page.extract_text() or "" for page in reader.pages).strip()


# Module-level state
_system_prompt = ""
_llm = None
_history = []


def build_agent(cv_text: str, jd_text: str = "", candidate_name: str = ""):
    """
    Build a simple stateful chat agent using ChatGroq directly.
    No deprecated agent helpers — works with all langchain versions.
    """
    global _system_prompt, _llm, _history

    from rag import retrieve_context
    retrieved = retrieve_context(cv_text, jd_text, k=5)
    top = retrieved["top_match"]

    name_ref = f"The candidate's name is {candidate_name}. " if candidate_name else ""

    role_lines = "\n".join(
        f"  {r.get('title', r.get('role', 'Unknown'))} ({r.get('category', '')}): "
        f"{r['match_pct']}% match — "
        f"Salary ₨{r.get('salary_min', 0):,}–₨{r.get('salary_max', 0):,} — "
        f"Demand: {r.get('market_demand', '')}"
        for r in retrieved["all_matches"][:5]
    )

    jd_section = (
        f"\nTarget JD provided by candidate:\n---\n{jd_text}\n---\n"
        f"Prioritize advice relevant to this specific JD.\n"
        if jd_text else ""
    )

    _system_prompt = f"""You are a professional AI career advisor specializing in data science and AI roles.
{name_ref}Your advice is grounded in real JD data retrieved via FAISS vector search.
Always be specific, encouraging, and reference the candidate's actual CV content.

━━━ CANDIDATE CV ━━━
{cv_text}
{jd_section}
━━━ FAISS RETRIEVED MATCHES ━━━

TOP MATCH: {top.get('title', top.get('role', 'N/A'))} — {top.get('category', 'N/A')} ({top.get('match_pct', 0)}% match)
Why this fits: {top.get('why_good_fit', '')}

ALL ROLE MATCHES:
{role_lines}

FULL CONTEXT:
{retrieved['raw_context']}

SKILL GAPS (missing from CV): {', '.join(retrieved['skill_gaps']) or 'None — strong alignment'}
RESUME SKILLS TO ADD: {', '.join(retrieved['resume_skills']) or 'CV already well-aligned'}

━━━ YOUR RESPONSIBILITIES ━━━

1. ROLE RECOMMENDATIONS — reference match % and CV evidence
2. SKILL GAPS — what it is, why it matters, how to learn it
3. RESUME RECOMMENDATIONS — format: "Add [X]: unlocks [role] — [reason]"
4. SALARY & MARKET INFO — use retrieved salary ranges
5. CAREER PATH — 3 steps with timeframes
6. JOB SEARCH — when asked, search and format: Title | Company | Location | Description

When generating structured analysis, respond using EXACTLY these tags:
TOP_ROLE: [role name]
MATCH_PCT: [number only]
WHY_RIGHT: [2-3 sentences personalised to this CV]
NEXT_STEPS:
- [step 1]
- [step 2]
- [step 3]
SKILL_GAPS:
- [skill]: [why and how to learn]
RESUME_ADD:
- Add [X]: unlocks [role] — [reason]
CAREER_PATH:
- [Step title]: [description and timeframe]
RUNNER_UP: [second best role]
RUNNER_UP_WHY: [1-2 sentences]

Always reference actual CV content. Never give generic advice."""

    _llm = get_llm()
    _history = []

    # Return a simple dict as the "agent" — run_agent uses it
    return {"ready": True}


def run_agent(agent_dict, user_input: str) -> str:
    """
    Run the agent: send system prompt + history + new message to Groq.
    Falls back gracefully if job search is requested.
    """
    global _system_prompt, _llm, _history

    if not _llm:
        return "Agent not initialized. Please upload your CV and click Get Career Match first."

    # Check if job search is requested
    search_keywords = ["find jobs", "search jobs", "job listings", "find me", "search for"]
    wants_search = any(kw in user_input.lower() for kw in search_keywords)

    search_result = ""
    if wants_search:
        try:
            tool = get_job_search_tool()
            search_result = tool.func(user_input)
            user_input = f"{user_input}\n\n[Job search results]:\n{search_result}"
        except Exception:
            pass

    # Build messages
    messages = [SystemMessage(content=_system_prompt)]
    for h in _history[-6:]:  # keep last 3 exchanges
        messages.append(HumanMessage(content=h["user"]))
        from langchain_core.messages import AIMessage
        messages.append(AIMessage(content=h["assistant"]))
    messages.append(HumanMessage(content=user_input))

    response = _llm.invoke(messages)
    reply = response.content

    _history.append({"user": user_input, "assistant": reply})
    return reply
