"""
agent.py
Architecture step: Retrieved Context → LLM → Personalized Advice
The LLM receives ALL retrieved JD data: match scores, salary ranges,
market demand, career paths, skill gaps, resume recommendations.
"""

import os
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        # Return a dummy tool if no SERPAPI key
        def no_search(q):
            return "Job search unavailable — add SERPAPI_API_KEY to .env to enable real-time job listings."
        return Tool(
            name="job_search",
            func=no_search,
            description="Search real-time job listings."
        )
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


def build_agent(cv_text: str, jd_text: str = "", candidate_name: str = ""):
    """
    1. retrieve_context() → FAISS search → structured dict
    2. Format into rich system prompt
    3. LLM generates personalized advice with salary + market data
    """
    from rag import retrieve_context
    retrieved = retrieve_context(cv_text, jd_text, k=5)
    top = retrieved["top_match"]

    name_ref = f"The candidate's name is {candidate_name}. " if candidate_name else ""

    # Build role breakdown string for prompt
    role_lines = "\n".join(
        f"  {r.get('title', r.get('role', 'Unknown'))} ({r.get('category', '')}): {r['match_pct']}% match — "
        f"Salary ₨{r.get('salary_min', 0):,}–₨{r.get('salary_max', 0):,} — "
        f"Demand: {r.get('market_demand', '')}"
        for r in retrieved["all_matches"][:5]
    )

    jd_section = (
        f"\nTarget JD provided by candidate:\n---\n{jd_text}\n---\n"
        f"Prioritize advice relevant to this specific JD.\n"
        if jd_text else ""
    )

    system_prompt = f"""You are a professional AI career advisor specializing in data science and AI roles.
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

1. ROLE RECOMMENDATIONS
   Reference match percentages from FAISS. Explain WHY each role fits
   using specific evidence from the CV.

2. SKILL GAPS
   For each gap skill: what it is, why it matters for the matched role,
   how to learn it (specific course/project/certification).

3. RESUME RECOMMENDATIONS — be very specific:
   Format each tip as: "Add [X]: unlocks [role] — [reason from JD data]"
   Suggest 2-3 concrete project ideas they can build and list on resume.
   Suggest certifications that map directly to their gap skills.

4. SALARY & MARKET INFO
   Reference the salary ranges from retrieved JD data.
   Comment on market demand for their top matched roles.

5. CAREER PATH
   Use career paths from matched JDs.
   Give 3 steps with realistic timeframes based on their current level.

6. JOB SEARCH
   When asked to find jobs, use job_search tool.
   Format: Title | Company | Location | Brief description
   Suggest which matched role to search first based on FAISS scores.

When generating structured analysis, respond using EXACTLY these tags:
TOP_ROLE: [role name]
MATCH_PCT: [number only, e.g. 97]
WHY_RIGHT: [2-3 sentences personalised to this CV]
NEXT_STEPS:
- [step 1]
- [step 2]
- [step 3]
SKILL_GAPS:
- [skill]: [why it matters and how to learn it]
- [skill]: [why it matters and how to learn it]
RESUME_ADD:
- Add [X]: unlocks [role] — [reason]
- Add [X]: unlocks [role] — [reason]
CAREER_PATH:
- [Step 1 title]: [description and timeframe]
- [Step 2 title]: [description and timeframe]
- [Step 3 title]: [description and timeframe]
RUNNER_UP: [second best role name]
RUNNER_UP_WHY: [1-2 sentences why]

Always reference actual CV content. Never give generic advice."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    llm = get_llm()
    tool = get_job_search_tool()
    agent = create_openai_tools_agent(llm, [tool], prompt)

    return AgentExecutor(
        agent=agent, tools=[tool],
        verbose=True, max_iterations=5,
        handle_parsing_errors=True,
    )


def run_agent(executor, user_input: str) -> str:
    return executor.invoke({"input": user_input})["output"]
