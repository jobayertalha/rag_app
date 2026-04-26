"""
quiz.py — Career Interest & Aptitude Quiz
Focus on AI/ML interest alignment, problem-solving, and career preferences.
"""

import random

BASE_QUESTIONS = [
    {
        "id": 1,
        "question": "How interested are you in building systems that can learn from data?",
        "options": ["Extremely interested", "Somewhat interested", "Neutral", "Not really interested"],
        "scores": [3, 2, 1, 0],
        "category": "ML Interest"
    },
    {
        "id": 2,
        "question": "How do you feel about mathematics?",
        "options": ["Love it", "Can manage", "Find it challenging", "Prefer to avoid"],
        "scores": [3, 2, 1, 0],
        "category": "Math Comfort"
    },
    {
        "id": 3,
        "question": "When you see a news article about AI breakthroughs, you:",
        "options": ["Read it with excitement", "Skim through it", "Just see headline", "Usually ignore it"],
        "scores": [3, 2, 1, 0],
        "category": "AI Curiosity"
    },
    {
        "id": 4,
        "question": "How comfortable are you with writing code?",
        "options": ["Very comfortable", "Somewhat comfortable", "Learning basics", "Not comfortable"],
        "scores": [3, 2, 1, 0],
        "category": "Coding Skill"
    },
    {
        "id": 5,
        "question": "Do you enjoy analyzing patterns in data?",
        "options": ["Love it", "Enjoy sometimes", "Neutral", "Not my thing"],
        "scores": [3, 2, 1, 0],
        "category": "Analytical Thinking"
    },
    {
        "id": 6,
        "question": "Would you spend 6+ months learning AI/ML?",
        "options": ["Absolutely", "Maybe", "Unlikely", "Definitely not"],
        "scores": [3, 2, 1, 0],
        "category": "Learning Commitment"
    },
    {
        "id": 7,
        "question": "How curious are you about how ChatGPT works?",
        "options": ["Very curious", "Somewhat curious", "Mildly curious", "Not curious"],
        "scores": [3, 2, 1, 0],
        "category": "AI Curiosity"
    },
    {
        "id": 8,
        "question": "What excites you more in a career?",
        "options": ["Building intelligent systems", "Working with data", "Software development", "Non-technical roles"],
        "scores": [3, 2, 1, 0],
        "category": "Career Preference"
    },
    {
        "id": 9,
        "question": "How do you feel about rapidly changing AI tech?",
        "options": ["Excited to learn", "Learn when needed", "Prefer stable tech", "Overwhelming"],
        "scores": [3, 2, 1, 0],
        "category": "Adaptability"
    },
    {
        "id": 10,
        "question": "Have you built or wanted to build AI projects?",
        "options": ["Built one already", "Want to build", "Maybe someday", "Not interested"],
        "scores": [3, 2, 1, 0],
        "category": "Hands-on Interest"
    },
]

ALTERNATE_QUESTIONS = [
    {
        "id": 11,
        "question": "How often do you follow AI/ML researchers?",
        "options": ["Regularly", "Occasionally", "Rarely", "Never"],
        "scores": [3, 2, 1, 0],
        "category": "AI Curiosity"
    },
    {
        "id": 12,
        "question": "Are you willing to learn cloud platforms for ML?",
        "options": ["Yes, excited", "Probably yes", "Maybe", "Not interested"],
        "scores": [3, 2, 1, 0],
        "category": "Learning Commitment"
    },
    {
        "id": 13,
        "question": "How do you solve complex problems?",
        "options": ["Systematically experiment", "Research solutions", "Ask for help", "Avoid if possible"],
        "scores": [3, 2, 1, 0],
        "category": "Problem Solving"
    },
    {
        "id": 14,
        "question": "Do you enjoy hackathons or Kaggle?",
        "options": ["Love them", "Done a few", "Interested", "Not interested"],
        "scores": [3, 2, 1, 0],
        "category": "Hands-on Interest"
    },
    {
        "id": 15,
        "question": "How important is explainable AI to you?",
        "options": ["Very important", "Somewhat important", "Not very", "Don't care"],
        "scores": [3, 2, 1, 0],
        "category": "Analytical Thinking"
    }
]

MAX_SCORE = 30


def get_shuffled_questions():
    all_questions = BASE_QUESTIONS.copy()
    existing_ids = {q["id"] for q in all_questions}
    for alt_q in ALTERNATE_QUESTIONS:
        if alt_q["id"] not in existing_ids:
            all_questions.append(alt_q)
    shuffled = random.sample(all_questions, min(10, len(all_questions)))
    for i, q in enumerate(shuffled):
        q["display_id"] = i + 1
    return shuffled


def calculate_interest_score(responses: dict, questions_list: list = None) -> dict:
    total_score = 0
    category_scores = {}
    
    if questions_list is None:
        questions_list = BASE_QUESTIONS[:10]
    
    for q in questions_list:
        qid = q.get("display_id", q["id"])
        selected = responses.get(qid, -1)
        if selected >= 0 and selected < len(q["scores"]):
            score = q["scores"][selected]
            total_score += score
            cat = q["category"]
            if cat not in category_scores:
                category_scores[cat] = {"score": 0, "max_possible": 0}
            category_scores[cat]["score"] += score
            category_scores[cat]["max_possible"] += max(q["scores"])
    
    pct = int((total_score / MAX_SCORE) * 100) if MAX_SCORE > 0 else 0
    pct = max(0, min(100, pct))
    
    if total_score >= 21:
        level = "STRONG ALIGNMENT"
        alignment = "high"
        color = "#10b981"
        icon = "🚀"
        detailed = f"Score: {total_score}/30 ({pct}%)\n\nYou have genuine passion for AI/ML concepts and hands-on coding. Focus 100% on AI/ML career path."
        roles = ["AI/ML Engineer", "Research Scientist", "NLP Engineer", "Computer Vision Engineer", "Data Scientist"]
        steps = ["Learn Python & ML fundamentals", "Build 2-3 portfolio projects", "Get certifications", "Apply for internships", "Join AI communities"]
    elif total_score >= 10:
        level = "MODERATE ALIGNMENT"
        alignment = "medium"
        color = "#f59e0b"
        icon = "🔍"
        detailed = f"Score: {total_score}/30 ({pct}%)\n\nYou have curiosity about AI/ML. Explore both AI/ML and traditional software development."
        roles = ["Data Analyst", "ML Engineer (entry-level)", "Software Developer with ML", "BI Analyst", "Data Engineer"]
        steps = ["Take intro AI/ML course", "Build one small AI project", "Explore data analysis", "Talk to professionals", "Consider dual-track learning"]
    else:
        level = "LOW ALIGNMENT"
        alignment = "low"
        color = "#ef4444"
        icon = "⚡"
        detailed = f"Score: {total_score}/30 ({pct}%)\n\nYour interests align with other tech fields. Focus on programming fundamentals first."
        roles = ["Software Developer", "Mobile App Developer", "Web Developer", "IT Support", "Project Manager"]
        steps = ["Master core programming", "Build web/mobile apps", "Consider CS degree", "Talk to career counselors", "Explore data analysis"]
    
    return {
        "score": total_score,
        "max_score": MAX_SCORE,
        "pct": pct,
        "level": level,
        "alignment": alignment,
        "color": color,
        "icon": icon,
        "recommendation": {
            "verdict": level,
            "message": detailed,
            "detailed_analysis": detailed,
            "roles": roles,
            "next_steps": steps
        },
        "category_scores": category_scores,
        "questions_used": len(questions_list) if questions_list else 10
    }


def reset_quiz():
    pass
