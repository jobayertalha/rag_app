"""
quiz.py — Career Interest & Aptitude Quiz
Focus on AI/ML interest alignment, problem-solving, and career preferences.
Questions shuffle each time to prevent bias.
"""

import random

# Base questions focused on AI/ML interest alignment
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
        "question": "How do you feel about mathematics (linear algebra, calculus, statistics)?",
        "options": ["Love it and good at it", "Can manage when needed", "Find it challenging", "Prefer to avoid math"],
        "scores": [3, 2, 1, 0],
        "category": "Math Comfort"
    },
    {
        "id": 3,
        "question": "When you see a news article about AI breakthroughs, you:",
        "options": ["Read it thoroughly with excitement", "Skim through it", "Just see the headline", "Usually ignore it"],
        "scores": [3, 2, 1, 0],
        "category": "AI Curiosity"
    },
    {
        "id": 4,
        "question": "How comfortable are you with writing code (Python, programming in general)?",
        "options": ["Very comfortable", "Somewhat comfortable", "Learning basics", "Not comfortable at all"],
        "scores": [3, 2, 1, 0],
        "category": "Coding Skill"
    },
    {
        "id": 5,
        "question": "Do you enjoy analyzing patterns in data or finding insights from information?",
        "options": ["Love it", "Enjoy sometimes", "Neutral", "Not my thing"],
        "scores": [3, 2, 1, 0],
        "category": "Analytical Thinking"
    },
    {
        "id": 6,
        "question": "Would you consider spending 6+ months learning AI/ML concepts and tools?",
        "options": ["Absolutely", "Maybe", "Unlikely", "Definitely not"],
        "scores": [3, 2, 1, 0],
        "category": "Learning Commitment"
    },
    {
        "id": 7,
        "question": "How curious are you about how ChatGPT, self-driving cars, or facial recognition works?",
        "options": ["Very curious - I research it", "Somewhat curious", "Mildly curious", "Not curious at all"],
        "scores": [3, 2, 1, 0],
        "category": "AI Curiosity"
    },
    {
        "id": 8,
        "question": "What excites you more in a potential career?",
        "options": [
            "Building intelligent systems that learn", 
            "Working with data and analytics", 
            "Traditional software development", 
            "Non-technical roles"
        ],
        "scores": [3, 2, 1, 0],
        "category": "Career Preference"
    },
    {
        "id": 9,
        "question": "How do you feel about keeping up with rapidly changing AI technologies?",
        "options": ["Excited to learn constantly", "Will learn when needed", "Prefer stable technologies", "Overwhelming"],
        "scores": [3, 2, 1, 0],
        "category": "Adaptability"
    },
    {
        "id": 10,
        "question": "Have you ever built or wanted to build a chatbot, recommendation system, or image classifier?",
        "options": ["Built one already", "Want to build", "Maybe someday", "Not interested"],
        "scores": [3, 2, 1, 0],
        "category": "Hands-on Interest"
    },
]

# Alternate questions pool (for variety when shuffling)
ALTERNATE_QUESTIONS = [
    {
        "id": 11,
        "question": "How often do you follow AI/ML researchers or publications (e.g., OpenAI, DeepMind, arXiv)?",
        "options": ["Regularly", "Occasionally", "Rarely", "Never"],
        "scores": [3, 2, 1, 0],
        "category": "AI Curiosity"
    },
    {
        "id": 12,
        "question": "Are you willing to learn cloud platforms (AWS, GCP, Azure) for deploying ML models?",
        "options": ["Yes, excited", "Probably yes", "Maybe", "Not interested"],
        "scores": [3, 2, 1, 0],
        "category": "Learning Commitment"
    },
    {
        "id": 13,
        "question": "How do you approach solving a complex problem with no clear solution?",
        "options": ["Systematically experiment", "Research existing solutions", "Ask for help", "Avoid if possible"],
        "scores": [3, 2, 1, 0],
        "category": "Problem Solving"
    },
    {
        "id": 14,
        "question": "Do you enjoy participating in hackathons, Kaggle competitions, or coding challenges?",
        "options": ["Love them", "Done a few", "Interested but not yet", "Not interested"],
        "scores": [3, 2, 1, 0],
        "category": "Hands-on Interest"
    },
    {
        "id": 15,
        "question": "How important is it for you to understand the 'why' behind a model's prediction (explainable AI)?",
        "options": ["Very important", "Somewhat important", "Not very important", "Don't care"],
        "scores": [3, 2, 1, 0],
        "category": "Analytical Thinking"
    }
]

# Maximum possible score
MAX_SCORE = sum(max(q["scores"]) for q in BASE_QUESTIONS[:10])  # 30 points max


def get_shuffled_questions():
    """Return shuffled questions, randomly selecting from base and alternate pool."""
    # Take all base questions (10) and add some alternates
    all_questions = BASE_QUESTIONS.copy()
    
    # Add alternate questions if they're not already included
    existing_ids = {q["id"] for q in all_questions}
    for alt_q in ALTERNATE_QUESTIONS:
        if alt_q["id"] not in existing_ids:
            all_questions.append(alt_q)
    
    # Shuffle the questions
    shuffled = random.sample(all_questions, min(10, len(all_questions)))
    
    # Renumber for display
    for i, q in enumerate(shuffled):
        q["display_id"] = i + 1
    
    return shuffled


def calculate_interest_score(responses: dict, questions_list: list = None) -> dict:
    """
    responses: {question_display_id: selected_option_index}
    questions_list: the list of questions used for this quiz session
    Returns score, level, and recommendations.
    Score range: 0-30 (10 questions × max 3 points each)
    """
    total_score = 0
    category_scores = {}
    
    # Use provided questions list, otherwise fallback to base questions
    if questions_list is None:
        questions_list = BASE_QUESTIONS[:10]
    
    for q in questions_list:
        qid = q.get("display_id", q["id"])
        selected = responses.get(qid, -1)
        if selected >= 0 and selected < len(q["scores"]):
            score = q["scores"][selected]
            total_score += score
            
            # Track category scores
            cat = q["category"]
            if cat not in category_scores:
                category_scores[cat] = {"score": 0, "max_possible": 0}
            category_scores[cat]["score"] += score
            category_scores[cat]["max_possible"] += max(q["scores"])
    
    # Calculate percentage (0-100)
    max_possible = MAX_SCORE
    pct = int((total_score / max_possible) * 100)
    pct = max(0, min(100, pct))
    
    # Determine alignment level and recommendations
    if total_score >= 21:  # 70%+ of 30
        level = "STRONG ALIGNMENT"
        alignment = "high"
        color = "#10b981"
        icon = "🚀"
        recommendation = {
            "verdict": "You are strongly aligned with AI/ML career paths!",
            "message": "Your responses show genuine interest, curiosity, and aptitude for AI/ML. You'd likely thrive in this field.",
            "roles": [
                "🤖 AI/ML Engineer",
                "🔬 Research Scientist", 
                "🧠 NLP Engineer",
                "👁️ Computer Vision Engineer",
                "📊 Data Scientist"
            ],
            "next_steps": [
                "Start with Python and ML fundamentals (3 months)",
                "Build 2-3 portfolio projects (chatbot, image classifier, recommendation system)",
                "Earn certifications from DeepLearning.ai or Coursera",
                "Apply for AI/ML internships or junior roles"
            ]
        }
    elif total_score >= 10:  # 33-69%
        level = "MODERATE ALIGNMENT"
        alignment = "medium"
        color = "#f59e0b"
        icon = "🔍"
        recommendation = {
            "verdict": "You have some interest in AI/ML, but should explore further.",
            "message": "You show curiosity in data and technology. AI/ML could be a good fit, but consider exploring adjacent fields too.",
            "roles": [
                "📈 Data Analyst",
                "🛠️ ML Engineer (entry-level)",
                "💻 Software Developer with ML focus",
                "📊 Business Intelligence Analyst",
                "🔧 Data Engineer"
            ],
            "next_steps": [
                "Take an introductory AI/ML course (Andrew Ng's ML course)",
                "Build one small project to test your interest",
                "Explore data analysis roles as an alternative",
                "Talk to professionals in both AI and traditional software roles"
            ]
        }
    else:  # <33%
        level = "LOW ALIGNMENT"
        alignment = "low"
        color = "#ef4444"
        icon = "⚡"
        recommendation = {
            "verdict": "AI/ML may not be the best fit for you right now.",
            "message": "Your responses suggest other fields might align better with your interests and strengths.",
            "roles": [
                "💻 Software Developer",
                "📱 Mobile App Developer",
                "🌐 Web Developer",
                "🔧 IT Support / System Administrator",
                "📋 Project Manager (Tech)"
            ],
            "next_steps": [
                "Explore traditional software development paths",
                "Consider what aspects of tech excite you most",
                "Try building a simple web or mobile app",
                "Speak with career counselors about alternative paths"
            ]
        }
    
    return {
        "score": total_score,
        "max_score": max_possible,
        "pct": pct,
        "level": level,
        "alignment": alignment,
        "color": color,
        "icon": icon,
        "recommendation": recommendation,
        "category_scores": category_scores,
        "questions_used": len(questions_list)
    }


def reset_quiz():
    """Reset quiz state (to be called from app.py)"""
    pass  # This will be handled by app.py session state
