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
    max_possible = 30  # MAX_SCORE
    pct = int((total_score / max_possible) * 100) if max_possible > 0 else 0
    pct = max(0, min(100, pct))
    
    # ============================================================
    # DETAILED SCORE ANALYSIS BASED ON YOUR REQUIREMENTS
    # Score ranges out of 30:
    # 21-30 points (70-100%): STRONG ALIGNMENT - Focus on AI/ML
    # 10-20 points (33-69%): MODERATE ALIGNMENT - Explore mixed fields
    # 0-9 points (0-33%): LOW ALIGNMENT - Not aligned with AI/ML
    # ============================================================
    
    # Detailed breakdown by score
    if total_score >= 21:  # 70%+ of 30
        level = "STRONG ALIGNMENT"
        alignment = "high"
        color = "#10b981"
        icon = "🚀"
        
        # Detailed analysis for high scorers
        strength_areas = []
        if category_scores.get("ML Interest", {}).get("score", 0) >= 5:
            strength_areas.append("Strong interest in ML systems")
        if category_scores.get("AI Curiosity", {}).get("score", 0) >= 4:
            strength_areas.append("High curiosity about AI breakthroughs")
        if category_scores.get("Hands-on Interest", {}).get("score", 0) >= 3:
            strength_areas.append("Hands-on project building interest")
        
        recommendation = {
            "verdict": "✅ STRONG ALIGNMENT - Focus on AI/ML Career Path!",
            "message": f"Your score of {total_score}/30 ({pct}%) indicates strong interest and aptitude for AI/ML. You should definitely pursue this field!",
            "detailed_analysis": f"""
            **Your Score Analysis:** {total_score}/30 points
            
            **What this means:** You have genuine passion for AI/ML concepts, mathematical thinking, and hands-on coding. 
            Your responses show {', '.join(strength_areas) if strength_areas else 'consistent interest across all areas'}.
            
            **Career Recommendation:** Focus 100% on AI/ML career path. You have the right mindset and curiosity.
            """,
            "roles": [
                "🤖 AI/ML Engineer",
                "🔬 Research Scientist", 
                "🧠 NLP Engineer",
                "👁️ Computer Vision Engineer",
                "📊 Data Scientist"
            ],
            "next_steps": [
                "🚀 Start with Python and ML fundamentals immediately",
                "💡 Build 2-3 portfolio projects in your area of interest",
                "📜 Earn certifications from DeepLearning.ai or Coursera",
                "🎯 Apply for AI/ML internships within 3-6 months",
                "🔗 Join AI/ML communities (Kaggle, Hugging Face, GitHub)"
            ]
        }
        
    elif total_score >= 10:  # 33-69% - Moderate alignment
        level = "MODERATE ALIGNMENT"
        alignment = "medium"
        color = "#f59e0b"
        icon = "🔍"
        
        # Determine which areas are stronger
        stronger_areas = []
        weaker_areas = []
        for cat, data in category_scores.items():
            if data["max_possible"] > 0:
                cat_pct = (data["score"] / data["max_possible"]) * 100
                if cat_pct >= 60:
                    stronger_areas.append(cat)
                elif cat_pct < 40:
                    weaker_areas.append(cat)
        
        recommendation = {
            "verdict": "🔍 MODERATE ALIGNMENT - Explore AI/ML Alongside Other Fields",
            "message": f"Your score of {total_score}/30 ({pct}%) shows some interest in AI/ML, but you might also enjoy other tech fields.",
            "detailed_analysis": f"""
            **Your Score Analysis:** {total_score}/30 points
            
            **What this means:** You have curiosity about AI/ML but may need more exposure to decide.
            
            **Your Stronger Areas:** {', '.join(stronger_areas) if stronger_areas else 'General interest across topics'}
            **Areas to Explore More:** {', '.join(weaker_areas) if weaker_areas else 'Consider taking an introductory course'}
            
            **Career Recommendation:** Explore both AI/ML and traditional software development. Consider a minor in AI/ML while pursuing CS degree.
            """,
            "roles": [
                "📈 Data Analyst (bridge to AI/ML)",
                "🛠️ ML Engineer (entry-level)",
                "💻 Software Developer with ML focus",
                "📊 Business Intelligence Analyst",
                "🔧 Data Engineer"
            ],
            "next_steps": [
                "📚 Take Andrew Ng's Machine Learning course first",
                "💻 Build one small AI project to test your interest",
                "🔀 Explore data analysis as an alternative path",
                "💬 Talk to professionals in both AI and traditional software roles",
                "📖 Consider a dual-track learning path (AI + traditional dev)"
            ]
        }
        
    else:  # <33% - Low alignment
        level = "LOW ALIGNMENT"
        alignment = "low"
        color = "#ef4444"
        icon = "⚡"
        
        # Find what interests them instead
        other_interests = []
        if category_scores.get("Coding Skill", {}).get("score", 0) >= 2:
            other_interests.append("Coding/Programming")
        
        recommendation = {
            "verdict": "⚡ LOW ALIGNMENT - AI/ML May Not Be Your Best Fit",
            "message": f"Your score of {total_score}/30 ({pct}%) suggests other technology fields might suit you better.",
            "detailed_analysis": f"""
            **Your Score Analysis:** {total_score}/30 points
            
            **What this means:** Your interests and strengths align more with other tech fields than pure AI/ML.
            
            **Consider These Alternatives:** {', '.join(other_interests) if other_interests else 'Traditional software development, web/mobile development, or IT operations'}
            
            **Career Recommendation:** Focus on building strong programming fundamentals first. AI/ML can be an elective later.
            """,
            "roles": [
                "💻 Software Developer",
                "📱 Mobile App Developer",
                "🌐 Web Developer",
                "🔧 IT Support / System Administrator",
                "📋 Project Manager (Tech)"
            ],
            "next_steps": [
                "💻 Master core programming first (Python, JavaScript)",
                "🏗️ Build web or mobile apps to find your passion",
                "🎓 Consider CS degree with AI/ML as elective, not major",
                "🗣️ Speak with career counselors about alternative paths",
                "📊 Data analysis could be a lighter entry point if interested"
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
        "questions_used": len(questions_list) if questions_list else 10
    }



def reset_quiz():
    """Reset quiz state (to be called from app.py)"""
    pass  # This will be handled by app.py session state
