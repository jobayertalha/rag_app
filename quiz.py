"""
quiz.py — Career Interest & Aptitude Quiz (No technical questions)
Focus on thinking style, problem-solving, and career preferences.
"""

QUESTIONS = [
    {
        "id": 1,
        "question": "How do you feel about solving complex puzzles or logic problems?",
        "options": ["Love it", "Enjoy it sometimes", "Neutral", "Prefer to avoid"],
        "scores": [2, 1, 0, -1],
        "category": "Problem Solving"
    },
    {
        "id": 2,
        "question": "When faced with a new technology, your first reaction is:",
        "options": ["Excited to learn it", "Curious but cautious", "Wait until I need it", "Prefer what I know"],
        "scores": [2, 1, 0, -1],
        "category": "Learning Style"
    },
    {
        "id": 3,
        "question": "Would you rather build something new or analyze existing data?",
        "options": ["Build new things", "Analyze data", "Both equally", "Neither"],
        "scores": [2, 1, 0, -1],
        "category": "Work Preference"
    },
    {
        "id": 4,
        "question": "How comfortable are you with mathematics and numbers?",
        "options": ["Very comfortable", "Somewhat comfortable", "Neutral", "Not comfortable"],
        "scores": [2, 1, 0, -1],
        "category": "Quantitative"
    },
    {
        "id": 5,
        "question": "Do you enjoy finding patterns in data or information?",
        "options": ["Always", "Often", "Sometimes", "Rarely"],
        "scores": [2, 1, 0, -1],
        "category": "Pattern Recognition"
    },
    {
        "id": 6,
        "question": "How would you describe your attention to detail?",
        "options": ["Very detailed", "Detailed enough", "Average", "Prefer big picture"],
        "scores": [2, 1, 0, -1],
        "category": "Work Style"
    },
    {
        "id": 7,
        "question": "When working on a project, you prefer:",
        "options": ["Clear structure and guidelines", "Flexibility to explore", "Working with a team", "Working independently"],
        "scores": [1, 2, 1, 1],
        "category": "Work Environment"
    },
    {
        "id": 8,
        "question": "How do you stay updated with technology trends?",
        "options": ["Follow actively", "Read occasionally", "When needed", "Don't follow"],
        "scores": [2, 1, 0, -1],
        "category": "Tech Engagement"
    },
    {
        "id": 9,
        "question": "What excites you most about a potential career?",
        "options": ["Solving challenging problems", "Creating innovative products", "Working with data", "Helping people"],
        "scores": [2, 2, 1, 1],
        "category": "Motivation"
    },
    {
        "id": 10,
        "question": "How do you handle ambiguity or unclear requirements?",
        "options": ["Thrive on it", "Manage well", "Need clarity", "Avoid it"],
        "scores": [2, 1, 0, -1],
        "category": "Adaptability"
    },
]

MAX_SCORE = sum(max(q["scores"]) for q in QUESTIONS)

# Role recommendations based on score ranges
ROLE_RECOMMENDATIONS = {
    "high": {
        "roles": ["🤖 AI/ML Engineer", "🔬 Research Scientist", "🧠 NLP Engineer", "👁️ Computer Vision Engineer"],
        "message": "You have strong analytical thinking and curiosity — perfect for cutting-edge AI roles!",
        "explanation": "Your profile shows high interest in problem-solving, learning new technologies, and working with complex systems. You'd thrive in research or development roles."
    },
    "medium": {
        "roles": ["📊 Data Scientist", "📈 Data Analyst", "🛠️ ML Engineer", "📉 Business Intelligence Analyst"],
        "message": "You have good analytical skills — with some focused learning, you can excel in data roles!",
        "explanation": "You enjoy working with data and solving problems. With some technical skill development, you'd be great in data-focused roles."
    },
    "low": {
        "roles": ["💻 Software Developer", "📱 App Developer", "🌐 Web Developer", "🔧 IT Support"],
        "message": "You might enjoy roles that focus more on building than research — explore development paths!",
        "explanation": "Your interests lean more toward building and creating rather than pure analysis. Consider software development or engineering roles."
    }
}


def calculate_interest_score(responses: dict) -> dict:
    """
    responses: {question_id: selected_option_index}
    Returns score, level, and recommendations.
    """
    total_score = 0
    category_scores = {}
    
    for q in QUESTIONS:
        qid = q["id"]
        selected = responses.get(qid, 0)
        score = q["scores"][selected] if selected < len(q["scores"]) else 0
        total_score += score
        
        # Track category scores
        cat = q["category"]
        if cat not in category_scores:
            category_scores[cat] = {"score": 0, "max": max(q["scores"])}
        category_scores[cat]["score"] += score
    
    # Calculate percentage (normalize to 0-100)
    max_possible = MAX_SCORE
    pct = int((total_score / max_possible) * 100)
    pct = max(0, min(100, pct))
    
    # Determine level
    if pct >= 70:
        level = "HIGH"
        tier = "high"
        color = "#10b981"
    elif pct >= 45:
        level = "MEDIUM"
        tier = "medium"
        color = "#f59e0b"
    else:
        level = "LOW"
        tier = "low"
        color = "#ef4444"
    
    rec = ROLE_RECOMMENDATIONS[tier]
    
    return {
        "score": total_score,
        "max_score": max_possible,
        "pct": pct,
        "level": level,
        "color": color,
        "recommended_roles": rec["roles"],
        "message": rec["message"],
        "explanation": rec["explanation"],
        "category_scores": category_scores,
    }
