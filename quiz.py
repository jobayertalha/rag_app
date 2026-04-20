"""
quiz.py — AI/ML Interest & Aptitude Quiz Module
Standalone module, imported and rendered by app.py.
"""

QUESTIONS = [
    {
        "id": 1,
        "category": "Python Basics",
        "question": "What is the output of: `print(type([]))`?",
        "options": [
            "<class 'list'>",
            "<class 'array'>",
            "<type 'list'>",
            "list"
        ],
        "correct": 0,
        "points": 1,
    },
    {
        "id": 2,
        "category": "Python Basics",
        "question": "Which of these is used for vectorized operations on arrays in Python?",
        "options": ["pandas", "NumPy", "matplotlib", "requests"],
        "correct": 1,
        "points": 1,
    },
    {
        "id": 3,
        "category": "Python Basics",
        "question": "What does a Python dictionary store?",
        "options": [
            "Ordered sequences of numbers",
            "Key-value pairs",
            "Only string values",
            "Immutable sequences"
        ],
        "correct": 1,
        "points": 1,
    },
    {
        "id": 4,
        "category": "Math & Statistics",
        "question": "What does the mean of a dataset represent?",
        "options": [
            "The most frequently occurring value",
            "The middle value when sorted",
            "The sum of all values divided by count",
            "The range of the dataset"
        ],
        "correct": 2,
        "points": 1,
    },
    {
        "id": 5,
        "category": "Math & Statistics",
        "question": "In a normal distribution, what percentage of data falls within ±1 standard deviation?",
        "options": ["50%", "68%", "95%", "99.7%"],
        "correct": 1,
        "points": 1,
    },
    {
        "id": 6,
        "category": "Math & Statistics",
        "question": "Which matrix operation is NOT valid if A is (3×2) and B is (3×2)?",
        "options": [
            "A + B",
            "A - B",
            "A × B (standard matrix multiply)",
            "Element-wise multiplication"
        ],
        "correct": 2,
        "points": 2,
    },
    {
        "id": 7,
        "category": "Logical Reasoning",
        "question": "If all models are algorithms, and all algorithms are code, then:",
        "options": [
            "All code is a model",
            "All models are code",
            "Some code is not an algorithm",
            "None of the above"
        ],
        "correct": 1,
        "points": 1,
    },
    {
        "id": 8,
        "category": "Logical Reasoning",
        "question": "A model has 90% training accuracy but 55% test accuracy. What is this called?",
        "options": ["Underfitting", "Overfitting", "Regularization", "Data leakage"],
        "correct": 1,
        "points": 2,
    },
    {
        "id": 9,
        "category": "AI/ML Concepts",
        "question": "Which technique is used to find similar documents in a large corpus using vector representations?",
        "options": [
            "Bubble Sort",
            "Semantic/Vector Search",
            "Binary Search",
            "Relational querying"
        ],
        "correct": 1,
        "points": 2,
    },
    {
        "id": 10,
        "category": "AI/ML Concepts",
        "question": "What is the purpose of a loss function in training a neural network?",
        "options": [
            "To visualize training progress",
            "To store model weights",
            "To measure prediction error and guide weight updates",
            "To initialize the model parameters"
        ],
        "correct": 2,
        "points": 2,
    },
    {
        "id": 11,
        "category": "AI/ML Concepts",
        "question": "Which of the following is a supervised learning algorithm?",
        "options": ["K-Means Clustering", "DBSCAN", "Principal Component Analysis", "Random Forest"],
        "correct": 3,
        "points": 1,
    },
    {
        "id": 12,
        "category": "AI/ML Concepts",
        "question": "What does LLM stand for in the context of modern AI?",
        "options": [
            "Linear Learning Model",
            "Large Language Model",
            "Layered Logic Machine",
            "Long Learning Module"
        ],
        "correct": 1,
        "points": 1,
    },
    {
        "id": 13,
        "category": "Interest & Motivation",
        "question": "How do you feel when you encounter a complex dataset with missing values and inconsistencies?",
        "options": [
            "Excited — I love cleaning and exploring data",
            "Neutral — I'll do it if required",
            "Uncomfortable — I prefer working with clean data only",
            "Disinterested — data work isn't for me"
        ],
        "correct": 0,
        "points": 2,
    },
    {
        "id": 14,
        "category": "Interest & Motivation",
        "question": "How often do you engage with AI/ML content outside of formal education?",
        "options": [
            "Regularly — papers, courses, projects",
            "Sometimes — when relevant to my work",
            "Rarely — only when assigned",
            "Never — I haven't started yet"
        ],
        "correct": 0,
        "points": 2,
    },
    {
        "id": 15,
        "category": "Interest & Motivation",
        "question": "You are given a week to build any project. What would you choose?",
        "options": [
            "A machine learning model or AI-powered app",
            "A data dashboard or analytics tool",
            "A standard web or mobile application",
            "I haven't thought about building personal projects"
        ],
        "correct": 0,
        "points": 2,
    },
]

MAX_SCORE = sum(q["points"] for q in QUESTIONS)

# Role recommendations keyed by score tier
ROLE_RECS = {
    "high": [
        "🤖 AI / LLM Engineer",
        "📊 Data Scientist",
        "🔬 ML Research Engineer",
        "🧠 NLP Engineer",
    ],
    "medium": [
        "📈 Data Analyst",
        "🛠️ ML Engineer (Junior)",
        "☁️ Cloud / MLOps Engineer",
        "📉 Business Intelligence Developer",
    ],
}


def calculate_quiz_score(answers: dict) -> dict:
    """
    answers: {question_id (int): selected_option_index (int)}
    Returns score dict with percentage, tier, and feedback.
    """
    earned = 0
    results = []

    for q in QUESTIONS:
        qid = q["id"]
        selected = answers.get(qid, -1)
        correct = q["correct"]
        is_correct = selected == correct
        pts = q["points"] if is_correct else 0
        earned += pts
        results.append({
            "id": qid,
            "category": q["category"],
            "question": q["question"],
            "selected": selected,
            "correct": correct,
            "is_correct": is_correct,
            "points_earned": pts,
            "points_possible": q["points"],
            "correct_answer": q["options"][correct],
            "selected_answer": q["options"][selected] if selected >= 0 else "Not answered",
        })

    pct = round((earned / MAX_SCORE) * 100)

    if pct >= 70:
        tier = "high"
        verdict = "suitable"
        verdict_msg = "✅ You are well-suited for the AI/ML field!"
        verdict_detail = (
            "Your aptitude scores, logical reasoning, and interest signals all point strongly "
            "toward a successful AI/ML career. You demonstrate the curiosity and analytical "
            "thinking that defines great AI practitioners."
        )
        color = "#10b981"
        roles = ROLE_RECS["high"]
    elif pct >= 50:
        tier = "medium"
        verdict = "suitable"
        verdict_msg = "✅ You have solid potential for the AI/ML field!"
        verdict_detail = (
            "You have a good foundation. With focused effort on the identified gaps, you can "
            "build a strong AI/ML career. Consider strengthening your mathematics and hands-on "
            "project experience."
        )
        color = "#f59e0b"
        roles = ROLE_RECS["medium"]
    else:
        tier = "low"
        verdict = "not_suitable"
        verdict_msg = "⚠️ AI/ML may not align with your current profile"
        verdict_detail = (
            "This doesn't mean AI/ML is off-limits — but it suggests you may benefit from "
            "strengthening your fundamentals first. Consider exploring adjacent fields while "
            "building your technical foundation."
        )
        color = "#ef4444"
        roles = []

    # Per-category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"earned": 0, "possible": 0, "correct": 0, "total": 0}
        categories[cat]["earned"] += r["points_earned"]
        categories[cat]["possible"] += r["points_possible"]
        categories[cat]["correct"] += int(r["is_correct"])
        categories[cat]["total"] += 1

    suggestions = []
    if tier == "low":
        suggestions = [
            "📚 Complete a Python fundamentals course (Codecademy, CS50P)",
            "📐 Brush up on statistics: Khan Academy Statistics is free and excellent",
            "🔍 Explore adjacent fields: Web Development, Mobile Apps, Cybersecurity",
            "🧩 Solve logic puzzles daily to sharpen reasoning skills",
            "🌱 Revisit AI/ML after 3–6 months of fundamentals work",
        ]
    elif tier == "medium":
        suggestions = [
            "🐍 Deepen Python skills: decorators, OOP, list comprehensions",
            "📊 Practice with real datasets on Kaggle",
            "🎓 Take Andrew Ng's Machine Learning Specialization (Coursera)",
            "🔧 Build 2–3 end-to-end ML projects for your portfolio",
            "☁️ Get a cloud certification (AWS ML Specialty or GCP Professional ML)",
        ]

    return {
        "earned": earned,
        "max_score": MAX_SCORE,
        "pct": pct,
        "tier": tier,
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "verdict_detail": verdict_detail,
        "color": color,
        "recommended_roles": roles,
        "suggestions": suggestions,
        "results": results,
        "categories": categories,
    }
