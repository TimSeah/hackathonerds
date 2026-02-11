import random
from faker import Faker

fake = Faker()

# ---------------------------
# CONFIG
# ---------------------------

THEMES = [
    "Exam Stress / Academic Pressure",
    "Family Problems",
    "Friendship / Social Issues",
    "Burnout / Emotional Exhaustion",
    "Loneliness / Isolation",
    "Life Direction / Purpose",
    "Self-Confidence / Self-Esteem"
]

WEIGHTS = {
    "theme": 0.35,
    "support": 0.20,
    "emotion": 0.15,
    "style": 0.15,
    "quality": 0.10,
    "availability": 0.05
}

DISTRESS_MAP = {"Low": 0.3, "Medium": 0.6, "High": 1.0}

# ---------------------------
# SYNTHETIC DATA GENERATORS
# ---------------------------

def generate_senior():
    return {
        "senior_id": fake.first_name(),
        "themes_experience": {t: random.random() for t in THEMES},
        "support_strength": {
            "empathy": random.random(),
            "advice": random.random(),
            "listening": random.random(),
            "motivation": random.random()
        },
        "communication_style": {
            "talkative": random.random(),
            "reserved": random.random(),
            "direct": random.random(),
            "reflective": random.random(),
            "advice_receptive": random.random(),
            "listening_focused": random.random()
        },
        "comfort_with_distress": random.choice([0.3, 0.6, 1.0]),
        "rating_score": random.uniform(0.6, 1.0),
        "availability": random.random()
    }


def generate_youth():
    return {
        "themes": [
            {"name": random.choice(THEMES), "confidence": round(random.uniform(0.6, 1.0), 2)}
        ],
        "support_type": {
            "primary": {"type": random.choice(["Emotional Empathy", "Practical Advice", "Active Listening"]), "confidence": 0.8}
        },
        "emotional_state": {
            "distress_level": random.choice(["Low", "Medium", "High"])
        },
        "communication_style": {
            "talkative": random.random(),
            "reserved": random.random(),
            "direct": random.random(),
            "reflective": random.random(),
            "advice_receptive": random.random(),
            "listening_focused": random.random()
        }
    }


# ---------------------------
# SCORING FUNCTIONS
# ---------------------------

def dot(d1, d2):
    return sum(d1.get(k, 0) * d2.get(k, 0) for k in d1.keys())


def compute_match_score(youth, senior):

    # Theme score
    theme_score = sum(
        t["confidence"] * senior["themes_experience"].get(t["name"], 0)
        for t in youth["themes"]
    )

    # Support score
    support_map = {
        "Emotional Empathy": "empathy",
        "Practical Advice": "advice",
        "Active Listening": "listening"
    }
    support_type = youth["support_type"]["primary"]["type"]
    support_score = senior["support_strength"].get(support_map[support_type], 0)

    # Emotion score
    y = DISTRESS_MAP[youth["emotional_state"]["distress_level"]]
    s = senior["comfort_with_distress"]
    emotion_score = 1 - abs(y - s)

    # Style score
    style_score = dot(youth["communication_style"], senior["communication_style"])

    # Final weighted score
    final_score = (
        WEIGHTS["theme"] * theme_score +
        WEIGHTS["support"] * support_score +
        WEIGHTS["emotion"] * emotion_score +
        WEIGHTS["style"] * style_score +
        WEIGHTS["quality"] * senior["rating_score"] +
        WEIGHTS["availability"] * senior["availability"]
    )

    return round(final_score, 3)


# ---------------------------
# MATCH ENGINE
# ---------------------------

def match(youth, seniors, top_k=3):

    scored = []
    for s in seniors:
        score = compute_match_score(youth, s)
        scored.append((score, s["senior_id"]))

    scored.sort(reverse=True)
    return scored[:top_k]


# ---------------------------
# RUN LOCAL TEST
# ---------------------------

if __name__ == "__main__":

    print("\nGenerating synthetic senior database...")
    seniors_db = [generate_senior() for _ in range(50)]

    print("Running 5 random youth match tests...\n")

    for i in range(5):
        youth = generate_youth()
        results = match(youth, seniors_db)

        print(f"Test {i+1}")
        print("Youth theme:", youth["themes"][0]["name"])
        print("Top Matches:", results)
        print("-" * 40)
