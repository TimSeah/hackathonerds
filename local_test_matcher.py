import random
import numpy as np
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

# Emotion embedding dimensions (simulated psychological states)
EMOTION_DIMENSIONS = [
    "anxiety", "sadness", "anger", "confusion", 
    "hopelessness", "overwhelm", "fear", "frustration"
]

# Coping styles based on psychological research
COPING_STYLES = [
    "problem_focused",      # Direct action/solutions
    "emotion_focused",      # Process feelings first
    "avoidant",            # Distraction-based
    "social_support",      # Connection-driven
    "meaning_making"       # Reframing/perspective
]

CONVERSATION_PREFERENCES = [
    "direct_advice",
    "reflective_listening",
    "collaborative_problem_solving",
    "validation_focused"
]

# Updated weights for psychologically smart matching
WEIGHTS = {
    "emotional_similarity": 0.35,
    "experience_overlap": 0.25,
    "coping_style_match": 0.15,
    "availability_overlap": 0.15,
    "helper_reliability_score": 0.10
}

DISTRESS_MAP = {"Low": 0.3, "Medium": 0.6, "High": 1.0}
ENERGY_LEVELS = ["depleted", "low", "moderate", "high"]

# ---------------------------
# SYNTHETIC DATA GENERATORS
# ---------------------------

def generate_emotion_embedding():
    """Generate normalized emotion embedding vector"""
    vec = np.random.rand(len(EMOTION_DIMENSIONS))
    return vec / np.linalg.norm(vec)  # Normalize


def generate_availability_windows():
    """Generate weekly availability (hour blocks)"""
    # Simulate 24-hour blocks across 7 days
    return {
        day: [random.choice([0, 1]) for _ in range(24)]
        for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    }


def generate_helper():
    """Generate Helper/Listener profile"""
    return {
        "user_id": fake.first_name(),
        "role": "helper",
        "themes_experience": {t: random.uniform(0.3, 1.0) for t in THEMES},
        
        # Emotion embedding - their emotional capacity/understanding
        "emotion_embedding": generate_emotion_embedding(),
        
        # Coping style expertise (what styles they understand/support best)
        "coping_style_expertise": {
            style: random.uniform(0.5, 1.0) for style in COPING_STYLES
        },
        
        # Conversation preferences
        "conversation_style": {
            pref: random.random() for pref in CONVERSATION_PREFERENCES
        },
        
        # Availability windows
        "availability_windows": generate_availability_windows(),
        
        # Energy level patterns
        "energy_level": random.choice(ENERGY_LEVELS),
        "energy_consistency": random.uniform(0.6, 1.0),  # How reliable their energy is
        
        # Reliability metrics
        "reliability_score": random.uniform(0.7, 1.0),
        "response_rate": random.uniform(0.75, 1.0),
        "completion_rate": random.uniform(0.80, 1.0),
        
        # Support strengths
        "support_strengths": {
            "empathy": random.uniform(0.6, 1.0),
            "lived_experience": random.uniform(0.5, 1.0),
            "active_listening": random.uniform(0.6, 1.0),
            "boundary_setting": random.uniform(0.5, 1.0)
        }
    }


def generate_seeker():
    """Generate Seeker profile (extracted from AI chat/vent)"""
    return {
        "user_id": fake.first_name(),
        "role": "seeker",
        "themes": [
            {"name": random.choice(THEMES), "intensity": round(random.uniform(0.6, 1.0), 2)}
            for _ in range(random.randint(1, 3))
        ],
        
        # Emotion embedding - current emotional state
        "emotion_embedding": generate_emotion_embedding(),
        
        # Preferred coping style (extracted from vent/chat)
        "coping_style_preference": {
            style: random.random() for style in COPING_STYLES
        },
        
        # What they're seeking
        "conversation_preference": {
            pref: random.random() for pref in CONVERSATION_PREFERENCES
        },
        
        # Availability
        "availability_windows": generate_availability_windows(),
        
        # Current state
        "energy_level": random.choice(ENERGY_LEVELS),
        "distress_level": random.choice(["Low", "Medium", "High"]),
        "urgency": random.uniform(0.3, 1.0)  # How soon they need support
    }


# ---------------------------
# PSYCHOLOGICALLY SMART SCORING FUNCTIONS
# ---------------------------

def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(vec1, vec2)
    norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return dot_product / norm_product if norm_product > 0 else 0


def emotion_embedding_similarity(seeker, helper):
    """
    0.35 weight: Measures emotional state alignment
    High score = helper understands seeker's emotional landscape
    """
    return cosine_similarity(
        seeker["emotion_embedding"],
        helper["emotion_embedding"]
    )


def experience_overlap_score(seeker, helper):
    """
    0.25 weight: Measures lived experience overlap with themes
    High score = helper has been through similar issues
    """
    overlap = 0.0
    total_intensity = 0.0
    
    for theme in seeker["themes"]:
        theme_name = theme["name"]
        intensity = theme["intensity"]
        helper_exp = helper["themes_experience"].get(theme_name, 0)
        
        overlap += intensity * helper_exp
        total_intensity += intensity
    
    return overlap / total_intensity if total_intensity > 0 else 0


def coping_style_compatibility(seeker, helper):
    """
    0.15 weight: Measures alignment in coping approaches
    High score = helper supports seeker's natural coping style
    """
    compatibility = 0.0
    
    for style in COPING_STYLES:
        seeker_pref = seeker["coping_style_preference"].get(style, 0)
        helper_expertise = helper["coping_style_expertise"].get(style, 0)
        compatibility += seeker_pref * helper_expertise
    
    return compatibility / len(COPING_STYLES)


def conversation_preference_match(seeker, helper):
    """
    Measures alignment in conversation style preferences
    Direct advice vs reflective listening vs collaborative problem-solving
    """
    match_score = 0.0
    
    for pref in CONVERSATION_PREFERENCES:
        seeker_want = seeker["conversation_preference"].get(pref, 0)
        helper_style = helper["conversation_style"].get(pref, 0)
        match_score += seeker_want * helper_style
    
    return match_score / len(CONVERSATION_PREFERENCES)


def availability_overlap_score(seeker, helper):
    """
    0.15 weight: Measures schedule compatibility
    High score = many overlapping time windows
    """
    total_overlap = 0
    total_slots = 0
    
    for day in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
        seeker_slots = seeker["availability_windows"][day]
        helper_slots = helper["availability_windows"][day]
        
        for s_avail, h_avail in zip(seeker_slots, helper_slots):
            if s_avail == 1 and h_avail == 1:
                total_overlap += 1
            total_slots += 1
    
    return total_overlap / total_slots if total_slots > 0 else 0


def energy_level_compatibility(seeker, helper):
    """
    Bonus: Energy matching (integrated into conversation match)
    Depleted seeker shouldn't get high-energy helper (overwhelming)
    """
    energy_map = {"depleted": 0, "low": 1, "moderate": 2, "high": 3}
    s_energy = energy_map[seeker["energy_level"]]
    h_energy = energy_map[helper["energy_level"]]
    
    # Prefer helpers 1-2 levels above seeker (supportive but not overwhelming)
    diff = h_energy - s_energy
    if 0 <= diff <= 2:
        return 1.0
    elif diff < 0:
        return 0.7  # Helper lower energy - still okay
    else:
        return 0.5  # Helper too high energy - might overwhelm


def helper_reliability_score(helper):
    """
    0.10 weight: Composite reliability metric
    Based on past performance, response rate, completion rate
    """
    return (
        0.5 * helper["reliability_score"] +
        0.3 * helper["response_rate"] +
        0.2 * helper["completion_rate"]
    )


def compute_dha_match_score(seeker, helper):
    """
    Dha Matching Algorithm - Psychologically Smart
    
    Weighted formula:
    score = 0.35 * emotional_similarity +
            0.25 * experience_overlap +
            0.15 * coping_style_match +
            0.15 * availability_overlap +
            0.10 * helper_reliability_score
    
    Plus conversation preference & energy bonuses
    """
    
    # Core weighted components
    emotional_sim = emotion_embedding_similarity(seeker, helper)
    experience_over = experience_overlap_score(seeker, helper)
    coping_match = coping_style_compatibility(seeker, helper)
    availability_over = availability_overlap_score(seeker, helper)
    reliability = helper_reliability_score(helper)
    
    core_score = (
        WEIGHTS["emotional_similarity"] * emotional_sim +
        WEIGHTS["experience_overlap"] * experience_over +
        WEIGHTS["coping_style_match"] * coping_match +
        WEIGHTS["availability_overlap"] * availability_over +
        WEIGHTS["helper_reliability_score"] * reliability
    )
    
    # Bonus adjustments (up to +0.15)
    conversation_bonus = 0.10 * conversation_preference_match(seeker, helper)
    energy_bonus = 0.05 * energy_level_compatibility(seeker, helper)
    
    final_score = core_score + conversation_bonus + energy_bonus
    
    return round(final_score, 3), {
        "emotional_similarity": round(emotional_sim, 3),
        "experience_overlap": round(experience_over, 3),
        "coping_style_match": round(coping_match, 3),
        "availability_overlap": round(availability_over, 3),
        "reliability_score": round(reliability, 3),
        "conversation_bonus": round(conversation_bonus, 3),
        "energy_bonus": round(energy_bonus, 3)
    }


# ---------------------------
# DHA MATCH ENGINE
# ---------------------------

def match_seeker_to_helpers(seeker, helpers, top_k=5, min_score=0.5):
    """
    Main matching function - returns top K helpers for a seeker
    
    Args:
        seeker: Seeker profile dict
        helpers: List of Helper profile dicts
        top_k: Number of top matches to return
        min_score: Minimum score threshold (filters out poor matches)
    
    Returns:
        List of (score, helper_id, breakdown) tuples
    """
    scored = []
    
    for helper in helpers:
        score, breakdown = compute_dha_match_score(seeker, helper)
        
        # Only include if meets minimum threshold
        if score >= min_score:
            scored.append((score, helper["user_id"], breakdown))
    
    # Sort by score descending
    scored.sort(reverse=True, key=lambda x: x[0])
    
    return scored[:top_k]


def discover_by_theme(theme_name, helpers, top_k=10):
    """
    Netflix-style discovery: Browse helpers by theme
    Returns helpers sorted by experience in that theme
    
    Args:
        theme_name: Name of the theme to browse
        helpers: List of Helper profile dicts
        top_k: Number of results to return
    
    Returns:
        List of (experience_score, helper_id) tuples
    """
    themed = []
    
    for helper in helpers:
        exp_score = helper["themes_experience"].get(theme_name, 0)
        reliability = helper_reliability_score(helper)
        
        # Combined score: 70% experience, 30% reliability
        combined = 0.7 * exp_score + 0.3 * reliability
        themed.append((round(combined, 3), helper["user_id"]))
    
    themed.sort(reverse=True)
    return themed[:top_k]


# ---------------------------
# RUN LOCAL TEST - DHA ALGORITHM v3
# ---------------------------

if __name__ == "__main__":
    
    print("\n" + "="*60)
    print("DHA MATCHING ALGORITHM v3 - PSYCHOLOGICALLY SMART")
    print("="*60)
    
    # Generate synthetic helper database (mixed youth-youth, youth-senior)
    print("\n[1] Generating 50 Helper/Listener profiles...")
    helpers_db = [generate_helper() for _ in range(50)]
    print(f"    ✓ {len(helpers_db)} helpers ready")
    
    # Test seeker-to-helper matching
    print("\n[2] Testing Seeker → Helper Matching (Top 3)...")
    print("-" * 60)
    
    for i in range(3):
        seeker = generate_seeker()
        matches = match_seeker_to_helpers(seeker, helpers_db, top_k=3)
        
        print(f"\n🆘 SEEKER {i+1}: {seeker['user_id']}")
        print(f"   Themes: {[t['name'] for t in seeker['themes']]}")
        print(f"   Distress: {seeker['distress_level']} | Energy: {seeker['energy_level']}")
        print(f"   Seeking: {max(seeker['conversation_preference'], key=seeker['conversation_preference'].get)}")
        
        print(f"\n   🤝 TOP MATCHES:")
        for rank, (score, helper_id, breakdown) in enumerate(matches, 1):
            print(f"   {rank}. {helper_id} (Score: {score})")
            print(f"      └─ Emotional: {breakdown['emotional_similarity']} | "
                  f"Experience: {breakdown['experience_overlap']} | "
                  f"Coping: {breakdown['coping_style_match']}")
            print(f"         Availability: {breakdown['availability_overlap']} | "
                  f"Reliability: {breakdown['reliability_score']}")
    
    # Test theme discovery
    print("\n" + "="*60)
    print("[3] Testing Netflix-Style Theme Discovery...")
    print("-" * 60)
    
    test_theme = random.choice(THEMES)
    discoveries = discover_by_theme(test_theme, helpers_db, top_k=5)
    
    print(f"\n🔍 Browsing: '{test_theme}'")
    print(f"\n   Top Helpers with this experience:")
    for rank, (score, helper_id) in enumerate(discoveries, 1):
        print(f"   {rank}. {helper_id} (Experience Score: {score})")
    
    print("\n" + "="*60)
    print("✅ Dha Algorithm v3 Test Complete")
    print("="*60)
    print("\nKey Features Demonstrated:")
    print("  • Emotion embedding similarity (35%)")
    print("  • Experience overlap scoring (25%)")
    print("  • Coping style compatibility (15%)")
    print("  • Availability window matching (15%)")
    print("  • Helper reliability scores (10%)")
    print("  • Conversation preference bonuses")
    print("  • Energy level matching")
    print("  • Theme discovery browsing")
    print()
