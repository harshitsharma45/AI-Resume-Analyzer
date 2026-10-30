import numpy as np
FEATURE_NAMES = ["tfidf_similarity", "skill_ratio", "matched_skills", "missing_skills", "years_experience", "normalized_experience", "education_score", "candidate_skills"]
def load_or_create_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    sim = rng.uniform(0, 1, n)
    ratio = np.clip(sim + rng.normal(0, .15, n), 0, 1)
    matched = rng.integers(0, 14, n)
    missing = rng.integers(0, 12, n)
    years = rng.uniform(0, 12, n)
    education = rng.choice([0.4, 1.0], n, p=[0.25, 0.75])
    candidate = matched + rng.integers(0, 12, n)
    X = np.column_stack([sim, ratio, matched, missing, years, np.clip(years/3, 0, 3), education, candidate])
    signal = .4 * sim + .4 * ratio + .12 * np.clip(years / 5, 0, 1) + .08 * education
    y = (signal + rng.normal(0, .08, n) >= .55).astype(int)
    return X, y
def get_training_features_and_labels():
    return load_or_create_dataset()
