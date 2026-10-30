from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RESUME_DIR = DATA_DIR / "resumes"
MODEL_DIR = DATA_DIR / "models"
for folder in (DATA_DIR, RESUME_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)
DEFAULT_WEIGHTS = {"tfidf_similarity": 0.35, "skills_match": 0.40, "experience_match": 0.15, "education_match": 0.10}
THRESHOLDS = {"shortlist": 70, "review": 45}
THEME_CONFIG = {"primary": "#2563EB", "background": "#F8FAFC"}
