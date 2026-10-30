import re
import numpy as np
from config import DEFAULT_WEIGHTS, THRESHOLDS
from matching.vectorizer import TFIDFMatcher
from matching.gap_analyzer import analyze_skill_gap
from nlp.entity_extractor import analyze_resume_nlp, extract_skills, extract_experience, extract_education
class ResumeJobMatcher:
    def __init__(self, weights=None):
        self.weights = weights or DEFAULT_WEIGHTS
        self.tfidf_matcher = TFIDFMatcher()
    def evaluate_experience_score(self, years, jd_text):
        required = extract_experience(jd_text)["total_years"]
        return {"score": 100.0 if required <= 0 else round(min(100.0, years / required * 100), 2), "required_years": required}
    def evaluate_education_score(self, education, jd_text):
        requested = extract_education(jd_text)["highest"]
        actual = education["highest"]
        ranks = {"unknown": 0, "graduate": 1, "postgraduate": 2}
        return {"score": 100.0 if requested == "unknown" or ranks[actual] >= ranks[requested] else 40.0}
    def match(self, resume_text, jd_text):
        if not resume_text.strip() or not jd_text.strip():
            raise ValueError("Resume and job description must contain readable text")
        analysis = analyze_resume_nlp(resume_text)
        candidate = analysis["skills"]["all_skills"]
        required = extract_skills(jd_text)["all_skills"]
        gap = analyze_skill_gap(candidate, required)
        similarity = self.tfidf_matcher.calculate_similarity(resume_text, jd_text)
        tfidf_score = similarity * 100
        skills_score = gap["match_percentage"] if required else tfidf_score
        exp_score = self.evaluate_experience_score(analysis["experience"]["total_years"], jd_text)["score"]
        edu_score = self.evaluate_education_score(analysis["education"], jd_text)["score"]
        subscores = {"tfidf_score": round(tfidf_score, 2), "skill_score": round(skills_score, 2), "experience_score": exp_score, "education_score": edu_score}
        score = sum(self.weights[k] * v for k, v in zip(["tfidf_similarity", "skills_match", "experience_match", "education_match"], subscores.values()))
        features = [similarity, len(gap["matched_skills"]) / max(len(required), 1), len(gap["matched_skills"]), len(gap["missing_skills"]), analysis["experience"]["total_years"], min(3.0, analysis["experience"]["total_years"] / 3), edu_score / 100, len(candidate)]
        return {"overall_score": round(float(np.clip(score, 0, 100)), 2), "subscores": subscores, "gap_analysis": gap, "feature_vector": features, "candidate_analysis": analysis, "recommendation": "Shortlist" if score >= THRESHOLDS["shortlist"] else "Review" if score >= THRESHOLDS["review"] else "Low Match"}
