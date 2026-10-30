def analyze_skill_gap(candidate_skills, job_skills):
    candidate, required = set(candidate_skills), set(job_skills)
    matched = sorted(candidate & required)
    missing = sorted(required - candidate)
    return {"matched_skills": matched, "missing_skills": missing, "extra_skills": sorted(candidate - required), "match_percentage": round(100 * len(matched) / len(required), 2) if required else 0.0}
