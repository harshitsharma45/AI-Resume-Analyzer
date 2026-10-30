import re
SKILL_GROUPS = {
    "Programming": ["python", "java", "javascript", "sql", "r", "c++", "html", "css"],
    "Analytics": ["excel", "power bi", "tableau", "pandas", "numpy", "statistics", "data analysis", "data visualization"],
    "AI / ML": ["machine learning", "deep learning", "nlp", "tensorflow", "pytorch", "scikit-learn", "neural network"],
    "Business": ["communication", "leadership", "project management", "stakeholder management", "agile", "marketing", "sales"]
}
def extract_skills(text):
    text = (text or "").lower()
    groups = {group: sorted({skill for skill in skills if re.search(r"(?<![\w])" + re.escape(skill) + r"(?![\w])", text)}) for group, skills in SKILL_GROUPS.items()}
    return {"categories": groups, "all_skills": sorted(set().union(*map(set, groups.values())))}
def extract_experience(text):
    years = [float(n) for n in re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*years?", (text or "").lower())]
    return {"total_years": min(max(years, default=0.0), 40.0)}
def extract_education(text):
    text = (text or "").lower()
    levels = [term for term in ["phd", "doctorate", "master", "mba", "m.tech", "mca", "bachelor", "b.tech", "bca", "bsc", "degree"] if term in text]
    return {"qualifications": levels, "highest": ("postgraduate" if any(x in levels for x in ["phd", "doctorate", "master", "mba", "m.tech", "mca"]) else "graduate" if levels else "unknown")}
def analyze_resume_nlp(text):
    return {"skills": extract_skills(text), "experience": extract_experience(text), "education": extract_education(text)}
