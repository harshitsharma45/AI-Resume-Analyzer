from config import RESUME_DIR
SAMPLE_JOB_DESCRIPTIONS = {
    "Data Analyst": "Data Analyst: Bachelor degree, 1 year experience, Python, SQL, Excel, Power BI, statistics, data analysis, communication.",
    "ML Engineer": "Machine Learning Engineer: Bachelor degree, 2 years experience, Python, machine learning, scikit-learn, NLP, TensorFlow, SQL.",
    "Business Analyst": "Business Analyst: MBA preferred, 2 years experience, Excel, Power BI, stakeholder management, communication, project management, SQL."
}
SAMPLE_RESUMES = {
    "Alex - Data Analyst": "Bachelor degree in Computer Science. 2 years experience in Python, SQL, Excel, Power BI, pandas, statistics, data analysis and communication.",
    "Sam - ML Engineer": "Master degree. 3 years experience in Python, machine learning, NLP, TensorFlow, scikit-learn, SQL and deep learning.",
    "Jordan - Business Analyst": "MBA with 2 years experience in Excel, project management, stakeholder management, marketing, communication and Power BI."
}
def initialize_demo_files():
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
