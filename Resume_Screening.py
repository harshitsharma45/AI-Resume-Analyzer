import streamlit as st
import pandas as pd
from config import DEFAULT_WEIGHTS, THRESHOLDS
from database.db_manager import DatabaseManager
from nlp.text_extractor import extract_text_from_file
from matching.matcher import ResumeJobMatcher
from models.ml_models import MLModelManager
from models.ann_model import ANNModelManager
from models.model_evaluator import evaluate_all_models
from utils.sample_data import SAMPLE_JOB_DESCRIPTIONS, SAMPLE_RESUMES
from utils.pdf_generator import generate_candidate_pdf_report
from utils.ui_components import create_score_gauge, create_radar_competency_chart, create_skills_category_bar_chart, create_confusion_matrix_heatmap, create_ann_training_history_chart

st.set_page_config(page_title="AI Resume Screening", page_icon="💼", layout="wide")
st.title("💼 AI Resume Screening & Job Matching")
st.caption("Academic demo • transparent matching • human review required")
st.warning("Educational demonstration only. Do not use scores or synthetic-model predictions as hiring decisions. Obtain consent before uploading resumes.")
@st.cache_resource
def get_db_manager(): return DatabaseManager()
@st.cache_resource
def get_ml_manager(): return MLModelManager().load_or_train()
@st.cache_resource
def get_ann_manager(): return ANNModelManager().load_or_train()
matcher = ResumeJobMatcher()
nav = st.sidebar.radio("Navigation", ["🎯 Single Resume Match", "👥 Batch Screening", "📋 Job Description Center", "🧠 ML & Neural Network Center", "🗄️ Screening History", "🎓 Project Overview"])

def get_jd(key):
    choice = st.selectbox("Job description template", ["Custom"] + list(SAMPLE_JOB_DESCRIPTIONS), key=key+"choice")
    default = SAMPLE_JOB_DESCRIPTIONS.get(choice, "")
    return st.text_area("Job description", value=default, height=160, key=key+choice)

def show_result(result, candidate, job):
    a,b,c = st.columns(3)
    a.metric("Overall match", f"{result['overall_score']:.1f}%")
    b.metric("Recommendation", result["recommendation"])
    c.metric("Matched skills", len(result["gap_analysis"]["matched_skills"]))
    left,right = st.columns(2)
    with left: st.plotly_chart(create_score_gauge(result["overall_score"]), use_container_width=True)
    with right: st.plotly_chart(create_radar_competency_chart(result["subscores"]), use_container_width=True)
    st.subheader("Skills gap")
    st.write("**Matched:**", ", ".join(result["gap_analysis"]["matched_skills"]) or "None detected")
    st.write("**Missing:**", ", ".join(result["gap_analysis"]["missing_skills"]) or "None detected")
    st.plotly_chart(create_skills_category_bar_chart(result["candidate_analysis"]["skills"]["categories"]), use_container_width=True)
    with st.expander("Score breakdown and heuristic extraction"):
        st.json({"subscores": result["subscores"], "experience": result["candidate_analysis"]["experience"], "education": result["candidate_analysis"]["education"], "weights": DEFAULT_WEIGHTS})
    st.download_button("📄 Download candidate PDF report", generate_candidate_pdf_report(candidate, job, result), file_name="screening_report.pdf", mime="application/pdf")

if nav == "🎯 Single Resume Match":
    st.header("Single resume match & analysis")
    job_title = st.text_input("Job title", "Data Analyst")
    jd = get_jd("single")
    mode = st.radio("Resume source", ["Upload", "Sample", "Paste text"], horizontal=True)
    candidate, resume = "Candidate", ""
    if mode == "Upload":
        uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
        if uploaded:
            candidate = uploaded.name
            try: resume = extract_text_from_file(uploaded)
            except Exception as exc: st.error(f"Unable to read resume: {exc}")
    elif mode == "Sample":
        candidate = st.selectbox("Sample candidate", list(SAMPLE_RESUMES))
        resume = SAMPLE_RESUMES[candidate]
        st.text_area("Sample resume", resume, height=140, disabled=True)
    else:
        candidate = st.text_input("Candidate name", "Candidate")
        resume = st.text_area("Resume text", height=180)
    if st.button("Analyze resume", type="primary"):
        if not resume.strip() or not jd.strip(): st.error("Provide a readable resume and job description.")
        else:
            result = matcher.match(resume, jd)
            st.session_state["single_result"] = (result, candidate, job_title)
            get_db_manager().save_screening(candidate, job_title, result["overall_score"], result["recommendation"])
    if "single_result" in st.session_state:
        result, candidate, job_title = st.session_state["single_result"]
        show_result(result, candidate, job_title)
        with st.expander("Experimental synthetic-data ML predictions (not for decisions)"):
            st.json(get_ml_manager().predict(result["feature_vector"]))
            st.json({"ann_mlp": get_ann_manager().predict(result["feature_vector"])})

elif nav == "👥 Batch Screening":
    st.header("Batch screening & ranking")
    jd = get_jd("batch")
    title = st.text_input("Role title", "Data Analyst")
    mode = st.radio("Candidate source", ["Upload multiple", "Sample resumes"], horizontal=True)
    uploads = st.file_uploader("Upload multiple PDF/DOCX/TXT resumes", type=["pdf", "docx", "txt"], accept_multiple_files=True) if mode == "Upload multiple" else []
    if st.button("Rank candidates", type="primary"):
        rows = []
        sources = [(x.name, x) for x in uploads] if mode == "Upload multiple" else list(SAMPLE_RESUMES.items())
        if not jd.strip() or not sources: st.error("Provide a job description and at least one resume.")
        else:
            for name, source in sources:
                try:
                    resume = extract_text_from_file(source) if mode == "Upload multiple" else source
                    result = matcher.match(resume, jd)
                    rows.append({"Candidate": name, "Score": result["overall_score"], "Recommendation": result["recommendation"], "Matched skills": ", ".join(result["gap_analysis"]["matched_skills"]), "Missing skills": ", ".join(result["gap_analysis"]["missing_skills"])})
                    get_db_manager().save_screening(name, title, result["overall_score"], result["recommendation"])
                except Exception as exc: st.warning(f"Skipped {name}: {exc}")
            st.session_state["batch_results"] = rows
    if st.session_state.get("batch_results"):
        df = pd.DataFrame(st.session_state["batch_results"]).sort_values("Score", ascending=False)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.bar_chart(df.set_index("Candidate")["Score"])
        st.download_button("Download ranking CSV", df.to_csv(index=False).encode("utf-8"), "candidate_ranking.csv", "text/csv")

elif nav == "📋 Job Description Center":
    st.header("Job description center")
    for title, description in SAMPLE_JOB_DESCRIPTIONS.items():
        with st.expander(title): st.write(description)
    st.info("Choose a template on the screening pages or paste your own job description.")

elif nav == "🧠 ML & Neural Network Center":
    st.header("ML & Neural Network Center")
    st.info("Models are trained on synthetic generated data and are only demonstrations, not validated hiring models.")
    if st.button("Evaluate models"):
        with st.spinner("Training and evaluating..."):
            st.session_state["evaluation"] = evaluate_all_models()
    if "evaluation" in st.session_state:
        evaluations = st.session_state["evaluation"]
        st.dataframe(pd.DataFrame([{"Model": name, "Accuracy": v["accuracy"], "ROC AUC": v["auc"]} for name, v in evaluations.items()]), hide_index=True)
        chosen = st.selectbox("Confusion matrix", list(evaluations))
        st.plotly_chart(create_confusion_matrix_heatmap(evaluations[chosen]["confusion_matrix"]), use_container_width=True)
        st.subheader("ANN training loss")
        st.plotly_chart(create_ann_training_history_chart(get_ann_manager().history), use_container_width=True)

elif nav == "🗄️ Screening History":
    st.header("Screening history")
    st.caption("SQLite history is local to the running instance and may be lost on cloud restarts.")
    history = get_db_manager().get_history()
    if history:
        df = pd.DataFrame(history, columns=["Timestamp UTC", "Candidate", "Role", "Score", "Recommendation"])
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Export history CSV", df.to_csv(index=False).encode(), "screening_history.csv", "text/csv")
    else: st.info("No screenings saved yet.")
    if st.button("Clear local history"):
        get_db_manager().clear_history()
        st.rerun()

else:
    st.header("Academic project overview")
    st.markdown("""**Objective:** Compare resume content to job requirements using NLP, TF-IDF similarity, skills gaps, and weighted scoring.

**Pipeline:** Upload → text extraction → skill/experience/education heuristics → TF-IDF → weighted match score → visualization → export.

**Algorithms:** TF-IDF cosine similarity; Logistic Regression; Random Forest; ANN using scikit-learn MLP.

**Evaluation:** ML models are trained and tested on generated synthetic feature vectors. They are not validated for actual recruitment decisions.

**Limitations:** No scanned-PDF OCR, keyword-only skill detection, heuristic experience extraction, ephemeral local SQLite storage, and synthetic model labels. Always require human judgment, fairness assessment, and consent.""")
