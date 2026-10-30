# AI Resume Screening & Job Matching System

A Streamlit academic demonstration that extracts text from PDF/DOCX/TXT resumes, detects skills, compares resumes to job descriptions with TF-IDF, calculates a transparent weighted match score, ranks multiple applicants, exports CSV/PDF, and demonstrates logistic regression, random forest and an ANN (scikit-learn MLP).

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run Resume_Screening.py
```

## Deploy to Streamlit Community Cloud
1. Upload **all** project files and folders to GitHub, preserving their structure.
2. In Streamlit Cloud, select repository, branch `main`, and main file path `Resume_Screening.py`.
3. Deploy. `requirements.txt` is at the repository root.

## Notes and limitations
- **Educational demonstration only.** Do not use the score or model predictions to automatically accept/reject candidates or make employment decisions. Human review and validation are required.
- The ML and ANN models train on **synthetic** examples, not real hiring outcomes. Their probabilities are **not calibrated hiring probabilities**.
- Text extraction does not OCR scanned PDFs; use text-based PDFs, DOCX or TXT.
- Skills are detected by keyword matching; experience and education detection are heuristic.
- SQLite history is stored on the Streamlit instance and can disappear after restart; not suitable for persistent production storage.
- Do not upload personal data without consent. Avoid sensitive attributes and audit for fairness.
- TensorFlow is intentionally replaced by scikit-learn's `MLPClassifier` for lighter, more reliable Cloud deployment.
