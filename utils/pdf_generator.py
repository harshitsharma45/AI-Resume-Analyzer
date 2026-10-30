from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def generate_candidate_pdf_report(candidate, job_title, result):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    y = 790
    for line in ["Resume Screening Report", f"Candidate: {candidate}", f"Role: {job_title}", f"Overall match: {result['overall_score']}%", f"Recommendation: {result['recommendation']}", "Matched skills: " + ", ".join(result["gap_analysis"]["matched_skills"]), "Missing skills: " + ", ".join(result["gap_analysis"]["missing_skills"])]:
        for i in range(0, len(line), 90):
            c.drawString(40, y, line[i:i+90]); y -= 22
    c.save()
    buffer.seek(0)
    return buffer.getvalue()
