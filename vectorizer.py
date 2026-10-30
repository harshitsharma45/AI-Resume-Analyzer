from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
class TFIDFMatcher:
    def calculate_similarity(self, resume_text, jd_text):
        if not (resume_text or "").strip() or not (jd_text or "").strip():
            return 0.0
        try:
            vectors = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform([resume_text, jd_text])
            return float(cosine_similarity(vectors[0], vectors[1])[0][0])
        except ValueError:
            return 0.0
