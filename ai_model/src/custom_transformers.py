import re
import ast
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Cleaning functions (taken directly from steps 3 and 4) ---

def clean_text_general(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = text.replace("\\n", " ").replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_skill(skill):
    skill = str(skill).strip().lower()
    return re.sub(r"\s+", " ", skill)


def skills_to_text(x):
    try:
        items = ast.literal_eval(x)
        return " ".join(normalize_skill(s) for s in items if str(s).strip())
    except Exception:
        return ""


def parse_skills_set(x):
    try:
        items = ast.literal_eval(x)
        return set(normalize_skill(s) for s in items if str(s).strip())
    except Exception:
        return set()


def required_to_text(x):
    if pd.isna(x):
        return ""
    return " ".join(normalize_skill(s) for s in x.split("\n") if s.strip())


def parse_required_set(x):
    if pd.isna(x):
        return set()
    return set(normalize_skill(s) for s in x.split("\n") if s.strip())


NUMERIC_COLS = [
    "skills_overlap", "experience_years_min",
    "has_career_objective", "has_certifications", "has_languages",
    "has_extra_curricular", "has_skills_required",
    "has_experience_requirement", "has_age_requirement",
]


class ResumeFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Takes a DataFrame of RAW columns (as in the original CSV) and
    produces a DataFrame of features ready for the ColumnTransformer:
    resume_text, job_text, job_position_name, + all numeric columns.
    Combines in a single step what was done in steps 3 and 4.
    """

    def fit(self, X, y=None):
        # Nothing to learn here: only deterministic transformations
        return self

    def transform(self, X):
        X = X.copy()

        career_objective_clean = X["career_objective"].apply(clean_text_general)
        responsibilities_1_clean = X["responsibilities.1"].apply(clean_text_general)
        educational_req_clean = X["educationaL_requirements"].apply(clean_text_general)

        resume_text = career_objective_clean + " " + X["skills"].apply(skills_to_text)
        job_text = (
            responsibilities_1_clean + " " +
            educational_req_clean + " " +
            X["skills_required"].apply(required_to_text)
        )

        skills_set = X["skills"].apply(parse_skills_set)
        required_set = X["skills_required"].apply(parse_required_set)
        skills_overlap = [
            len(a & b) for a, b in zip(skills_set, required_set)
        ]

        experience_years_min = (
            X["experiencere_requirement"].astype(str).str.extract(r"(\d+)").astype(float)[0]
            .fillna(0)
        )

        out = pd.DataFrame({
            "resume_text": resume_text,
            "job_text": job_text,
            "job_position_name": X["job_position_name"],
            "skills_overlap": skills_overlap,
            "experience_years_min": experience_years_min,
            "has_career_objective": X["career_objective"].notna().astype(int),
            "has_certifications": X["certification_providers"].notna().astype(int),
            "has_languages": X["languages"].notna().astype(int),
            "has_extra_curricular": X["extra_curricular_activity_types"].notna().astype(int),
            "has_skills_required": X["skills_required"].notna().astype(int),
            "has_experience_requirement": X["experiencere_requirement"].notna().astype(int),
            "has_age_requirement": X["age_requirement"].notna().astype(int),
        })
        return out


class TextSimilarityVectorizer(BaseEstimator, TransformerMixin):
    """
    Receives a DataFrame with 2 columns [resume_text, job_text].
    Vectorizes both with a SHARED vocabulary TF-IDF (fit on the two
    concatenated columns), then calculates cosine similarity row by row.
    Output: hstack(resume_tfidf, job_tfidf, similarity) — same logic as in step 4.
    """

    def __init__(self, max_features=200, ngram_range=(1, 2), stop_words="english"):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.stop_words = stop_words

    def _split_cols(self, X):
        if isinstance(X, pd.DataFrame):
            return X["resume_text"], X["job_text"]
        X = np.asarray(X)
        return pd.Series(X[:, 0]), pd.Series(X[:, 1])

    def fit(self, X, y=None):
        resume_col, job_col = self._split_cols(X)
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words=self.stop_words,
        )
        self.vectorizer_.fit(pd.concat([resume_col, job_col]))
        return self

    def transform(self, X):
        resume_col, job_col = self._split_cols(X)
        resume_tfidf = self.vectorizer_.transform(resume_col)
        job_tfidf = self.vectorizer_.transform(job_col)

        sims = np.array([
            cosine_similarity(resume_tfidf[i], job_tfidf[i])[0, 0]
            for i in range(resume_tfidf.shape[0])
        ]).reshape(-1, 1)

        return sparse.hstack([resume_tfidf, job_tfidf, sparse.csr_matrix(sims)]).tocsr()