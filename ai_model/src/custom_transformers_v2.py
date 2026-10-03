import ast
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# --- Parsing helpers, matching notebook 11 (map_current_*_to_canonical) ---

def normalize_skill(skill):
    return " ".join(str(skill).strip().lower().split())


def safe_literal_list(x):
    try:
        return ast.literal_eval(x)
    except Exception:
        return []


def parse_skills_list(x):
    return [normalize_skill(s) for s in safe_literal_list(x) if str(s).strip()]


def parse_required_list(x):
    if pd.isna(x):
        return []
    return [normalize_skill(s) for s in str(x).split("\n") if s.strip()]


def join_list_col(x):
    """Matches notebook 12, cell 1 (building education_text)."""
    items = safe_literal_list(x)
    return " ".join(str(i) for i in items if str(i).strip() and str(i).lower() != "nan")


# --- Build "canonical" text, EXACTLY as in map_current_*_to_canonical (notebook 11) ---
# No cleaning or sorting: reproduce the mapping exactly, including the literal "nan" for missing fields.

def build_resume_raw_text(row):
    skills = parse_skills_list(row.get("skills"))
    parts = [str(row.get("career_objective", "")), " ".join(skills)]
    return " ".join(p for p in parts if p).strip()


def build_job_raw_text(row):
    skills_required = parse_required_list(row.get("skills_required"))
    parts = [str(row.get("responsibilities.1", "")), " ".join(skills_required)]
    return " ".join(p for p in parts if p).strip()


def build_education_candidate_text(row):
    """degree_names + major_field_of_studies (notebook 12, cell 1)."""
    text = (join_list_col(row.get("degree_names")) + " " + join_list_col(row.get("major_field_of_studies"))).strip()
    return text if text else None


class ResumeFeatureBuilderV2(BaseEstimator, TransformerMixin):
    """
    Takes a DataFrame with RAW columns (same names as the original CSV:
    career_objective, skills, responsibilities.1, skills_required,
    educationaL_requirements, degree_names, major_field_of_studies) et produit
    the 3 features retained for the v2 model: semantic_similarity,
    education_similarity, skills_soft_coverage.

    The embedding model is NEVER serialized in the pickle: only its name
    (embedding_model_name) is stored, and it is reloaded from the local
    cache after each deserialization (see __getstate__).
    """

    def __init__(self, embedding_model_name=EMBEDDING_MODEL_NAME):
        self.embedding_model_name = embedding_model_name

    def _get_embedder(self):
        if getattr(self, "_embedder", None) is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_embedder"] = None
        return state

    def fit(self, X, y=None):
        return self

    def _embed_or_nan(self, texts, embedder):
        clean = texts.fillna("").astype(str).str.strip() if hasattr(texts, "fillna") else pd.Series(texts).fillna("").astype(str).str.strip()
        uniques = clean[clean != ""].unique().tolist()
        if not uniques:
            return np.full((len(clean), embedder.get_sentence_embedding_dimension()), np.nan, dtype=np.float32)
        vecs = embedder.encode(uniques, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        lookup = dict(zip(uniques, vecs))
        out = np.full((len(clean), vecs.shape[1]), np.nan, dtype=np.float32)
        for i, t in enumerate(clean):
            if t:
                out[i] = lookup[t]
        return out

    @staticmethod
    def _rowwise_cosine(a, b):
        return np.einsum("ij,ij->i", a, b)

    def _skills_soft_coverage_row(self, candidate_skills, required_skills, embedder):
        if not candidate_skills or not required_skills:
            return np.nan
        all_skills = sorted(set(candidate_skills) | set(required_skills))
        vecs = embedder.encode(all_skills, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        idx = {s: i for i, s in enumerate(all_skills)}
        C = vecs[[idx[s] for s in candidate_skills]]
        R = vecs[[idx[s] for s in required_skills]]
        return float((R @ C.T).max(axis=1).mean())

    def transform(self, X):
        embedder = self._get_embedder()
        X = X.reset_index(drop=True)

        resume_raw_text = X.apply(build_resume_raw_text, axis=1)
        job_raw_text = X.apply(build_job_raw_text, axis=1)
        resume_vecs = self._embed_or_nan(resume_raw_text, embedder)
        job_vecs = self._embed_or_nan(job_raw_text, embedder)
        semantic_similarity = self._rowwise_cosine(resume_vecs, job_vecs)

        education_candidate_text = X.apply(build_education_candidate_text, axis=1)
        education_required_text = X["educationaL_requirements"]
        edu_cand_vecs = self._embed_or_nan(education_candidate_text, embedder)
        edu_req_vecs = self._embed_or_nan(education_required_text, embedder)
        education_similarity = self._rowwise_cosine(edu_cand_vecs, edu_req_vecs)

        candidate_skills = X["skills"].apply(parse_skills_list)
        required_skills = X["skills_required"].apply(parse_required_list)
        skills_soft_coverage = [
            self._skills_soft_coverage_row(c, r, embedder)
            for c, r in zip(candidate_skills, required_skills)
        ]

        return pd.DataFrame({
            "semantic_similarity": semantic_similarity,
            "education_similarity": education_similarity,
            "skills_soft_coverage": skills_soft_coverage,
        })
        
class CalibratedRankPipeline(BaseEstimator, RegressorMixin):
    """Chains ResumeFeatureBuilderV2 -> Ridge (rank) -> isotonic calibrator (score 0-1).
    A single object: .predict(raw DataFrame) -> calibrated score."""

    def __init__(self, feature_builder, rank_estimator, calibrator):
        self.feature_builder = feature_builder
        self.rank_estimator = rank_estimator
        self.calibrator = calibrator

    def fit(self, X, y=None):
        return self  # The 3 components are already trained separately (steps 5 and 8).

    def predict(self, X):
        features = self.feature_builder.transform(X)
        raw_rank = self.rank_estimator.predict(features)
        return self.calibrator.predict(raw_rank)        