import numpy as np
import pandas as pd

# V2 model inputs (see plan, step 3)
MODEL_FEATURES = [
    "semantic_similarity", "objective_responsibilities_similarity", "education_similarity",
    "skills_soft_coverage", "experience_gap", "candidate_experience_years",
]
ABLATION_CANDIDATES = ["skills_coverage", "skills_jaccard"]


def assign_block_folds(resume_ids, job_ids, n_resume_folds=5, n_job_folds=5,
                       seed=42, job_strata=None):
    """Assign a resume fold and a job fold to each row.
    job_strata: dict {job_id: stratum}; jobs in the same stratum are assigned
    to folds in round-robin order."""
    rng = np.random.default_rng(seed)

    def assign(ids, n_folds, strata=None):
        ids = rng.permutation(np.unique(ids))
        if strata is not None:
            ids = ids[np.argsort([strata[i] for i in ids], kind="stable")]
        return {i: k % n_folds for k, i in enumerate(ids)}

    resume_map = assign(resume_ids, n_resume_folds)
    job_map = assign(job_ids, n_job_folds, job_strata)
    return (np.array([resume_map[r] for r in resume_ids]),
            np.array([job_map[j] for j in job_ids]))


def iter_block_splits(resume_fold, job_fold):
    """Yield one split per cell (resume fold, job fold).
    Test = resumes in the fold AND jobs in the fold; train = neither that
    resume fold nor that job fold."""
    resume_fold, job_fold = np.asarray(resume_fold), np.asarray(job_fold)
    for rf in np.unique(resume_fold):
        for jf in np.unique(job_fold):
            train_idx = np.where((resume_fold != rf) & (job_fold != jf))[0]
            test_idx = np.where((resume_fold == rf) & (job_fold == jf))[0]
            yield rf, jf, train_idx, test_idx


def within_job_spearman(y_true, y_pred, job_ids, min_rows=20):
    """Calculate Spearman per job, then average: measures the ability to rank
    resumes for the same job posting."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred), "j": np.asarray(job_ids)})
    rs = []
    for _, g in df.groupby("j"):
        if len(g) >= min_rows and g["y"].nunique() > 1 and g["p"].nunique() > 1:
            rs.append(g["y"].corr(g["p"], method="spearman"))
    return float(np.mean(rs)) if rs else np.nan