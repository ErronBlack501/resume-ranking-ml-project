# Model Card — Resume Ranking v2

## Summary

- **Model**: Ridge Regression (scikit-learn), trained on within-job relative ranks, followed by isotonic calibration to the 0–1 scale
- **Task**: predict a resume/job matching score, usable to rank multiple resumes for the same job posting
- **Version**: v2
- **Date**: 2026-10-02
- **MLflow run**: `ridge_v2_rank_tuned`

## Fundamental difference from v1

V1 received `job_position_name` as a one-hot feature. The model could associate
a score with a job title rather than its actual content — confirmed by external
validation, where v1 collapsed to an almost constant prediction (standard
deviation 0.018) when given positions outside its 28 training categories.

V2 has no access to job identity. All its features measure a relationship
between the resume text and the job posting, computed with sentence embeddings
(`sentence-transformers/all-MiniLM-L6-v2`) rather than TF-IDF. This makes it
possible to capture synonyms ("postgres" / "postgresql") that exact word
counting misses.

## Model features

Only 3 features were retained after ablation from 6 initial candidates (see
Known limitations):

| Feature | Calculation |
|---|---|
| `semantic_similarity` | Cosine similarity between the resume text embedding and the job posting embedding |
| `education_similarity` | Cosine similarity between the candidate's education embedding and the job requirements |
| `skills_soft_coverage` | For each required skill, similarity to the candidate's closest skill, averaged |

Tested and excluded features: `skills_coverage`/`skills_jaccard` (exact string
matching, almost always 0), `candidate_experience_years` and `experience_gap`
(correlated at 0.95, no signal after removing redundancy),
`objective_responsibilities_similarity` (correlated at 0.72 with
`semantic_similarity`, an inflated feature), and the candidate-side `has_*`
flags (only one, `has_extra_curricular`, showed a consistent effect, but it
was unrelated to job fit — excluded by design, not for lack of signal).

`job_position_name` is retained as metadata (logs, API warnings) but never
contributes to the score.

## Training target — an important detail

The model is not trained to predict `matched_score` directly, but the
**relative rank** of a resume among resumes evaluated for the same job in the
training dataset. This choice follows an explicit test: trained on the raw
score, Ridge performed worse than a simple average of the 3 features
(Spearman 0.346 vs 0.370); trained on rank, the difference from that average
became non-significant (0.366, p=0.12 in a paired Wilcoxon test). The rank
output is then recalibrated to the 0–1 `matched_score` scale using isotonic
regression, preserving prediction order (original/calibrated Spearman =
0.9966).

## Performance

**Internal** (blocked cross-validation, 5 resume folds × 5 job folds = 25
splits; resumes and jobs are never simultaneously present in both train and
test):

| Metric | Value |
|---|---|
| Within-job Spearman (mean across 25 splits) | 0.366 ± 0.093 |
| Baseline: simple average of the 3 features | 0.370 ± 0.093 |
| Baseline: variance explained by job alone | 28.4% |

None of the tested models (Ridge, Random Forest, Gradient Boosting)
significantly outperforms a simple combination of the 3 best features on this
dataset. With so few informative signals and about 6,000 training rows per
split, there was no hidden complexity to exploit beyond a well-chosen average.
Ridge on rank is retained despite this tie because it does not require
hand-picking weights and can continue learning if the dataset is expanded.

**External** (`batuhanmtl/job_resume_fit`, 2,385 unseen resume/job pairs, 3
independent scoring methods):

| | v1 | v2 |
|---|---|---|
| Prediction standard deviation | 0.018 | 0.088 |
| Correlation with `ai_match_score` (LLM judgment) | ~0.02–0.10 | 0.706 (overall) / 0.550 (within category) |
| Correlation with `skill_string_match_score` | ~0.02–0.10 | 0.569 / 0.447 |
| Correlation with `fuzzy_match_score` | ~0.02–0.10 | 0.682 / 0.532 |

This is the main evidence that v2 generalizes: it remains discriminative and
relevant for resumes and jobs entirely outside its training data, where v1
collapsed.

## Known limitations

**Disagreement with the training dataset itself.** On a sample of 500 rows from
the original dataset, the mean absolute difference between the calibrated
score and `matched_score` is 0.121, with 6.2% of rows differing by more than
0.3. One case examined in detail (a resume with game theory and cryptography
skills, scored 0.917 for a Senior iOS Engineer position) shows a disagreement
consistent with the features (`semantic_similarity` = 0.063,
`skills_soft_coverage` = 0.143 — a very poor match according to the text):
the model evaluates the content correctly, while the original score seems
unreliable in this case. As noted in the EDA, the exact formula behind
`matched_score` has never been fully identified.

**Compressed calibrated scores.** The calibrated score's standard deviation
(0.051) is significantly lower than that of the original `matched_score`
(~0.17). The rank model captures the order of candidates for a given job, but
not the absolute magnitude of the differences between them. As a direct
consequence, fixed thresholds (such as "Strong if score ≥ 0.7") would not make
sense for this distribution — the user-facing display (step 11) should use
percentiles relative to the training distribution, not absolute cutoffs.

**`education_similarity` depends on education text being present.** It is
missing in some external sources (e.g., `batuhanmtl`, which provides no
education text). This is handled natively by the imputer, but reduces the
model to 2 effective features in that case.

**Only one embedding model was tested** (`all-MiniLM-L6-v2`, lightweight and
fast on CPU). A larger model might improve `semantic_similarity`; this was
not tested due to project time/compute budget.

**Generalization verified on only one external dataset.** `batuhanmtl` covers
23 broad categories; performance on very specific job titles or
unrepresented sectors (healthcare, law, etc.) remains to be verified.

## Loading the model

```python
import sys
sys.path.append("src")  # required to deserialize CalibratedRankPipeline and ResumeFeatureBuilderV2
import joblib

pipeline = joblib.load("data/processed/full_pipeline_v2.pkl")
score = pipeline.predict(raw_dataframe)  # array of floats between 0 and 1
```

⚠️ `src/custom_transformers_v2.py` must be present and importable. The first
call to `.predict()` downloads/loads `sentence-transformers/all-MiniLM-L6-v2`
(about 90 MB, cached locally afterward).

## Key dependencies

- scikit-learn 1.9.0
- sentence-transformers (see `pyproject.toml` for the exact pinned version)
- pandas, numpy, scipy
