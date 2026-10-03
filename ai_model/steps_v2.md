V2 PLAN — Generalizable Matching Architecture
============================================

Objective: move from a model that memorizes 28 fixed positions to one that
learns a genuine compatibility function f(Resume, Job) → score, able to
evaluate a resume and a position never seen during training.

STEP 1  [DONE]
V1 diagnosis
        - External validation on batuhanmtl/job_resume_fit
        - Result: predicted_score is nearly constant (std=0.018),
          with correlations close to 0 against the 3 external scores
        - Identified cause: job_position_name (one-hot) + TF-IDF vocabulary
          too specific to the original dataset
        ↓
STEP 2  [DONE]
Multi-source normalization — canonical schema
        - Define the canonical schema (Resume / Job)
        - Map resume_data_for_ranking.csv → canonical
        - Map job_descriptions.csv → canonical (376 roles, stratified sample
          of 300/role, 112,800 rows)
        - Map Resume.csv → canonical (2,484 real resumes, raw text)
        - Save the 4 canonical tables (Parquet)
        ↓
STEP 3
Build matching features (from canonical data)
        - semantic_similarity (sentence-transformers, replaces TF-IDF)
        - skills_coverage (intersection / required skills)
        - skills_jaccard (intersection / union)
        - objective_responsibilities_similarity
        - education_similarity
        - candidate_experience_years (date parsing, capped at 45 years)
        - experience_gap (candidate - required)
        - has_xxx (flags, unchanged)
        - job_position_name: removed from the model, kept as metadata
        ↓
STEP 4
New train/test split
        - Group by resume AND job (double grouping)
        - Check: 0 resumes and 0 jobs shared between train/test
        ↓
STEP 5
Retrain Ridge / Random Forest / Gradient Boosting (v2)
        - Use only the new features
        - Compare the 3 models (MSE/MAE/R²)
        ↓
STEP 6
Compare v1 vs v2 — internal test
        - Same methodology as v1 step 8 (grouped cross-validation)
        - Expected: internal R² may decrease (normal if the
          model stops relying on job_position_name)
        ↓
STEP 7
External validation — the ultimate test
        - Rerun notebook 10 (batuhanmtl) with the v2 pipeline
        - Directly compare v1 vs v2 correlations
        - Qualitative sanity checks with canonical_jobs_external /
          canonical_resumes_external (manually assembled pairs)
        ↓
STEP 8
Select and tune the final v2 model
        - RandomizedSearchCV + MLflow (same methodology as v1 step 9)
        - MLflow tag "v2" + comparison with the existing v1 run
        ↓
STEP 9
New sklearn v2 pipeline
        - ResumeFeatureBuilderV2 / JobFeatureBuilderV2 (custom_transformers)
        - Integrates the canonical schema + sentence-transformers
        - End-to-end test (raw data → score)
        ↓
STEP 10
Save and version
        - DVC: full_pipeline_v2.pkl
        - model_card_v2.md (v1/v2 comparison, remaining known limitations)
        ↓
STEP 11
Update the API
        - New request schema if needed (job_position_name as
          optional metadata, no longer a determining feature)
        - Retest edge cases (unknown position, missing fields...)
        ↓
STEP 12
Final v1 vs v2 review
        - Quantitative summary: generalization, residual bias,
          known limitations, v3 ideas (labeled matching dataset,
          multi-resume ranking)
