# Model Card — Resume Ranking

## Summary

- **Model**: Random Forest Regressor (scikit-learn)
- **Task**: Predict `matched_score` (0-1) from a resume and a job posting
- **Version**: v1
- **Date**: 2026-09-26
- **MLflow Run**: `random_forest_v1_tuned` (run_id `36843610cea64272a7e4d5c4e488b569`)

## Performance (test set, split by resume)

| Metric | Value |
|---|---|
| R² | 0.486 |
| MAE | 0.096 |
| MSE | 0.016 |

## Hyperparameters
n_estimators: 300
min_samples_leaf: 3
max_features: sqrt
max_depth: None


## Training Data

- `data/raw/resume_data_for_ranking.csv` (9,544 rows, 340 unique resumes × 28 positions)
- Train/test split by resume (`GroupShuffleSplit`, test_size=0.2, random_state=42) to avoid data leakage

## Features Used

- TF-IDF (200 words, unigrams+bigrams) on resume text and job text, shared vocabulary
- Cosine similarity between the two texts (`text_similarity`)
- One-hot encoding of `job_position_name` (`handle_unknown="ignore"`)
- Numeric features: `skills_overlap`, `experience_years_min`, flags `has_xxx` (presence of resume sections)

## Known Limitation

The model relies partly on job-specific words (e.g., `autocad`, `civil engineering`) correlated with a structural bias in the dataset (physical engineering positions have lower scores in this dataset), rather than solely on semantic resume/job matching. To be reviewed when the dataset is enriched with more varied resumes and positions (v2).

## How to Load the Model

```python
import sys
sys.path.append("src")  # necessary to deserialize custom classes
import joblib

pipeline = joblib.load("data/processed/full_pipeline.pkl")
prediction = pipeline.predict(new_raw_dataframe)
```

⚠️ The file `src/custom_transformers.py` must be present and importable to load this pipeline (it contains `ResumeFeatureBuilder` and `TextSimilarityVectorizer`).

## Key Dependencies

- scikit-learn 1.9.0
- pandas, numpy, scipy
