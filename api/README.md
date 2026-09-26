# Resume Ranking API

FastAPI API exposing the resume/job matching model trained in `ai_model/`.

## Installation

```bash
uv sync
```

## Retrieve/update the model

The model (`model/full_pipeline.pkl`) is imported from `ai_model` via DVC.

```bash
dvc update model/full_pipeline.pkl.dvc
```

## Start the server

```bash
uv run python -m api
```

The API starts on `http://localhost:8000`. Interactive documentation: `http://localhost:8000/docs`.

## Endpoints

### `GET /health`
Checks that the API and model are loaded.

### `POST /predict`
Predicts a resume/job matching score (0 to 1).

**Expected format** (all fields optional, see `src/api/schemas.py` for details):

```json
{
  "career_objective": "Experienced software engineer...",
  "skills": ["Python", "SQL", "Docker"],
  "job_position_name": "Data Engineer",
  "skills_required": "Python\nSQL\nAWS",
  "experiencere_requirement": "3+ years",
  "educationaL_requirements": "Bachelor's degree in CS",
  "responsibilities.1": "Design and maintain data pipelines"
}
```

**Response:**

```json
{
  "matched_score": 0.72,
  "job_position_known": true,
  "warnings": []
}
```

- `job_position_known: false` signals that `job_position_name` is not part of the 28 positions seen during training — the model then relies solely on textual content for this signal.
- `warnings` returns non-blocking problems (insufficient data, unknown position...).
- An unrecognized field in the request (wrongly named, for example) returns an explicit 422 error rather than being silently ignored.

## Known model limitation

See `../ai_model/docs/model_card.md` — the model relies partly on job-specific words correlated with a bias in the training dataset (28 fixed positions), rather than solely on semantic matching. To be re-evaluated with an enriched dataset (v2).

## Frozen versions (must remain synchronized with `ai_model`)

- Python 3.13
- scikit-learn 1.9.0
- pandas 3.0.6
- numpy 2.5.3

⚠️ These versions must correspond exactly to those used in `ai_model` to deserialize `full_pipeline.pkl` without error.
