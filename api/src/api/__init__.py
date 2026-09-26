import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict

from custom_transformers import ResumeFeatureBuilder, TextSimilarityVectorizer  # noqa: F401
from api.schemas import MatchRequest, MatchResponse


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    status: str
    model_loaded: bool


PIPELINE_PATH = Path(__file__).resolve().parents[2] / "model" / "full_pipeline.pkl"


def create_app() -> FastAPI:
    application = FastAPI(
        title="Resume Ranking API",
        description="Prédit un score de matching CV/poste à partir d'un CV et d'une offre au format structuré.",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )

    state = {"pipeline": None, "known_job_positions": set()}

    @application.on_event("startup")
    def load_pipeline():
        state["pipeline"] = joblib.load(PIPELINE_PATH)
        ohe = state["pipeline"].named_steps["preprocessor"].named_transformers_["job_position"]
        state["known_job_positions"].update(ohe.categories_[0])

    @application.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=state["pipeline"] is not None)

    @application.post("/predict", response_model=MatchResponse)
    def predict(payload: MatchRequest) -> MatchResponse:
        if state["pipeline"] is None:
            raise HTTPException(status_code=503, detail="Le modèle n'est pas encore chargé.")

        warnings = []

        if not payload.career_objective and not payload.skills:
            warnings.append("Aucune information CV fournie — le score risque d'être peu fiable.")
        if not payload.skills_required and not payload.responsibilities_1:
            warnings.append("Aucune information poste fournie — le score risque d'être peu fiable.")

        job_known = (
            payload.job_position_name in state["known_job_positions"]
            if payload.job_position_name else False
        )
        if payload.job_position_name and not job_known:
            warnings.append(
                f"Le poste '{payload.job_position_name}' n'était pas dans les données d'entraînement — "
                "le modèle se base uniquement sur le contenu textuel pour ce champ."
            )

        skills_str = str(payload.skills) if payload.skills else np.nan
        row = {
            "career_objective": payload.career_objective if payload.career_objective else np.nan,
            "skills": skills_str,
            "certification_providers": payload.certification_providers if payload.certification_providers else np.nan,
            "languages": payload.languages if payload.languages else np.nan,
            "extra_curricular_activity_types": payload.extra_curricular_activity_types if payload.extra_curricular_activity_types else np.nan,
"job_position_name": payload.job_position_name if payload.job_position_name else "unknown_position",            "skills_required": payload.skills_required if payload.skills_required else np.nan,
            "educationaL_requirements": payload.educationaL_requirements if payload.educationaL_requirements else np.nan,
            "responsibilities.1": payload.responsibilities_1 if payload.responsibilities_1 else np.nan,
            "experiencere_requirement": payload.experiencere_requirement if payload.experiencere_requirement else np.nan,
            "age_requirement": payload.age_requirement if payload.age_requirement else np.nan,
        }
        df = pd.DataFrame([row])

        try:
            score = state["pipeline"].predict(df)[0]
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Erreur lors de la prédiction : {e}")

        score = float(np.clip(score, 0.0, 1.0))

        return MatchResponse(
            matched_score=round(score, 4),
            job_position_known=job_known,
            warnings=warnings,
        )

    return application


app = create_app()