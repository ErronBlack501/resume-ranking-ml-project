from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class MatchRequest(BaseModel):
    """
    Format JSON attendu par l'API. Tous les champs sont optionnels — un champ
    absent est traité exactement comme un NaN dans le dataset d'entraînement
    (voir ResumeFeatureBuilder : has_xxx=0, texte vide).
    """
    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
        json_schema_extra={
            "example": {
                "career_objective": "Experienced software engineer specialized in backend systems.",
                "skills": ["Python", "SQL", "Docker", "FastAPI"],
                "job_position_name": "Data Engineer",
                "skills_required": "Python\nSQL\nAWS\nAirflow",
                "experiencere_requirement": "3+ years",
                "educationaL_requirements": "Bachelor's degree in Computer Science",
                "responsibilities.1": "Design and maintain data pipelines",
            }
        },
    )

    # --- Côté CV ---
    career_objective: Optional[str] = None
    skills: Optional[List[str]] = Field(default=None, description="Liste des compétences du candidat")
    certification_providers: Optional[str] = Field(
        default=None, description="Renseigné si le candidat a des certifications"
    )
    languages: Optional[str] = Field(
        default=None, description="Renseigné si le candidat a listé des langues"
    )
    extra_curricular_activity_types: Optional[str] = Field(
        default=None, description="Renseigné si le candidat a des activités extra-scolaires"
    )

    # --- Côté poste ---
    job_position_name: Optional[str] = Field(
        default=None, description="Intitulé du poste. Si inconnu du dataset d'entraînement, le modèle continue de fonctionner avec un signal en moins."
    )
    skills_required: Optional[str] = Field(
        default=None, description="Compétences requises, une par ligne (séparées par \\n)"
    )
    educationaL_requirements: Optional[str] = None
    responsibilities_1: Optional[str] = Field(
        default=None, alias="responsibilities.1", description="Responsabilités du poste"
    )
    experiencere_requirement: Optional[str] = Field(
        default=None, description="Expérience requise, ex: '3+ years'"
    )
    age_requirement: Optional[str] = Field(
        default=None, description="Renseigné si le poste a une exigence d'âge"
    )


class MatchResponse(BaseModel):
    matched_score: float = Field(..., description="Score de matching prédit, entre 0 et 1")
    job_position_known: bool = Field(
        ..., description="False si job_position_name n'était pas dans les 28 postes du dataset d'entraînement."
    )
    warnings: List[str] = Field(default_factory=list, description="Avertissements non bloquants sur la qualité des données fournies")