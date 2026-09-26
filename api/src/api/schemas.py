from typing import Optional, List
from pydantic import BaseModel, Field, ConfigDict


class MatchRequest(BaseModel):
    """
    JSON format expected by the API. All fields are optional — a missing
    field is treated exactly as a NaN in the training dataset
    (see ResumeFeatureBuilder: has_xxx=0, empty text).
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

    # --- Resume side ---
    career_objective: Optional[str] = None
    skills: Optional[List[str]] = Field(default=None, description="List of candidate skills")
    certification_providers: Optional[str] = Field(
        default=None, description="Filled if the candidate has certifications"
    )
    languages: Optional[str] = Field(
        default=None, description="Filled if the candidate has listed languages"
    )
    extra_curricular_activity_types: Optional[str] = Field(
        default=None, description="Filled if the candidate has extra-curricular activities"
    )

    # --- Job side ---
    job_position_name: Optional[str] = Field(
        default=None, description="Job title. If unknown in the training dataset, the model continues to function with one less signal."
    )
    skills_required: Optional[str] = Field(
        default=None, description="Required skills, one per line (separated by \\n)"
    )
    educationaL_requirements: Optional[str] = None
    responsibilities_1: Optional[str] = Field(
        default=None, alias="responsibilities.1", description="Job responsibilities"
    )
    experiencere_requirement: Optional[str] = Field(
        default=None, description="Required experience, e.g., '3+ years'"
    )
    age_requirement: Optional[str] = Field(
        default=None, description="Filled if the position has an age requirement"
    )


class MatchResponse(BaseModel):
    matched_score: float = Field(..., description="Predicted matching score, between 0 and 1")
    job_position_known: bool = Field(
        ..., description="False if job_position_name was not in the 28 positions of the training dataset."
    )
    warnings: List[str] = Field(default_factory=list, description="Non-blocking warnings about the quality of provided data")