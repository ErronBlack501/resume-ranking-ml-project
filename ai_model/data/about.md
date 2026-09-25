About Dataset
📄 Resume Data for Ranking
This dataset provides structured resume information paired with job descriptions to support the development of resume ranking algorithms, candidate-job matching models, and AI-driven recruitment systems.

📊 Dataset Overview

Total Records: 9,544
Total Features: 35
Format: CSV
Primary Goal: Evaluate how well a resume matches a given job posting using the matched_score field.

🧠 Resume Features

career_objective
skills
educational_institution_name, degree_names, passing_years, major_field_of_studies, educational_results
professional_company_names, positions, responsibilities
certification_providers, certification_skills, issue_dates, expiry_dates
languages, proficiency_levels
extra_curricular_activity_types, extra_curricular_organization_names, role_positions
📋 Job Posting Features

job_position_name
educationaL_requirements
experiencere_requirement
age_requirement
skills_required
responsibilities.1 (job-specific responsibilities)
🎯 Target Variable
matched_score
A float score between 0 and 1 indicating how closely a candidate’s resume matches the job posting.

✅ Potential Use Cases

Resume Ranking Systems
Intelligent Candidate Screening
Job Recommendation Engines
NLP-based Resume Parsing
Feature Engineering for HR Analytics
⚠️ Notes

Some columns contain missing or incomplete values and may require preprocessing.
Features are mostly stored in stringified list formats or free-text, ideal for NLP applications.