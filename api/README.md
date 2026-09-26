# Resume Ranking API

API FastAPI exposant le modèle de matching CV/poste entraîné dans `ai_model/`.

## Installation

```bash
uv sync
```

## Récupérer/mettre à jour le modèle

Le modèle (`model/full_pipeline.pkl`) est importé depuis `ai_model` via DVC.

```bash
dvc update model/full_pipeline.pkl.dvc
```

## Lancer le serveur

```bash
uv run python -m api
```

L'API démarre sur `http://localhost:8000`. Documentation interactive : `http://localhost:8000/docs`.

## Endpoints

### `GET /health`
Vérifie que l'API et le modèle sont chargés.

### `POST /predict`
Prédit un score de matching CV/poste (0 à 1).

**Format attendu** (tous les champs optionnels, voir `src/api/schemas.py` pour le détail) :

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

**Réponse :**

```json
{
  "matched_score": 0.72,
  "job_position_known": true,
  "warnings": []
}
```

- `job_position_known: false` signale que `job_position_name` ne fait pas partie des 28 postes vus à l'entraînement — le modèle se base alors uniquement sur le contenu textuel pour ce signal.
- `warnings` remonte les problèmes non bloquants (données insuffisantes, poste inconnu...).
- Un champ non reconnu dans la requête (mal nommé, par exemple) renvoie une erreur 422 explicite plutôt que d'être ignoré silencieusement.

## Limite connue du modèle

Voir `../ai_model/docs/model_card.md` — le modèle s'appuie en partie sur des mots-métiers corrélés à un biais du dataset d'entraînement (28 postes fixes), plutôt qu'uniquement sur le matching sémantique. À réévaluer avec un dataset enrichi (v2).

## Versions figées (doivent rester synchronisées avec `ai_model`)

- Python 3.13
- scikit-learn 1.9.0
- pandas 3.0.6
- numpy 2.5.3

⚠️ Ces versions doivent correspondre exactement à celles utilisées dans `ai_model` pour désérialiser `full_pipeline.pkl` sans erreur.