# Fiche modèle — Resume Ranking

## Résumé

- **Modèle** : Random Forest Regressor (scikit-learn)
- **Tâche** : Prédire `matched_score` (0-1) à partir d'un CV et d'une offre d'emploi
- **Version** : v1
- **Date** : 2026-09-26
- **Run MLflow** : `random_forest_v1_tuned` (run_id `36843610cea64272a7e4d5c4e488b569`)

## Performance (test set, split par CV)

| Métrique | Valeur |
|---|---|
| R² | 0.486 |
| MAE | 0.096 |
| MSE | 0.016 |

## Hyperparamètres
n_estimators: 300
min_samples_leaf: 3
max_features: sqrt
max_depth: None


## Données d'entraînement

- `data/raw/resume_data_for_ranking.csv` (9 544 lignes, 340 CV uniques × 28 postes)
- Split train/test par CV (`GroupShuffleSplit`, test_size=0.2, random_state=42) pour éviter la fuite de données

## Features utilisées

- TF-IDF (200 mots, unigrammes+bigrammes) sur le texte CV et le texte poste, vocabulaire partagé
- Similarité cosinus entre les deux textes (`text_similarity`)
- One-hot encoding de `job_position_name` (`handle_unknown="ignore"`)
- Features numériques : `skills_overlap`, `experience_years_min`, flags `has_xxx` (présence des sections du CV)

## Limite connue

Le modèle s'appuie en partie sur des mots-métiers (ex : `autocad`, `civil engineering`) corrélés à un biais structurel du dataset (les postes d'ingénierie physique ont des scores plus bas dans ce dataset), plutôt que uniquement sur le matching sémantique CV/poste. À revoir quand le dataset sera enrichi en CV et postes plus variés (v2).

## Comment charger le modèle

```python
import sys
sys.path.append("src")  # nécessaire pour désérialiser les classes custom
import joblib

pipeline = joblib.load("data/processed/full_pipeline.pkl")
prediction = pipeline.predict(nouveau_dataframe_brut)
```

⚠️ Le fichier `src/custom_transformers.py` doit être présent et importable pour charger ce pipeline (il contient `ResumeFeatureBuilder` et `TextSimilarityVectorizer`).

## Dépendances clés

- scikit-learn 1.9.0
- pandas, numpy, scipy