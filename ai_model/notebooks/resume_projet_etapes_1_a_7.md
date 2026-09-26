# Résumé du projet — Resume Ranking

> Point d'étape après les étapes 1 à 7 (compréhension du dataset → premiers modèles), avant l'étape 8 (cross-validation). Document de référence humain, pour se souvenir de ce qui a été fait et pourquoi.

---

## Objectif du projet

Prédire `matched_score` (un score entre 0 et 1) qui mesure à quel point un CV correspond à une offre d'emploi, à partir d'un dataset de 9 544 lignes (Resume Data for Ranking).

Roadmap complète du projet :
`1. Comprendre le dataset → 2. EDA → 3. Nettoyage → 4. Features → 5. Ridge → 6. Random Forest → 7. Gradient Boosting → 8. Cross-validation → 9. Choix du modèle → 10. Pipeline sklearn → 11. Sauvegarde → 12. API`

---

## Étapes 1-2 — Comprendre le dataset & EDA

### Ce qu'on a découvert

- **9 544 lignes, 35 colonnes**, aucun doublon exact, aucune valeur du score hors de [0, 1].
- **Découverte structurante** : le dataset n'est pas 9 544 candidatures réelles indépendantes, mais **340 CV uniques croisés avec 28 postes** (quasi produit cartésien : 340 × 28 ≈ 9 520, très proche des 9 544 lignes réelles). Chaque CV apparaît donc ~28 fois, avec un score différent à chaque fois selon le poste visé.
- **Conséquence directe et critique** : un split train/test aléatoire classique provoquerait une fuite de données (le même CV se retrouvant en train et en test). → **Décision : split obligatoire par groupe de CV** (`GroupShuffleSplit`), appliquée depuis l'étape 4.
- Le score `matched_score` a une distribution **discrète par paliers** (229 valeurs distinctes seulement, pics nets à 0.33/0.65/0.85...) → probablement généré par une formule/algorithme plutôt qu'attribué librement.
- **`job_position_name` est fortement corrélé au score** : médiane ~0.45 pour les postes d'ingénierie physique (Civil Engineer, Mechanical Designer...) contre ~0.80 pour les postes tech (Machine Learning Engineer, DBA...). C'est un **biais structurel du dataset** qu'on a retrouvé ensuite dans tous les modèles.
- **Aucun lien entre la longueur du texte et le score** (nombre de compétences, longueur de `career_objective`) → le volume de texte n'est pas un signal utile.
- **Le chevauchement de compétences CV ↔ poste (`skills_overlap`) est corrélé positivement au score**, mais reste faible en valeur brute (~95% des lignes à overlap = 0) à cause d'un matching texte trop strict.
- **Valeurs manquantes structurées par blocs** (ex : toutes les colonnes de certification manquent ensemble) → ce sont des sections de CV non remplies, pas des erreurs de collecte (MNAR structurel), donc pas d'imputation statistique nécessaire.

---

## Étape 3 — Nettoyage des textes

Fichier : `02_data_cleaning.ipynb` → sortie `data/processed/resume_data_clean.csv` (9 544 lignes, 43 colonnes), suivi avec DVC.

### Décisions prises et justification

| Action | Pourquoi |
|---|---|
| Nettoyage agressif du texte libre (minuscules, ponctuation supprimée) pour `career_objective`, `responsibilities`, etc. | Préparer la vectorisation (TF-IDF) |
| Nettoyage **léger** des compétences (`normalize_skill`) : minuscule + espaces, sans supprimer la ponctuation technique | Éviter de casser des termes comme `C++`, `.NET` |
| Recalcul de `skills_overlap` après normalisation | Vérifier le gain du nettoyage sur le matching |
| Remplacement des NaN texte par chaîne vide (pas d'imputation statistique) | Les valeurs manquantes sont des sections de CV non renseignées, pas des erreurs |
| Ajout d'indicateurs binaires `has_xxx` (has_certifications, has_languages...) avant de perdre l'info | L'absence d'une section peut être informative en soi |
| Suppression des colonnes >90% vides (`address`, `languages`, `proficiency_levels`, etc.) | Trop creuses pour apporter un signal fiable |

---

## Étape 4 — Feature engineering

Fichier : `03_feature_engineering.ipynb` → sorties `X_train.npz`, `X_test.npz`, `y_train.npy`, `y_test.npy`, `tfidf_vectorizer.pkl`, `job_position_encoder.pkl`.

### Ce qui a été construit

1. **Split train/test par CV** (`GroupShuffleSplit`, groupé sur `skills`, avec gestion des NaN via un identifiant unique par ligne) — 0 CV en commun vérifié entre train et test.
2. **TF-IDF** (200 mots max, unigrammes + bigrammes) entraîné **uniquement sur le train**, appliqué séparément au texte du CV (`resume_text`) et au texte du poste (`job_text`), avec un vocabulaire partagé pour permettre la comparaison.
3. **`text_similarity`** : similarité cosinus entre le vecteur TF-IDF du CV et celui du poste — feature conçue pour remplacer le `skills_overlap` brut, trop strict.
4. **One-hot encoding de `job_position_name`**, avec `handle_unknown="ignore"` pour ne pas planter face à un poste jamais vu en production.
5. **Features numériques conservées** : `skills_overlap`, `experience_years_min`, tous les flags `has_xxx`.
6. **Matrice finale `X`** : ~438 colonnes (200 + 200 TF-IDF + 1 similarité + 9 numériques + ~28 catégorielles one-hot).

### Point de vigilance retenu

Encoder/vectorizer toujours **fit sur le train seul**, jamais sur tout le dataset — sinon fuite de données via le vocabulaire appris, même règle que pour le split.

---

## Étapes 5-7 — Premiers modèles

### Résultats comparés

| Métrique | Ridge (baseline) | Random Forest | Gradient Boosting |
|---|---|---|---|
| MSE | 0.0247 | **0.0171** | 0.0182 |
| MAE | 0.1227 | **0.0983** | 0.1036 |
| R² | 0.2067 | **0.4513** | 0.4155 |

**Random Forest est le meilleur des trois sur ce split**, Gradient Boosting juste derrière, Ridge nettement plus faible (mais reste une baseline utile — elle confirme qu'il y a un vrai signal dans les données, pas juste du bruit).

### Ce qu'on a appris de chaque modèle

- **Ridge** : confirme que `text_similarity` est la feature la plus utile pour prédire un score élevé. Montre un effet de "régression vers la moyenne" marqué (sur-prédit les scores bas, sous-prédit les scores hauts) — comportement typique d'un modèle linéaire régularisé.
- **Random Forest** : meilleure performance globale, effet de régression vers la moyenne moins marqué.
- **Gradient Boosting** : performance proche de Random Forest mais légèrement inférieure — probablement parce que le dataset (340 CV uniques) est trop petit/peu varié pour que GB exploite pleinement sa capacité de correction progressive des erreurs.

### Un biais qui revient dans les 3 modèles

Les mots `autocad`, `civil engineering`, `mechanical engineering` **font baisser** la prédiction dans tous les modèles, alors que `text_similarity` reste toujours dans le top des features importantes. Interprétation : le modèle apprend en partie le **vrai signal de matching** (`text_similarity`), mais aussi un **raccourci lié au biais structurel poste ↔ score** identifié dès l'EDA (les postes d'ingénierie physique ont des scores plus bas dans ce dataset, indépendamment de la qualité réelle du matching).

**Ce n'est pas une erreur de code** — c'est une limite du dataset (structure fermée à 28 postes, avec un biais de scoring entre catégories) à garder en tête pour le choix final du modèle (étape 9).

---

## Décisions ouvertes / pistes pour plus tard (pas maintenant)

- **Plus de données** (plus de CV variés, plus de secteurs) pourrait réduire le biais poste ↔ score et améliorer la généralisation — à envisager seulement une fois le pipeline complet terminé (étape 12), pas avant.
- **Embeddings pré-entraînés** (Hugging Face `sentence-transformers`, sans finetuning) comme alternative/complément au TF-IDF, à tester une fois qu'un modèle de référence tourne bien.
- **Finetuning d'un modèle de langage (BERT...)** : écarté pour l'instant, trop risqué (overfitting) avec seulement 340 CV uniques. À reconsidérer avec un dataset plus riche.

---

## Prochaine étape

**Étape 8 — Cross-validation** : évaluer Ridge/RF/GB sur plusieurs splits (au lieu d'un seul), pour savoir si l'écart RF > GB observé est un vrai écart robuste ou le fruit du hasard sur ce split précis — et préparer la décision finale du modèle à l'étape 9.
