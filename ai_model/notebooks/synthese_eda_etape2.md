# Synthèse de l'EDA — Resume Ranking Dataset

> Document de référence humain : ce fichier résume et **interprète** tous les résultats obtenus dans `01_dataset_exploration.ipynb` (étapes 1 et 2), avec les décisions à prendre pour l'étape 3 (nettoyage des textes) et leur justification.

---

## 1. Vue d'ensemble du dataset

- **9 544 lignes**, **35 colonnes**
- **0 doublon exact** (aucune ligne strictement identique à une autre)
- La cible à prédire est `matched_score`, un score continu entre **0 et 0.97** (moyenne 0.66, médiane 0.68)

**Ce que ça veut dire concrètement :** le dataset est propre au sens strict (pas de lignes dupliquées, pas de valeurs aberrantes du type score négatif ou > 1). Mais "pas de doublons exacts" ne veut pas dire "pas de répétition" — voir point 2.

---

## 2. Découverte clé : ce n'est pas 9 544 CV différents

En comptant les valeurs uniques de la colonne `skills`, on trouve seulement **340 CV distincts**, alors que le job posting (`job_position_name`) compte **28 postes distincts**.

En croisant plusieurs colonnes de profil (`career_objective`, `skills`, `educational_institution_name`, `degree_names`, `professional_company_names`, `positions`), on retrouve **344 combinaisons uniques** — un chiffre très proche de 340. Ça confirme que la colonne `skills` suffit quasiment à elle seule à identifier un CV.

Et si on regarde la fréquence des postes (`job_position_name`), chaque poste apparaît environ **341 à 342 fois**.

**Interprétation :** 340 CV × 28 postes ≈ 9 520, ce qui est très proche des 9 544 lignes réelles. **Ce dataset n'est donc pas une collection de candidatures réelles, mais un produit cartésien (quasi complet) : chaque CV a été évalué contre (presque) chaque offre d'emploi.** Ce n'est pas un défaut du dataset, mais il faut absolument le garder en tête :

- **Un même CV apparaît ~28 fois**, avec un score différent à chaque fois selon le poste visé.
- Le score dépend donc de la **paire (CV, poste)**, pas du CV seul.

### Pourquoi c'est important pour la suite

Si on fait un split train/test aléatoire classique (`train_test_split` ligne par ligne), **le même CV se retrouvera à la fois dans le train et dans le test** (juste évalué contre un poste différent). Le modèle peut alors "reconnaître" un CV qu'il a déjà vu à l'entraînement → score de validation trop optimiste, qui ne reflète pas la vraie performance sur un CV totalement inconnu.

**Décision pour la suite :** utiliser un split **groupé par CV** (`GroupShuffleSplit` ou `GroupKFold` de scikit-learn, avec `groups = df["skills"]` ou un identifiant de CV dédié), pour garantir qu'un CV donné n'apparaît jamais à la fois en train et en test.

---

## 3. Distribution du score (`matched_score`)

![histogramme](attachment) *(voir cellule 18 du notebook)*

L'histogramme n'est **pas une courbe lisse** : on voit des pics nets à certaines valeurs précises (autour de 0.33, 0.65, 0.85...), avec seulement **229 valeurs distinctes** sur 9 544 lignes, et certaines valeurs qui reviennent très souvent (`0.85` apparaît 1 470 fois, `0.65` apparaît 1 321 fois).

**Interprétation :** le score n'a probablement pas été attribué "à la main" ou de façon purement continue, mais généré par une **formule/algorithme** combinant un petit nombre de critères discrets (ex : nombre de compétences en commun, correspondance du niveau d'expérience, etc.), ce qui produit naturellement des valeurs récurrentes.

**Décision pour la suite :** ce n'est pas un problème en soi pour un modèle de régression, mais ça veut dire qu'on pourrait potentiellement reconstruire une partie de la logique du score via du feature engineering (voir point 5). Ça vaut aussi le coup de garder cette info en tête pour l'évaluation du modèle : ne pas s'attendre à un signal "bruité aléatoirement" mais plutôt à une fonction assez structurée.

---

## 4. Score moyen par poste

Le score médian varie fortement selon le poste visé : de **~0.45** pour des postes comme *Site Engineer*, *Mechanical Designer*, *Civil Engineer*, jusqu'à **~0.80** pour des postes techniques comme *Machine Learning Engineer*, *Database Administrator*, *AI Engineer*, *Full Stack Developer*.

**Interprétation :** soit les CV du dataset sont globalement plus orientés "tech/data" et matchent donc mieux avec les postes tech, soit l'algorithme de scoring est structurellement plus généreux sur ces postes. Dans les deux cas, **`job_position_name` est une variable très informative** pour prédire le score.

**Décision pour la suite :** encoder `job_position_name` comme feature catégorielle (one-hot ou target encoding) — c'est un candidat fort, pas juste une variable de contexte.

---

## 5. Longueur du texte vs score : pas de corrélation visible

En traçant `matched_score` en fonction du nombre de compétences listées (`skills_count`) et de la longueur du texte `career_objective`, **aucune tendance claire ne se dégage** — le nuage de points est plat, avec juste les mêmes bandes horizontales dues aux valeurs de score discrètes (point 3).

**Interprétation :** avoir un CV plus "rempli" (plus de mots, plus de compétences listées) n'augmente pas mécaniquement le score. C'est plutôt rassurant : ça veut dire que le score reflète probablement une vraie logique de **pertinence** (correspondance au poste) plutôt qu'un simple effet de volume de texte.

**Décision pour la suite :** ne pas se fier à des features de type "longueur brute du texte" comme prédicteurs principaux. Elles pourront être gardées comme features secondaires, mais l'effort doit porter sur des features de **matching** (point 6).

---

## 6. Chevauchement de compétences (`skills_overlap`) : le signal le plus clair trouvé jusqu'ici

En calculant le nombre de compétences en commun (texte brut, en minuscules) entre `skills` (CV) et `skills_required` (poste), on observe une **vraie tendance croissante** :

| Compétences en commun | Score médian approximatif |
|---|---|
| 0 | ~0.68 |
| 1 | ~0.75 |
| 2 | ~0.78 |
| 3 | ~0.89 |

**Interprétation :** c'est la variable la plus clairement corrélée au score trouvée pendant l'EDA. Mais il faut noter que la **grande majorité des lignes ont un overlap de 0**, alors qu'intuitivement beaucoup de CV devraient partager au moins une compétence avec le poste visé. C'est probablement parce que la comparaison est faite sur des chaînes de texte brutes (ex : `"Python"` vs `"python programming"` ne matchent pas alors que c'est la même compétence).

**Décision pour la suite (justifie directement l'étape 3) :** le nettoyage de texte doit inclure une **normalisation forte** des compétences (minuscules, suppression de ponctuation, gestion des synonymes/abréviations courantes) avant de recalculer ce chevauchement — sinon on sous-estime largement le vrai matching. Cette feature `skills_overlap` (une fois le texte bien nettoyé) sera probablement une des features les plus importantes du modèle final.

---

## 7. Valeurs manquantes : structurées, pas aléatoires

| Colonne | % manquant |
|---|---|
| `languages`, `proficiency_levels` | 92.7% |
| `address` | 91.8% |
| `issue_dates`, `certification_skills`, `certification_providers`, `expiry_dates`, `online_links` | 79.0% |
| `extra_curricular_*` (4 colonnes) | 64.1% |
| `career_objective` | 50.3% |
| `age_requirement` | 42.8% |
| `skills_required` | 17.8% |
| `experiencere_requirement` | 14.3% |
| Colonnes d'éducation (`degree_names`, `passing_years`, etc.) | 0.9% |
| `skills` | 0.6% |
| `job_position_name`, `responsibilities`, `responsibilities.1`, `educationaL_requirements`, `matched_score` | 0% |

La heatmap des valeurs manquantes (cellule 22) montre que ces colonnes manquent **par blocs cohérents** : quand `certification_providers` est manquant, `certification_skills`, `online_links`, `issue_dates` et `expiry_dates` le sont aussi (même chose pour le bloc "activités extra-scolaires" et le bloc "langues").

**Interprétation :** ce ne sont pas des erreurs de collecte de données. Ce sont des **sections optionnelles du CV que le candidat n'a simplement pas remplies** (tout le monde n'a pas de certifications, tout le monde n'a pas listé de langues parlées). C'est ce qu'on appelle du **MNAR structurel** (Missing Not At Random, mais de façon prévisible et logique).

**Décision pour la suite :** ne pas essayer d'imputer statistiquement ces valeurs (moyenne, mode, etc.) — ça n'aurait aucun sens de "deviner" une certification manquante. À la place :
- Remplacer les textes manquants par une chaîne vide `""` (pas de perte d'info, le modèle NLP traitera ça naturellement comme "rien à dire").
- Envisager d'ajouter un indicateur binaire du type `has_certification`, `has_languages` — l'*absence* de section peut elle-même être informative.
- Les colonnes à >90% de vide (`address`, `languages`, `proficiency_levels`) sont probablement à **exclure** du modèle : trop creuses pour apporter un signal fiable.

---

## 8. Expérience requise vs score : signal faible

En regroupant le score par nombre d'années d'expérience minimum requis (`experiencere_requirement` parsé), les médianes restent proches les unes des autres (0.58 à 0.75 selon le palier), sans tendance monotone nette. Il y a un peu plus de variance/de valeurs basses pour les paliers élevés (5 et 15 ans), mais rien d'aussi net que le chevauchement de compétences (point 6).

**Interprétation :** l'expérience requise seule n'explique pas grand-chose du score. Elle joue peut-être un rôle en interaction avec d'autres critères, mais pas comme variable isolée.

**Décision pour la suite :** garder cette feature (elle ne coûte rien à extraire) mais ne pas en attendre beaucoup — elle ne sera probablement pas dans le top des features importantes.

---

## 9. Synthèse — Ce qu'on retient pour l'étape 3

| # | Constat de l'EDA | Décision pour l'étape 3 / la suite | Pourquoi |
|---|---|---|---|
| 1 | 340 CV réutilisés ~28 fois chacun | Split train/test **par CV** (`GroupShuffleSplit`) | Éviter la fuite de données (le même CV en train et en test) |
| 2 | Score discret, généré par formule | Pas d'action de nettoyage, mais à garder en tête pour l'évaluation du modèle | Le signal est structuré, pas bruité aléatoirement |
| 3 | `job_position_name` très corrélé au score | Encoder comme feature catégorielle forte | Variable clé identifiée dès l'EDA |
| 4 | Pas de lien longueur de texte / score | Ne pas se reposer sur des features de longueur brute | Le volume de texte ne reflète pas la pertinence |
| 5 | `skills_overlap` corrélé au score, mais sous-estimé par matching de texte brut | **Priorité n°1 du nettoyage** : normaliser les compétences (minuscule, ponctuation, synonymes) pour un meilleur matching | C'est le signal le plus fort trouvé, mais il est bridé par la qualité du texte actuel |
| 6 | Valeurs manquantes structurées par blocs (sections de CV non remplies) | Remplacer par chaîne vide + ajouter des indicateurs `has_xxx` ; exclure les colonnes >90% vides (`address`, `languages`, `proficiency_levels`) | Ce sont des absences logiques, pas des erreurs — l'absence peut être informative |
| 7 | Expérience requise : signal faible | Garder la feature mais priorité basse | Peu de pouvoir explicatif isolé |

**Conclusion :** l'étape 3 (nettoyage des textes) doit se concentrer en priorité sur :
1. La normalisation des colonnes de compétences (`skills`, `skills_required`) pour fiabiliser le calcul de `skills_overlap`.
2. Le nettoyage standard des textes libres (`career_objective`, `responsibilities`, `responsibilities.1`) : minuscules, suppression de caractères parasites, gestion des `\n`.
3. La gestion explicite des valeurs manquantes (chaîne vide + indicateurs binaires), sans imputation statistique.
4. La mise en place dès maintenant de la logique de split par CV, pour que toutes les étapes suivantes (features, modèles) soient construites sur une base saine.
