# Prédiction de maintenance des moteurs d’avion (NASA C-MAPSS)

Projet de **maintenance prédictive** sur des moteurs d’avion, à partir du jeu de données simulé **NASA C-MAPSS**.  
Objectif : identifier les moteurs **à risque à court terme** pour planifier la maintenance avant la casse.

> Format : projet concentré dans **un notebook unique**, pensé pour un recruteur / lecteur technique.

---

## 1. Objectif du projet

Dans la littérature, le dataset C-MAPSS est souvent utilisé pour de la **régression de RUL**  
(*Remaining Useful Life* : nombre de cycles restants avant la panne).

Ici, je reformule le problème en **classification binaire** plus proche d’un cas métier :

> “Faut-il planifier une opération de maintenance dans les **N prochains cycles** ?”

Pour cela :

- je calcule une RUL pour chaque cycle,
- je définis une fenêtre métier `MAINTENANCE_WINDOW = 30` cycles,
- je crée une cible binaire :
  - `1` → moteur **à risque** (RUL ≤ 30 cycles),
  - `0` → moteur **non à risque**.

---

## 2. Jeu de données

- Source : données simulées NASA **C-MAPSS**, version pré-traitée.
- Fichiers utilisés (placés dans `Docs/`) :

  - `PM_train.csv` – données d’apprentissage
  - `PM_test.csv` – données de test
  - `PM_truth.csv` – RUL réelle associée au test

- Colonnes (standard C-MAPSS) :
  - `id` : identifiant du moteur
  - `cycle` : numéro de cycle
  - `setting_*` : conditions de fonctionnement
  - `sensor_*` : mesures capteurs (température, pression, etc.)
  - colonnes dérivées : RUL, features de séries temporelles, ratios, etc.

---

## 3. Pipeline de modélisation

Tout le pipeline est dans le notebook :

`aircraft_engine_predictive_maintenance.ipynb`

### 3.1 Préparation & EDA

- Chargement de `PM_train.csv`, `PM_test.csv`, `PM_truth.csv`.
- Vérifications de base :
  - dimensions, types, échantillons de lignes,
  - répartition de la cible `label_maintenance`.
- Nettoyage :
  - tri par `id` et `cycle`,
  - gestion des valeurs manquantes (forward-fill par moteur puis remplissage résiduel simple).

### 3.2 Feature engineering

À partir des capteurs et des cycles, je crée plusieurs blocs de features :

- **Features temporelles simples**
  - `log_cycle`, `sqrt_cycle`, `cycle_squared`.
- **Lags (valeurs passées)**  
  Décalages sur 1, 3, 5 cycles pour les capteurs les plus importants.
- **Deltas par rapport à l’état “neuf”**
  - `sensor_X_delta_first` = valeur actuelle – valeur au premier cycle de ce moteur.
- **Rolling windows**
  - moyennes glissantes (`rolling_mean`) sur 5 cycles,
  - volatilité (`rolling_std`) sur 10 cycles pour quelques capteurs.
- **Ratios de capteurs**
  - sélection des capteurs à plus forte variance,
  - création de ratios entre ces capteurs pour capturer des interactions physiques.

Ensuite :

- extraction des variables numériques,
- suppression des features **trop corrélées** (seuil > 0.98),
- constitution de la liste finale `FEATURE_COLS`.

### 3.3 Construction de la cible

1. **Calcul de la RUL**

   - Pour le **train** :
     - pour chaque moteur `id`, je récupère le `cycle` maximum,
     - RUL = `max_cycle(id) - cycle`.
   - Pour le **test** :
     - `PM_truth.csv` donne la RUL au dernier cycle de chaque moteur,
     - je reconstruis la RUL pour tous les cycles en remontant à partir de cette valeur.

2. **Fenêtre de maintenance**

   - choix métier : `MAINTENANCE_WINDOW = 30` cycles,
   - objectif : identifier les moteurs qui approchent d’une fin de vie **à court terme**.

3. **Variable cible**

   - création de `label_maintenance` :

     ```python
     TARGET_COL = "label_maintenance"
     df[TARGET_COL] = (df["RUL"] <= MAINTENANCE_WINDOW).astype(int)
     ```

   - `1` = moteur à risque (maintenance à planifier rapidement),
   - `0` = moteur non à risque.

   La distribution est **déséquilibrée** : peu d’exemples positifs, ce qui motive l’usage de `class_weight` / `scale_pos_weight` dans les modèles.

### 3.4 Split train / validation / test

Le split est fait **par moteur** pour éviter toute fuite temporelle :

- un même `id` ne peut pas apparaître à la fois dans `train` et `validation`,
- cela simule le fait de généraliser sur de **nouveaux moteurs** jamais vus.

Concrètement :

- échantillonnage des `id` -> `train_ids` / `val_ids`,
- vérification : `assert len(set(train_ids) & set(val_ids)) == 0`

- construction des jeux :
  - `X_train`, `y_train`
  - `X_val`, `y_val`
  - `X_test`, `y_test` (dérivé de `PM_test` + `PM_truth`)

### 3.5 Modèles testés

Plusieurs modèles de difficulté croissante :

- **DummyClassifier**
  - stratégie : prédire toujours la classe majoritaire,
  - sert de baseline minimale (référence “bête”).

- **LogisticRegression** (dans un pipeline avec `StandardScaler`)
  - gestion du déséquilibre : `class_weight="balanced"`,
  - modèle linéaire, rapide et interprétable (poids par feature).

- **RandomForestClassifier**
  - forêt d’arbres pour capter des interactions non linéaires,
  - `class_weight="balanced"` pour mieux traiter la classe minoritaire,
  - importance des features disponible pour interprétation.

- **XGBClassifier** (modèle final retenu)
  - principaux paramètres (ordre de grandeur) :
    - `n_estimators = 400`
    - `max_depth = 6`
    - `learning_rate = 0.05`
    - `subsample = 0.8`
    - `colsample_bytree = 0.8`
    - `scale_pos_weight` ajusté en fonction du ratio négatifs / positifs
  - bon compromis entre performance, robustesse et temps d’entraînement.

### 3.6 Métriques d’évaluation

Pour comparer les modèles, j’utilise :

- **Accuracy** globale,
- **F1-score** (macro / par classe, avec focus sur la classe `1`),
- **ROC-AUC**,
- **Classification report** détaillé,
- **Matrice de confusion** (lecture rapide des faux positifs / faux négatifs).

Les scores sont calculés sur :

- le **jeu de validation** pour le choix de modèle et d’hyperparamètres,
- le **jeu de test** pour la performance finale.


## 4. Résultats

Le meilleur modèle retenu est le **XGBClassifier**, qui a montré une excellente capacité à discriminer les moteurs sains des moteurs à risque, tout en maintenant un équilibre satisfaisant entre précision et rappel pour la classe minoritaire (moteurs à risque).

### 4.1 Performances sur le jeu de test

Le modèle a été entraîné sur l'ensemble du jeu d'entraînement (train + validation) et évalué une seule fois sur le jeu de test (`X_test`, `y_test`).

Voici les scores obtenus :

| Métrique   | Score  | Interprétation                                                                 |
|-----------|--------|----------------------------------------------------------------------------------|
| Accuracy  | 99.2 % | Le modèle prédit correctement l'état du moteur dans la quasi-totalité des cas.  |
| ROC-AUC   | 0.998  | Capacité quasi-parfaite à classer les moteurs à risque plus haut que les sains. |
| F1-Score  | 0.853  | Bonne performance sur la classe positive (moteurs à risque) malgré le déséquilibre. |

### 4.2 Analyse des prédictions

La matrice de confusion sur le jeu de test révèle les points forts et les limites du modèle :

- **Faux Négatifs (FN)** : très peu nombreux. C'est un point crucial en maintenance prédictive, car manquer une panne peut avoir des conséquences graves.
- **Faux Positifs (FP)** : restent contenus. Les fausses alertes sont limitées, ce qui évite des interventions de maintenance inutiles et coûteuses.

### 4.3 Interprétabilité du modèle

Pour comprendre pourquoi le modèle prend ses décisions, j'ai utilisé deux méthodes d'interprétabilité :

**Feature importance (XGBoost)**  
- Les variables les plus influentes sont principalement des moyennes glissantes (*rolling mean*) sur les capteurs clés (par exemple `s4_rolling_mean_5`, `s11_rolling_mean_5`).
- Cela confirme que la tendance récente des capteurs est plus informative que leur valeur instantanée.

**SHAP (SHapley Additive exPlanations)**  
- L'analyse SHAP valide la pertinence physique du modèle. Par exemple, une augmentation anormale de la température ou de la pression (capteurs `s11`, `s4`) pousse fortement la prédiction vers la classe "à risque".
- Les ratios entre capteurs (par exemple `s9_over_s14`) apparaissent également comme des indicateurs pertinents, capturant des relations thermodynamiques complexes.



