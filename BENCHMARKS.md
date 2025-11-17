# SparsePLS Benchmarks

Ce document décrit les benchmarks de performance pour SparsePLS.

## Exécution des Benchmarks

Pour exécuter tous les benchmarks :

```bash
python benchmarks.py
```

L'exécution complète prend environ 2-5 minutes selon votre machine.

## Benchmarks Disponibles

### 1. Dataset Size Scalability
Teste la performance de SparsePLS avec différentes tailles de datasets (100 à 2000 échantillons).

**Métriques mesurées :**
- Temps d'entraînement
- Temps de prédiction
- MSE
- Score R²

**Output :** `benchmark_dataset_size.png` / `.csv`

### 2. Robustness to Noise
Évalue la robustesse du modèle face à différents niveaux de bruit (0.0 à 5.0).

**Métriques mesurées :**
- MSE
- Score R²
- Nombre de features sélectionnées

**Output :** `benchmark_noise_levels.png` / `.csv`

### 3. Model Comparison
Compare SparsePLS avec d'autres méthodes de régression :
- Standard PLS
- Ridge Regression
- Lasso Regression
- ElasticNet

**Métriques mesurées :**
- Score de validation croisée (5-fold)
- MSE sur le test set
- Score R²
- Temps d'entraînement
- Nombre de features utilisées

**Output :** `benchmark_model_comparison.png` / `.csv`

### 4. Feature Dimensionality
Teste la performance avec différentes dimensionnalités (10 à 200 features).

**Métriques mesurées :**
- Taux de sélection de features
- Temps d'entraînement
- MSE
- Score R²

**Output :** `benchmark_feature_dimensionality.png` / `.csv`

### 5. Real-World Dataset (Diabetes)
Teste SparsePLS sur le dataset diabetes de scikit-learn avec différents paramètres de parcimonie (alpha).

**Métriques mesurées :**
- MSE
- Score R²
- Nombre de features sélectionnées

**Output :** `benchmark_real_data.png` / `.csv`

## Interprétation des Résultats

### MSE (Mean Squared Error)
Plus la valeur est basse, meilleure est la performance. Le MSE mesure l'erreur quadratique moyenne entre les prédictions et les valeurs réelles.

### R² Score
Plus la valeur est proche de 1.0, meilleure est la performance. Un R² de 1.0 indique une prédiction parfaite.

### Feature Selection
SparsePLS sélectionne automatiquement les features les plus pertinentes. Un bon modèle sélectionne peu de features tout en maintenant une bonne performance.

## Exécution Individuelle

Pour exécuter un benchmark spécifique :

```python
from benchmarks import benchmark_dataset_size

results = benchmark_dataset_size()
print(results)
```

Fonctions disponibles :
- `benchmark_dataset_size()`
- `benchmark_noise_levels()`
- `benchmark_model_comparison()`
- `benchmark_feature_dimensionality()`
- `benchmark_real_data()`

## Dépendances

Assurez-vous que toutes les dépendances sont installées :

```bash
pip install -r requirements.txt
pip install matplotlib
```

## Personnalisation

Vous pouvez modifier les paramètres dans `benchmarks.py` :
- Tailles de datasets : variable `sizes`
- Niveaux de bruit : variable `noise_levels`
- Nombre de features : variable `feature_counts`
- Paramètres du modèle : `n_components`, `alpha`, etc.

## Notes

- Les graphiques sont sauvegardés en haute résolution (300 DPI)
- Les résultats numériques sont exportés en CSV pour analyse ultérieure
- Les temps d'exécution peuvent varier selon votre matériel
- Pour des résultats reproductibles, tous les benchmarks utilisent `random_state=42`
