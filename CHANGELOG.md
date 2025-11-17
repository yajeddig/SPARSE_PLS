# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Unreleased]

### À venir
- Améliorations futures ici

---

## [0.1.3] - 2025-11-17

### Added
- Méthode `optimize_parameters()` pour l'optimisation automatique des hyperparamètres via validation croisée
- Script de benchmarks complet (`benchmarks.py`) avec 5 scénarios de test
  - Dataset Size Scalability (100-2000 échantillons)
  - Robustness to Noise (niveaux 0.0-5.0)
  - Model Comparison (vs Ridge, Lasso, ElasticNet, PLS)
  - Feature Dimensionality (10-200 features)
  - Real-World Dataset (Diabetes)
- 5 nouveaux exemples dans `sandbox.ipynb`
  - Régression multi-targets (multi-output)
  - Intégration dans pipelines sklearn
  - Visualisation des composantes latentes
  - Application sur dataset réel (Diabetes)
  - Comparaison avec autres modèles
- Test unitaire pour `optimize_parameters()` dans `tests/test_model.py`
- Workflow GitHub Actions pour tests automatiques sur PRs (`.github/workflows/test.yml`)
  - Tests multi-OS (Ubuntu, Windows, macOS)
  - Tests multi-Python (3.8, 3.9, 3.10, 3.11)
  - Vérifications de qualité (black, isort, flake8)
  - Coverage reporting
- Script de release automatisé (`release.sh`) pour faciliter les publications
- Documentation complète
  - `QUICKSTART_RELEASE.md` - Guide rapide 5 minutes
  - `RELEASE_GUIDE.md` - Guide complet du processus de release
  - `CONTRIBUTING.md` - Guide de contribution
  - `BENCHMARKS.md` - Documentation des benchmarks
  - `AUTOMATION_SETUP_SUMMARY.md` - Résumé de la configuration
  - `.github/PULL_REQUEST_TEMPLATE.md` - Template standardisé
- `__version__` dans `sparse_pls/__init__.py` pour versioning runtime
- `.gitattributes` pour gestion correcte des line endings

### Changed
- Workflow PyPI (`.github/workflows/publish-pypi.yml`) amélioré
  - Déclenchement par tags Git `v*.*.*` au lieu de push direct sur main
  - Tests automatiques exécutés avant publication
  - Création automatique de GitHub Releases avec artifacts
  - Extraction automatique de la version depuis le tag
- `requirements.txt` : versions exactes remplacées par ranges flexibles
  - `numpy: 1.26.4 → >=1.20.0,<2.0.0` (compatibilité Python 3.8)
  - `pandas: 2.2.2 → >=1.2.0`
  - `scikit-learn: 1.5.1 → >=1.0.0`
  - `scipy: 1.13.1 → >=1.6.0`
  - `joblib: 1.4.2 → >=1.0.0`
- `readme.md` amélioré avec
  - Badges PyPI, versions Python, license, tests, downloads
  - Section installation depuis PyPI
  - Section développement et contribution
  - Section processus de release

### Fixed
- Validation des données : remplacement de `_validate_data()` par `check_X_y()` pour meilleure compatibilité sklearn
- Formatage de tous les fichiers Python avec black
- Type hints Python 3.8 : `tuple[...]` remplacé par `Tuple[...]` (import from typing)
- Gestion explicite des feature names pour DataFrames
- Compatibilité multi-versions Python (3.8-3.11) et multi-OS (Ubuntu, Windows, macOS)

---

## [0.1.2] - 2024-XX-XX

### Added
- Tests unitaires complets
- Documentation technique et scientifique
- Classe `DataPreprocessor` pour le prétraitement des données

### Changed
- Amélioration de la gestion des noms de colonnes
- Correction des problèmes de casse (README.md)

---

## [0.1.1] - 2024-XX-XX

### Added
- Configuration GitHub Actions initiale
- Support pour pandas DataFrames
- Méthodes de visualisation (`plot_weights`, `plot_selected_features`)

---

## [0.1.0] - 2024-XX-XX

### Added
- Implémentation initiale de SparsePLS
- Support pour la régression simple et multi-target
- Sélection de variables par parcimonie L1
- Réduction de dimensionnalité
- Compatibilité avec scikit-learn

---

[Unreleased]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yajeddig/SPARSE_PLS/releases/tag/v0.1.0
