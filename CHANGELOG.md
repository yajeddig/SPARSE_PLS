# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [Unreleased]

### À venir
- Améliorations futures ici

---

## [0.1.3] - 2025-01-XX (À venir)

### Added
- Méthode `optimize_parameters()` pour l'optimisation automatique des hyperparamètres
- Script de benchmarks complet (`benchmarks.py`) avec 5 scénarios de test
- Documentation des benchmarks (`BENCHMARKS.md`)
- 5 nouveaux exemples dans `sandbox.ipynb`
  - Régression multi-targets
  - Intégration dans pipelines sklearn
  - Visualisation des composantes latentes
  - Application sur dataset réel (Diabetes)
  - Comparaison de modèles
- Test unitaire pour `optimize_parameters()`
- Workflow GitHub Actions pour tests automatiques sur PR
- Script de release automatisé (`release.sh`)
- CHANGELOG.md pour suivre les modifications

### Changed
- Workflow PyPI amélioré : déclenchement par tags au lieu de push direct
- Tests exécutés automatiquement avant publication
- Création automatique de GitHub Releases

### Fixed
- N/A

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

[Unreleased]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.2...HEAD
[0.1.3]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/yajeddig/SPARSE_PLS/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/yajeddig/SPARSE_PLS/releases/tag/v0.1.0
