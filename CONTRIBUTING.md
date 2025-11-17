# Guide de Contribution

Merci de votre intérêt pour contribuer à SPARSE_PLS ! Ce document explique comment contribuer au projet.

## 🚀 Démarrage Rapide

### 1. Fork et Clone

```bash
# Fork le repository sur GitHub, puis :
git clone https://github.com/VOTRE_USERNAME/SPARSE_PLS.git
cd SPARSE_PLS
```

### 2. Configuration de l'environnement

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
pip install pytest pytest-cov matplotlib black isort flake8

# Installer le package en mode développement
pip install -e .
```

### 3. Créer une branche

```bash
git checkout -b feature/ma-nouvelle-fonctionnalite
# ou
git checkout -b fix/correction-bug
```

## 📝 Workflow de Développement

### Écrire du code

1. **Suivez le style PEP 8**
   ```bash
   # Formater avec black
   black sparse_pls/ tests/

   # Trier les imports
   isort sparse_pls/ tests/

   # Vérifier avec flake8
   flake8 sparse_pls/ tests/ --max-line-length=120
   ```

2. **Ajoutez des tests**
   - Tous les nouveaux codes doivent avoir des tests
   - Placez les tests dans `tests/`
   - Nommez les fichiers `test_*.py`

3. **Documentez votre code**
   - Utilisez des docstrings (format NumPy/SciPy)
   - Ajoutez des exemples si pertinent
   - Mettez à jour la documentation si nécessaire

### Exécuter les tests

```bash
# Tous les tests
pytest tests/ -v

# Avec coverage
pytest tests/ -v --cov=sparse_pls --cov-report=html

# Un test spécifique
pytest tests/test_model.py::test_optimize_parameters -v
```

### Commiter les changements

```bash
# Staging
git add .

# Commit avec un message descriptif
git commit -m "feat: ajoute la fonctionnalité X

Description détaillée de ce qui a été fait et pourquoi.
"
```

**Format des messages de commit :**
- `feat:` Nouvelle fonctionnalité
- `fix:` Correction de bug
- `docs:` Documentation uniquement
- `style:` Formatage, point-virgules manquants, etc.
- `refactor:` Refactoring sans changement de fonctionnalité
- `test:` Ajout de tests
- `chore:` Tâches de maintenance

### Créer une Pull Request

```bash
# Pusher votre branche
git push origin feature/ma-nouvelle-fonctionnalite

# Puis créer une PR sur GitHub
```

**Dans la PR, incluez :**
- ✅ Description claire de ce qui a été fait
- ✅ Pourquoi ce changement est nécessaire
- ✅ Références aux issues (si applicable)
- ✅ Screenshots (si applicable)
- ✅ Tests ajoutés

## 🧪 Standards de Qualité

### Tests

- **Coverage minimale :** 80%
- **Tous les tests doivent passer** sur Python 3.8, 3.9, 3.10, 3.11
- **Tests sur OS :** Ubuntu, Windows, macOS

### Code Style

```python
# ✅ BON
def fit(self, X: np.ndarray, Y: np.ndarray) -> 'SparsePLS':
    """
    Fit the Sparse PLS model to the training data.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training data.
    Y : np.ndarray of shape (n_samples,) or (n_samples, n_targets)
        Target values.

    Returns
    -------
    SparsePLS
        Fitted model instance.
    """
    # Implementation...
    return self

# ❌ MAUVAIS
def fit(self,X,Y):
    # No docstring, no type hints, poor spacing
    return self
```

### Documentation

- Utilisez le format de docstring NumPy/SciPy
- Incluez des exemples pour les fonctionnalités publiques
- Mettez à jour `readme.md` si nécessaire
- Ajoutez des entrées dans `CHANGELOG.md`

## 🐛 Signaler des Bugs

**Avant de créer une issue :**
1. Vérifiez que le bug n'a pas déjà été signalé
2. Testez avec la dernière version

**Créez une issue avec :**
- Description claire du problème
- Étapes pour reproduire
- Comportement attendu vs actuel
- Votre environnement (OS, Python version, etc.)
- Code minimal pour reproduire

**Template :**
```markdown
### Description
Brève description du bug

### Étapes pour reproduire
1. Faire ceci
2. Faire cela
3. Observer le problème

### Comportement attendu
Ce qui devrait se passer

### Comportement actuel
Ce qui se passe réellement

### Environnement
- OS: Ubuntu 22.04
- Python: 3.9.7
- sparse_pls: 0.1.2
- Installation: pip
```

## 💡 Proposer des Fonctionnalités

**Avant de proposer :**
1. Vérifiez que ça n'existe pas déjà
2. Assurez-vous que c'est dans le scope du projet

**Créez une issue avec :**
- Description claire de la fonctionnalité
- Use case / cas d'utilisation
- Exemple de code souhaité (API proposal)
- Pourquoi c'est important

## 🔄 Process de Review

**Après avoir créé une PR :**

1. **GitHub Actions** exécute automatiquement :
   - Tests sur multiple OS et versions Python
   - Vérifications de style (black, isort, flake8)
   - Coverage report

2. **Review par les mainteneurs**
   - Vérification de la qualité du code
   - Vérification des tests
   - Vérification de la documentation

3. **Modifications si nécessaire**
   - Répondez aux commentaires
   - Faites les changements demandés
   - Poussez les updates

4. **Merge**
   - Une fois approuvée, la PR sera mergée
   - Votre contribution sera dans la prochaine release ! 🎉

## 📦 Processus de Release

**Pour les mainteneurs uniquement.**

Voir [RELEASE_GUIDE.md](RELEASE_GUIDE.md) pour le processus détaillé.

**En bref :**
```bash
# Mettre à jour CHANGELOG.md avec les changements
vim CHANGELOG.md

# Exécuter le script de release
./release.sh patch  # ou minor, ou major

# Le reste est automatique via GitHub Actions
```

## 📚 Ressources Utiles

- [Documentation Scikit-learn](https://scikit-learn.org/stable/) - Pour la compatibilité
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/) - Format de documentation
- [PEP 8](https://peps.python.org/pep-0008/) - Style guide Python
- [Semantic Versioning](https://semver.org/) - Versioning guide
- [Keep a Changelog](https://keepachangelog.com/) - Changelog format

## ❓ Questions ?

- **Issues GitHub :** https://github.com/yajeddig/SPARSE_PLS/issues
- **Discussions :** https://github.com/yajeddig/SPARSE_PLS/discussions

## 🙏 Merci !

Votre contribution aide à améliorer SPARSE_PLS pour toute la communauté.

---

**Code of Conduct :** Soyez respectueux et professionnel dans toutes les interactions.
