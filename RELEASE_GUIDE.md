# Guide de Release pour SPARSE_PLS

Ce guide explique comment publier une nouvelle version de SPARSE_PLS sur PyPI de manière automatisée.

## 📋 Pré-requis

### 1. Configuration GitHub Secrets

Assurez-vous que les secrets suivants sont configurés dans votre repository GitHub :

**Settings → Secrets and variables → Actions → New repository secret**

- **`PYPI_API_TOKEN`** : Token API PyPI
  - Aller sur https://pypi.org/manage/account/token/
  - Créer un nouveau token avec scope "Entire account" ou spécifique au projet
  - Copier le token (commence par `pypi-...`)

### 2. Vérifications avant release

```bash
# Vous êtes sur la branche main
git checkout main
git pull origin main

# Tous les tests passent
python -m pytest tests/

# Aucune modification non commitée
git status
```

## 🚀 Processus de Release Automatisé

### Méthode 1: Script automatique (RECOMMANDÉ)

Le script `release.sh` automatise tout le processus :

```bash
# Release patch (0.1.2 → 0.1.3)
./release.sh patch

# Release minor (0.1.2 → 0.2.0)
./release.sh minor

# Release major (0.1.2 → 1.0.0)
./release.sh major
```

**Le script va :**
1. ✅ Vérifier que vous êtes sur `main`
2. ✅ Vérifier qu'il n'y a pas de modifications non commitées
3. ✅ Calculer la nouvelle version
4. ✅ Mettre à jour `setup.py`
5. ✅ Mettre à jour `CHANGELOG.md`
6. ⏸️  **Vous demander d'éditer le CHANGELOG**
7. ✅ Commiter les changements
8. ✅ Créer le tag Git
9. ✅ Pusher vers GitHub

**Ensuite, GitHub Actions va automatiquement :**
- ✅ Exécuter tous les tests
- ✅ Builder le package
- ✅ Publier sur PyPI
- ✅ Créer une GitHub Release

### Méthode 2: Manuelle

Si vous préférez le contrôle total :

#### Étape 1: Mettre à jour la version

```bash
# Éditer setup.py
vim setup.py
# Changer version='0.1.2' → version='0.1.3'
```

#### Étape 2: Mettre à jour le CHANGELOG

```bash
vim CHANGELOG.md
# Ajouter une section pour la nouvelle version avec les changements
```

#### Étape 3: Commiter

```bash
git add setup.py CHANGELOG.md
git commit -m "chore: bump version to 0.1.3"
```

#### Étape 4: Créer le tag

```bash
# Créer un tag annoté
git tag -a v0.1.3 -m "Release version 0.1.3"
```

#### Étape 5: Pusher

```bash
# Pusher le commit
git push origin main

# Pusher le tag (IMPORTANT: c'est ce qui déclenche la publication)
git push origin v0.1.3
```

## 📊 Suivi de la Publication

Une fois le tag poussé :

1. **GitHub Actions** : https://github.com/yajeddig/SPARSE_PLS/actions
   - Onglet "Publish to PyPI"
   - Vérifier que le workflow s'exécute correctement

2. **PyPI** : https://pypi.org/project/sparse-pls/
   - La nouvelle version apparaîtra dans quelques minutes

3. **GitHub Releases** : https://github.com/yajeddig/SPARSE_PLS/releases
   - Une release sera créée automatiquement

## 🔧 Workflow GitHub Actions

### Workflow de Publication (`publish-pypi.yml`)

**Déclenchement :** Push de tags `v*.*.*` (ex: v0.1.3)

**Jobs :**
1. **Test** : Exécute tous les tests
2. **Build & Publish** :
   - Build le package
   - Publie sur PyPI
   - Crée une GitHub Release

### Workflow de Tests (`test.yml`)

**Déclenchement :** PR et push vers main/dev

**Tests sur :**
- OS: Ubuntu, Windows, macOS
- Python: 3.8, 3.9, 3.10, 3.11

**Vérifie :**
- Tests unitaires avec coverage
- Formatage du code (black)
- Import sorting (isort)
- Linting (flake8)

## 🐛 Résolution de Problèmes

### Le workflow GitHub Actions échoue

```bash
# Voir les logs détaillés
https://github.com/yajeddig/SPARSE_PLS/actions

# Causes communes:
# - Tests qui échouent → Corriger les tests
# - Secret PYPI_API_TOKEN manquant → Configurer le secret
# - Version déjà publiée → Incrémenter la version
```

### PyPI rejette la publication

**Erreur : "File already exists"**
- Vous essayez de publier une version qui existe déjà
- Solution : Incrémenter la version

**Erreur : "Invalid credentials"**
- Le token PyPI est invalide ou manquant
- Solution : Regénérer le token et mettre à jour le secret GitHub

### Annuler une release

```bash
# Supprimer le tag localement
git tag -d v0.1.3

# Supprimer le tag sur GitHub
git push origin :refs/tags/v0.1.3

# Note: Vous ne pouvez PAS supprimer une version déjà publiée sur PyPI
# Vous devrez publier une nouvelle version corrective
```

## 📝 Bonnes Pratiques

### Semantic Versioning

- **MAJOR** (1.0.0) : Changements incompatibles avec l'API
- **MINOR** (0.1.0) : Nouvelles fonctionnalités rétrocompatibles
- **PATCH** (0.0.1) : Corrections de bugs rétrocompatibles

### Avant chaque release

1. ✅ Tous les tests passent
2. ✅ Le CHANGELOG est à jour
3. ✅ La documentation est à jour
4. ✅ Les dépendances sont à jour
5. ✅ Le code est mergé dans `main`

### Contenu du CHANGELOG

Pour chaque version, documenter :
- **Added** : Nouvelles fonctionnalités
- **Changed** : Modifications de fonctionnalités existantes
- **Deprecated** : Fonctionnalités bientôt supprimées
- **Removed** : Fonctionnalités supprimées
- **Fixed** : Corrections de bugs
- **Security** : Correctifs de sécurité

## 🔄 Workflow Complet

```
┌─────────────────────┐
│  Développement      │
│  sur branche dev    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Pull Request       │
│  vers main          │
│  (tests auto)       │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Merge dans main    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ./release.sh patch │
│  (local)            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Push tag v0.1.3    │
│  vers GitHub        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  GitHub Actions     │
│  - Tests            │
│  - Build            │
│  - Publish PyPI     │
│  - Create Release   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  ✅ Package publié  │
│  sur PyPI           │
└─────────────────────┘
```

## 📞 Support

En cas de problème :
- Issues GitHub : https://github.com/yajeddig/SPARSE_PLS/issues
- Documentation PyPI : https://packaging.python.org/
- Documentation GitHub Actions : https://docs.github.com/en/actions
