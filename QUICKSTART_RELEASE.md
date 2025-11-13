# 🚀 Guide Rapide : Publication Automatique sur PyPI

Ce guide explique en 5 minutes comment le nouveau système de publication automatique fonctionne.

## ⚡ TL;DR - Pour Publier une Nouvelle Version

```bash
# 1. Assurez-vous d'être sur main avec tous les changements mergés
git checkout main
git pull origin main

# 2. Exécutez le script de release
./release.sh patch  # ou minor, ou major

# 3. Éditez le CHANGELOG quand demandé, puis appuyez sur Entrée

# 4. C'est tout ! GitHub Actions s'occupe du reste
```

**Le package sera automatiquement publié sur PyPI dans ~5 minutes.**

---

## 🔄 Ancien vs Nouveau Workflow

### ❌ Ancien Workflow (Problématique)

```
Push sur main → ⚠️ Publication automatique
Problème : Publie la même version à chaque push = ERREUR
```

### ✅ Nouveau Workflow (Amélioré)

```
PR mergée → Tag créé (v0.1.3) → Tests → Build → PyPI
Avantages :
  ✓ Contrôle total sur les releases
  ✓ Tests automatiques avant publication
  ✓ Versioning automatique
  ✓ GitHub Releases créées automatiquement
```

---

## 📦 Comprendre le Nouveau Système

### 1. Développement (Branches de travail)

```bash
# Créer une branche pour votre feature
git checkout -b feature/nouvelle-fonctionnalite

# Développer et commiter
git add .
git commit -m "feat: ajoute nouvelle fonctionnalité"

# Pusher et créer une PR
git push origin feature/nouvelle-fonctionnalite
```

**→ GitHub Actions exécute les tests automatiquement sur la PR**

### 2. Merge dans Main

```bash
# Après review, merger la PR via l'interface GitHub
# OU en local:
git checkout main
git merge feature/nouvelle-fonctionnalite
git push origin main
```

**→ Rien n'est encore publié ! Vous contrôlez quand.**

### 3. Créer une Release

```bash
# Utiliser le script automatique
./release.sh patch

# Ce qui se passe :
# ✓ Vérifie que vous êtes sur main
# ✓ Calcule la nouvelle version (0.1.2 → 0.1.3)
# ✓ Met à jour setup.py et __init__.py
# ✓ Vous demande d'éditer CHANGELOG.md
# ✓ Crée un commit et un tag
# ✓ Push vers GitHub
```

### 4. Publication Automatique

Une fois le tag `v0.1.3` poussé, GitHub Actions :

```
1. ✓ Exécute tous les tests (pytest)
2. ✓ Build le package (.whl et .tar.gz)
3. ✓ Vérifie avec twine
4. ✓ Publie sur PyPI
5. ✓ Crée une GitHub Release
```

**Suivi :** https://github.com/yajeddig/SPARSE_PLS/actions

---

## 🎯 Semantic Versioning

Le script `release.sh` supporte 3 types de releases :

| Commande | Exemple | Quand l'utiliser |
|----------|---------|------------------|
| `./release.sh patch` | 0.1.2 → 0.1.3 | Bug fixes, petites corrections |
| `./release.sh minor` | 0.1.2 → 0.2.0 | Nouvelles fonctionnalités (rétrocompatibles) |
| `./release.sh major` | 0.1.2 → 1.0.0 | Breaking changes (incompatibilités) |

---

## 🔧 Configuration Requise (Une Seule Fois)

### Sur GitHub

1. **Aller dans Settings → Secrets and variables → Actions**
2. **Créer un nouveau secret :**
   - Name: `PYPI_API_TOKEN`
   - Value: Votre token PyPI

**Comment obtenir un token PyPI :**
```
1. Aller sur https://pypi.org/manage/account/token/
2. Créer un nouveau token
3. Scope: "Entire account" ou spécifique au projet
4. Copier le token (commence par pypi-...)
5. Le coller dans GitHub Secrets
```

### Sur votre machine (optionnel)

Pour rendre le script release.sh exécutable :
```bash
chmod +x release.sh
```

---

## 📝 Workflow Complet - Exemple

### Scénario : Vous avez ajouté une nouvelle fonctionnalité

```bash
# Étape 1 : Développement
git checkout -b feature/optimize-parameters
# ... développement ...
git add .
git commit -m "feat: add optimize_parameters method"
git push origin feature/optimize-parameters

# Étape 2 : Créer une PR sur GitHub
# → GitHub Actions teste automatiquement

# Étape 3 : Après review, merger la PR
# Cliquer sur "Merge pull request" sur GitHub

# Étape 4 : Préparer la release
git checkout main
git pull origin main

# Étape 5 : Mettre à jour le CHANGELOG
vim CHANGELOG.md
# Ajouter les détails dans la section [Unreleased]

# Étape 6 : Créer la release
./release.sh minor  # Nouvelle fonctionnalité = minor

# Étape 7 : Le script vous demande d'éditer CHANGELOG
# Éditer pour finaliser la section de cette version
# Sauvegarder et quitter

# Étape 8 : Confirmer
# Appuyer sur 'y' pour confirmer

# ✅ Terminé ! Vérifier sur :
# - https://github.com/yajeddig/SPARSE_PLS/actions
# - https://pypi.org/project/sparse-pls/
```

---

## ❓ FAQ

### Q: Que faire si j'ai oublié d'ajouter quelque chose au CHANGELOG ?

**R:** Pas de panique ! Éditez manuellement après :
```bash
vim CHANGELOG.md
git add CHANGELOG.md
git commit --amend -m "chore: bump version to 0.1.3"
git push -f origin main
git push -f origin v0.1.3
```

### Q: Comment annuler une release si j'ai fait une erreur ?

**R:** AVANT que GitHub Actions ne publie sur PyPI :
```bash
# Supprimer le tag localement
git tag -d v0.1.3

# Supprimer le tag sur GitHub
git push origin :refs/tags/v0.1.3

# Annuler le workflow dans GitHub Actions (si en cours)
```

**Note :** Une fois publié sur PyPI, on ne peut PAS supprimer. Il faut publier une version corrective.

### Q: Le workflow GitHub Actions échoue, que faire ?

**R:** Consulter les logs :
1. Aller sur https://github.com/yajeddig/SPARSE_PLS/actions
2. Cliquer sur le workflow en erreur
3. Examiner les logs pour identifier le problème

**Causes communes :**
- Tests qui échouent → Corriger les tests
- Token PyPI invalide → Regénérer et mettre à jour le secret
- Version déjà publiée → Incrémenter la version

### Q: Puis-je toujours publier manuellement ?

**R:** Oui, mais ce n'est pas recommandé :
```bash
# Manuel (déconseillé)
python -m build
twine upload dist/*
```

### Q: Comment tester avant de publier ?

**R:** Utilisez TestPyPI :
```bash
# 1. Build
python -m build

# 2. Upload sur TestPyPI
twine upload --repository testpypi dist/*

# 3. Tester l'installation
pip install --index-url https://test.pypi.org/simple/ sparse-pls
```

---

## 📚 Ressources Additionnelles

- **Guide Complet :** [RELEASE_GUIDE.md](RELEASE_GUIDE.md)
- **Contribution :** [CONTRIBUTING.md](CONTRIBUTING.md)
- **Changelog :** [CHANGELOG.md](CHANGELOG.md)
- **GitHub Actions :** https://github.com/yajeddig/SPARSE_PLS/actions
- **PyPI Package :** https://pypi.org/project/sparse-pls/

---

## 🎉 C'est Tout !

Le système est configuré pour rendre les releases simples et fiables. Suivez simplement le workflow et tout sera automatique.

**Rappel : Pour publier une nouvelle version**
```bash
./release.sh patch
```

C'est aussi simple que ça ! 🚀
