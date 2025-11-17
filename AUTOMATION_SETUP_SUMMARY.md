# 🎉 Système d'Automatisation PyPI - Configuration Terminée

Ce document résume toutes les améliorations apportées pour automatiser la publication sur PyPI.

---

## 📦 Ce Qui a Été Fait

### ✅ Problème Résolu

**Avant :** Le workflow GitHub Actions publiait automatiquement sur PyPI à chaque push vers `main`, causant des erreurs de "version déjà publiée".

**Maintenant :** Publication contrôlée par des tags Git avec tests automatiques avant publication.

---

## 🚀 Nouveaux Fichiers Créés

### 1. Workflows GitHub Actions

| Fichier | Description |
|---------|-------------|
| `.github/workflows/publish-pypi.yml` | Publication automatique sur PyPI (déclenchée par tags) |
| `.github/workflows/test.yml` | Tests automatiques sur PRs et push (multi-OS, multi-Python) |
| `.github/PULL_REQUEST_TEMPLATE.md` | Template standardisé pour les Pull Requests |

### 2. Scripts d'Automatisation

| Fichier | Description |
|---------|-------------|
| `release.sh` | Script interactif pour créer des releases facilement |

### 3. Documentation

| Fichier | Description |
|---------|-------------|
| `RELEASE_GUIDE.md` | Guide complet du processus de release (pour mainteneurs) |
| `QUICKSTART_RELEASE.md` | Guide rapide en 5 minutes |
| `CONTRIBUTING.md` | Guide de contribution pour les développeurs |
| `CHANGELOG.md` | Historique des versions (format Keep a Changelog) |

### 4. Configuration

| Fichier | Description |
|---------|-------------|
| `.gitattributes` | Configuration des line endings pour compatibilité Windows/Linux |
| `sparse_pls/__init__.py` | Ajout de `__version__` pour versioning runtime |

### 5. Fichiers Modifiés

| Fichier | Changements |
|---------|-------------|
| `readme.md` | Ajout de badges, section développement et release |

---

## 🔧 Configuration Requise (À FAIRE)

### Étape 1: Configurer le Secret PyPI sur GitHub

**IMPORTANT:** Cette étape est OBLIGATOIRE pour que la publication automatique fonctionne.

1. **Obtenir un Token PyPI**
   - Aller sur https://pypi.org/manage/account/token/
   - Cliquer sur "Add API token"
   - Token name: `SPARSE_PLS_GitHub_Actions`
   - Scope: Sélectionner "Project: sparse-pls" (ou "Entire account")
   - Cliquer sur "Create token"
   - **COPIER LE TOKEN** (commence par `pypi-...`) - vous ne le reverrez plus !

2. **Ajouter le Token dans GitHub**
   - Aller sur https://github.com/yajeddig/SPARSE_PLS/settings/secrets/actions
   - Cliquer sur "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Coller le token PyPI
   - Cliquer sur "Add secret"

### Étape 2: Merger cette PR

Une fois le secret configuré, merger cette PR dans `main`.

---

## 📖 Comment Utiliser le Nouveau Système

### Publication d'une Nouvelle Version (Workflow Simplifié)

```bash
# 1. S'assurer d'être sur main avec tout à jour
git checkout main
git pull origin main

# 2. Lancer le script de release
./release.sh patch    # Pour bug fix (0.1.2 → 0.1.3)
./release.sh minor    # Pour nouvelle feature (0.1.2 → 0.2.0)
./release.sh major    # Pour breaking change (0.1.2 → 1.0.0)

# 3. Le script vous demandera d'éditer CHANGELOG.md
#    Ajoutez les détails de la release, sauvegardez et confirmez

# 4. Confirmer quand demandé (y/N)

# ✅ C'EST TOUT ! GitHub Actions s'occupe du reste
```

**Le système va automatiquement :**
1. ✓ Créer le tag Git (ex: v0.1.3)
2. ✓ Pusher vers GitHub
3. ✓ Déclencher GitHub Actions
4. ✓ Exécuter tous les tests
5. ✓ Builder le package
6. ✓ Publier sur PyPI
7. ✓ Créer une GitHub Release

**Suivi en temps réel :** https://github.com/yajeddig/SPARSE_PLS/actions

---

## 🔄 Nouveau Workflow Complet

### Pour les Contributeurs

```
1. Fork du repo
2. Créer une branche feature/ma-feature
3. Développer et commiter
4. Push et créer une PR
   → GitHub Actions teste automatiquement
5. Après review et approval, merge dans main
```

### Pour les Mainteneurs (Release)

```
1. Toutes les PRs mergées dans main
2. Exécuter ./release.sh [patch|minor|major]
3. Éditer CHANGELOG.md quand demandé
4. Confirmer
   → Tag créé et poussé
   → GitHub Actions publie automatiquement
```

---

## 📊 Avantages du Nouveau Système

| Avant | Après |
|-------|-------|
| ❌ Publication à chaque push | ✅ Publication contrôlée par tags |
| ❌ Erreurs de version dupliquée | ✅ Versioning automatique |
| ❌ Pas de tests avant publication | ✅ Tests obligatoires |
| ❌ Process manuel complexe | ✅ Un seul script `./release.sh` |
| ❌ Pas de GitHub Releases | ✅ Releases automatiques avec artifacts |
| ❌ Pas de CI sur PRs | ✅ Tests sur toutes les PRs |
| ❌ Tests sur un seul OS/Python | ✅ Tests multi-OS et multi-Python |

---

## 🎯 Prochaine Release (Exemple Pratique)

Voici exactement ce qu'il faudra faire pour la prochaine release (v0.1.3) :

```bash
# 1. Merger cette PR dans main
git checkout main
git pull origin main

# 2. Exécuter le script
./release.sh patch

# Le script affiche:
# Version actuelle: 0.1.2
# Nouvelle version: 0.1.3 (bump: patch)
# Continuer avec cette release? (y/N)

# 3. Taper 'y' et Entrée

# 4. L'éditeur s'ouvre avec CHANGELOG.md
# Modifier la section [0.1.3] pour ajouter:
## [0.1.3] - 2025-01-XX

### Added
- Automated PyPI publishing system
- CI/CD workflows for testing
- Release automation script
- Comprehensive documentation (RELEASE_GUIDE, CONTRIBUTING, etc.)

### Changed
- Workflow now triggers on git tags instead of push to main
- README updated with badges and development section

# 5. Sauvegarder et quitter l'éditeur
# Le script crée le commit, le tag, et push automatiquement

# 6. Aller sur https://github.com/yajeddig/SPARSE_PLS/actions
# et observer la publication automatique !

# 7. Dans 5 minutes, vérifier sur https://pypi.org/project/sparse-pls/
# La version 0.1.3 sera disponible ! 🎉
```

---

## 🐛 Résolution de Problèmes

### Le workflow GitHub Actions échoue

**Vérifier :**
1. Le secret `PYPI_API_TOKEN` est bien configuré
2. Le token PyPI est valide (pas expiré)
3. Les tests passent localement (`pytest tests/`)

**Logs :** https://github.com/yajeddig/SPARSE_PLS/actions

### Le script release.sh ne fonctionne pas

```bash
# Rendre le script exécutable
chmod +x release.sh

# Vérifier que vous êtes sur main
git checkout main
git pull origin main

# Vérifier qu'il n'y a pas de modifications non commitées
git status
```

### PyPI rejette la publication

**Erreur: "File already exists"**
- Une version avec ce numéro existe déjà
- Solution: Incrémenter la version manuellement et recréer le tag

**Erreur: "Invalid or non-existent authentication"**
- Le token PyPI est invalide
- Solution: Regénérer le token et mettre à jour le secret GitHub

---

## 📚 Documentation Disponible

| Document | Quand l'utiliser |
|----------|------------------|
| **QUICKSTART_RELEASE.md** | Commencer rapidement (5 min) |
| **RELEASE_GUIDE.md** | Guide complet pour les releases |
| **CONTRIBUTING.md** | Pour contribuer au projet |
| **CHANGELOG.md** | Voir l'historique des versions |

---

## ✅ Checklist de Mise en Route

- [ ] Configurer `PYPI_API_TOKEN` dans GitHub Secrets
- [ ] Merger cette PR dans `main`
- [ ] Tester le workflow avec une release (ex: v0.1.3)
- [ ] Vérifier que la publication sur PyPI fonctionne
- [ ] Lire QUICKSTART_RELEASE.md pour comprendre le workflow
- [ ] (Optionnel) Ajouter le badge Codecov au README

---

## 🎓 Formation d'Équipe

Pour former d'autres mainteneurs :

1. **Lire :** QUICKSTART_RELEASE.md (5 min)
2. **Regarder :** Une release en action sur GitHub Actions
3. **Pratiquer :** Faire une release de test

---

## 📞 Support

En cas de problème :
- **GitHub Issues :** https://github.com/yajeddig/SPARSE_PLS/issues
- **Documentation PyPI :** https://packaging.python.org/
- **GitHub Actions Docs :** https://docs.github.com/en/actions

---

## 🎊 Conclusion

Le système est maintenant entièrement automatisé !

**Pour publier une nouvelle version, il suffit de :**
```bash
./release.sh patch
```

Tout le reste est géré automatiquement par GitHub Actions.

**Prochaine étape :** Configurer le secret `PYPI_API_TOKEN` et faire une release de test ! 🚀

---

_Document créé le 2025-01-XX par Claude_
