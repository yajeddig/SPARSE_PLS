#!/bin/bash
# Script de release pour SPARSE_PLS
# Usage: ./release.sh [patch|minor|major]

set -e

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}SPARSE_PLS Release Script${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""

# Vérifier que nous sommes sur la branche main
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "main" ]; then
    echo -e "${RED}❌ Erreur: Vous devez être sur la branche 'main' pour faire une release${NC}"
    echo -e "Branche actuelle: $CURRENT_BRANCH"
    exit 1
fi

# Vérifier qu'il n'y a pas de modifications non commitées
if ! git diff-index --quiet HEAD --; then
    echo -e "${RED}❌ Erreur: Il y a des modifications non commitées${NC}"
    echo -e "Veuillez commiter ou stasher vos changements avant de continuer."
    git status --short
    exit 1
fi

# Récupérer la version actuelle depuis setup.py
CURRENT_VERSION=$(grep "version=" setup.py | sed "s/.*version='\([^']*\)'.*/\1/")
echo -e "Version actuelle: ${YELLOW}$CURRENT_VERSION${NC}"

# Déterminer le type de bump
BUMP_TYPE=${1:-patch}

if [ "$BUMP_TYPE" != "patch" ] && [ "$BUMP_TYPE" != "minor" ] && [ "$BUMP_TYPE" != "major" ]; then
    echo -e "${RED}❌ Type de bump invalide: $BUMP_TYPE${NC}"
    echo "Usage: ./release.sh [patch|minor|major]"
    exit 1
fi

# Calculer la nouvelle version
IFS='.' read -ra VERSION_PARTS <<< "$CURRENT_VERSION"
MAJOR=${VERSION_PARTS[0]}
MINOR=${VERSION_PARTS[1]}
PATCH=${VERSION_PARTS[2]}

case "$BUMP_TYPE" in
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    patch)
        PATCH=$((PATCH + 1))
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo -e "Nouvelle version: ${GREEN}$NEW_VERSION${NC} (bump: $BUMP_TYPE)"
echo ""

# Demander confirmation
read -p "Continuer avec cette release? (y/N) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Release annulée${NC}"
    exit 0
fi

# Mettre à jour setup.py
echo -e "${GREEN}📝 Mise à jour de setup.py...${NC}"
sed -i "s/version='[^']*'/version='$NEW_VERSION'/" setup.py

# Mettre à jour __init__.py si présent
if [ -f "sparse_pls/__init__.py" ]; then
    if grep -q "__version__" sparse_pls/__init__.py; then
        sed -i "s/__version__ = .*/__version__ = '$NEW_VERSION'/" sparse_pls/__init__.py
    else
        echo "__version__ = '$NEW_VERSION'" >> sparse_pls/__init__.py
    fi
fi

# Mettre à jour ou créer CHANGELOG.md
echo -e "${GREEN}📝 Mise à jour de CHANGELOG.md...${NC}"
DATE=$(date +%Y-%m-%d)

if [ ! -f "CHANGELOG.md" ]; then
    cat > CHANGELOG.md <<EOF
# Changelog

Toutes les modifications notables de ce projet seront documentées dans ce fichier.

Le format est basé sur [Keep a Changelog](https://keepachangelog.com/fr/1.0.0/),
et ce projet adhère au [Semantic Versioning](https://semver.org/lang/fr/).

## [$NEW_VERSION] - $DATE

### Added
- Première release

EOF
else
    # Insérer la nouvelle version après le header
    TEMP_FILE=$(mktemp)
    awk -v version="$NEW_VERSION" -v date="$DATE" '
        /^## \[Unreleased\]/ {
            print
            print ""
            print "## [" version "] - " date
            print ""
            print "### Added"
            print "- "
            print ""
            print "### Changed"
            print "- "
            print ""
            print "### Fixed"
            print "- "
            next
        }
        { print }
    ' CHANGELOG.md > "$TEMP_FILE"
    mv "$TEMP_FILE" CHANGELOG.md
fi

echo -e "${YELLOW}⚠️  Veuillez éditer CHANGELOG.md pour ajouter les détails de cette release${NC}"
echo -e "Appuyez sur Entrée quand c'est fait..."
read

# Commit les changements
echo -e "${GREEN}💾 Commit des changements...${NC}"
git add setup.py CHANGELOG.md sparse_pls/__init__.py 2>/dev/null || true
git commit -m "chore: bump version to $NEW_VERSION"

# Créer le tag
echo -e "${GREEN}🏷️  Création du tag v$NEW_VERSION...${NC}"
git tag -a "v$NEW_VERSION" -m "Release version $NEW_VERSION"

# Push
echo -e "${GREEN}⬆️  Push vers origin...${NC}"
git push origin main
git push origin "v$NEW_VERSION"

echo ""
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}✅ Release créée avec succès!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Version: ${GREEN}v$NEW_VERSION${NC}"
echo -e "Tag créé et poussé vers GitHub"
echo ""
echo -e "${YELLOW}📦 Le workflow GitHub Actions va maintenant:${NC}"
echo "  1. Exécuter les tests"
echo "  2. Builder le package"
echo "  3. Publier sur PyPI"
echo "  4. Créer une GitHub Release"
echo ""
echo -e "Suivez le processus sur: ${YELLOW}https://github.com/yajeddig/SPARSE_PLS/actions${NC}"
echo ""
