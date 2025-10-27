#!/bin/bash
# Script to rename the project from pyveda to a new name
# Usage: ./rename_project.sh <new_name>

set -e

if [ $# -eq 0 ]; then
    echo "Usage: ./rename_project.sh <new_name>"
    echo "Example: ./rename_project.sh veda-py"
    exit 1
fi

NEW_NAME=$1
OLD_NAME="pyveda"

echo "========================================"
echo "Renaming project from '$OLD_NAME' to '$NEW_NAME'"
echo "========================================"
echo ""

# Confirm with user
read -p "This will update all references. Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Step 1: Updating pyproject.toml..."
sed -i "s/name = \"$OLD_NAME\"/name = \"$NEW_NAME\"/" pyproject.toml

echo "Step 2: Updating README.md..."
sed -i "s/pip install $OLD_NAME/pip install $NEW_NAME/g" README.md
sed -i "s/$OLD_NAME\[/$NEW_NAME\[/g" README.md

echo "Step 3: Updating documentation..."
find docs -type f -name "*.md" -exec sed -i "s/pip install $OLD_NAME/pip install $NEW_NAME/g" {} +
find docs -type f -name "*.md" -exec sed -i "s/name = \"$OLD_NAME\"/name = \"$NEW_NAME\"/g" {} +

echo "Step 4: Updating GitHub workflows..."
find .github/workflows -type f -name "*.yml" -exec sed -i "s/pip install -e/pip install -e/g" {} +

echo "Step 5: Updating examples..."
# Update import statements if the module name needs to change
# find examples -type f -name "*.py" -exec sed -i "s/import $OLD_NAME/import ${NEW_NAME//-/_}/g" {} +

echo ""
echo "========================================"
echo "✓ Project renamed to '$NEW_NAME'"
echo "========================================"
echo ""
echo "⚠️  IMPORTANT: Manual steps required:"
echo "1. If module name differs from package name (e.g., veda-py → veda_py):"
echo "   - Rename src/pyveda/ directory to src/veda_py/"
echo "   - Update all import statements in code"
echo ""
echo "2. Update GitHub repository name:"
echo "   - Go to repository Settings → General"
echo "   - Change repository name"
echo "   - Update git remote: git remote set-url origin <new-url>"
echo ""
echo "3. Verify changes:"
echo "   git diff"
echo ""
echo "4. Test installation:"
echo "   pip install -e ."
echo ""
echo "5. Commit changes:"
echo "   git add ."
echo "   git commit -m 'refactor: rename project to $NEW_NAME'"
echo ""
