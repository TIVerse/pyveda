#!/bin/bash
# Rename project from pyveda to vedart
# This script updates all references throughout the codebase

set -e

echo "========================================"
echo "Renaming project: pyveda → VedaRT"
echo "========================================"
echo ""

# Update all Python files - imports and strings
echo "Step 1: Updating Python source files..."
find src -type f -name "*.py" -exec sed -i 's/pyveda/vedart/g' {} +
find tests -type f -name "*.py" -exec sed -i 's/pyveda/vedart/g' {} +
find examples -type f -name "*.py" -exec sed -i 's/pyveda/vedart/g' {} +

# Update documentation files
echo "Step 2: Updating documentation..."
find docs -type f -name "*.md" -exec sed -i 's/pyveda/vedart/g' {} +
find docs -type f -name "*.md" -exec sed -i 's/PyVeda/VedaRT/g' {} +
find docs -type f -name "*.md" -exec sed -i 's/pip install vedart/pip install vedart/g' {} +

# Update GitHub workflows
echo "Step 3: Updating GitHub workflows..."
find .github -type f -name "*.yml" -exec sed -i 's/pyveda/vedart/g' {} +
find .github -type f -name "*.yml" -exec sed -i 's/PyVeda/VedaRT/g' {} +

# Update test configuration
echo "Step 4: Updating test configuration..."
if [ -f "pytest.ini" ]; then
    sed -i 's/pyveda/vedart/g' pytest.ini
fi

if [ -f ".coveragerc" ]; then
    sed -i 's/pyveda/vedart/g' .coveragerc
fi

# Update CI test script
echo "Step 5: Updating CI test script..."
if [ -f "run_ci_tests.sh" ]; then
    sed -i 's/pyveda/vedart/g' run_ci_tests.sh
    sed -i 's/PyVeda/VedaRT/g' run_ci_tests.sh
fi

echo ""
echo "========================================"
echo "✓ References updated"
echo "========================================"
echo ""
echo "📝 Manual steps required:"
echo ""
echo "1. Rename source directory:"
echo "   mv src/pyveda src/vedart"
echo ""
echo "2. Update pyproject.toml packages section (if needed)"
echo ""
echo "3. Rename repository on GitHub:"
echo "   Settings → General → Repository name"
echo "   Change from 'pyveda' to 'vedart'"
echo ""
echo "4. Update git remote URL:"
echo "   git remote set-url origin https://github.com/TIVerse/vedart.git"
echo ""
echo "5. Rename project directory:"
echo "   cd .."
echo "   mv pyveda vedart"
echo "   cd vedart"
echo ""
echo "6. Test installation:"
echo "   pip uninstall -y vedart pyveda"
echo "   pip install -e ."
echo "   python -c 'import vedart; print(vedart.__version__)'"
echo ""
echo "7. Run tests:"
echo "   ./run_ci_tests.sh"
echo ""
echo "8. Commit changes:"
echo "   git add ."
echo "   git commit -m 'refactor: rename project to VedaRT'"
echo "   git push -u origin master"
echo ""
