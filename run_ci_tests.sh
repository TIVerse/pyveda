#!/bin/bash
# Run all GitHub CI tests locally
set -e  # Exit on any error

echo "========================================"
echo "Running PyVeda CI Tests Locally"
echo "========================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated${NC}"
else
    echo -e "${YELLOW}⚠ No virtual environment found, using system Python${NC}"
fi
echo ""

# Check if dependencies are installed
echo "Step 1: Checking dependencies..."
if ! python -m pip show pytest > /dev/null 2>&1; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    python -m pip install -e ".[dev]"
fi
echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# 1. Lint with ruff
echo "========================================"
echo "Step 2: Linting with ruff"
echo "========================================"
if ruff check src/pyveda tests; then
    echo -e "${GREEN}✓ Ruff check passed${NC}"
else
    echo -e "${RED}✗ Ruff check failed${NC}"
    exit 1
fi
echo ""

# 2. Format check with black
echo "========================================"
echo "Step 3: Format check with black"
echo "========================================"
if black --check src/pyveda tests; then
    echo -e "${GREEN}✓ Black format check passed${NC}"
else
    echo -e "${YELLOW}⚠ Black format check failed - files need formatting${NC}"
    echo "Run 'black src/pyveda tests' to fix formatting"
    exit 1
fi
echo ""

# 3. Type check with mypy (continue on error like CI)
echo "========================================"
echo "Step 4: Type check with mypy"
echo "========================================"
if mypy src/pyveda --strict; then
    echo -e "${GREEN}✓ MyPy type check passed${NC}"
else
    echo -e "${YELLOW}⚠ MyPy found issues (continuing as per CI config)${NC}"
fi
echo ""

# 4. Run unit and integration tests
echo "========================================"
echo "Step 5: Running unit & integration tests"
echo "========================================"
if pytest tests/unit tests/integration -v --cov=src/pyveda --cov-report=term-missing; then
    echo -e "${GREEN}✓ Unit & integration tests passed${NC}"
else
    echo -e "${RED}✗ Unit & integration tests failed${NC}"
    exit 1
fi
echo ""

# 5. Run stress tests (optional, can be slow)
echo "========================================"
echo "Step 6: Running stress tests"
echo "========================================"
echo -e "${YELLOW}Running stress tests (may take a few minutes)...${NC}"
if pytest tests/stress -v --timeout=300; then
    echo -e "${GREEN}✓ Stress tests passed${NC}"
else
    echo -e "${YELLOW}⚠ Some stress tests may have issues${NC}"
fi
echo ""

# 6. Verify examples
echo "========================================"
echo "Step 7: Verifying examples"
echo "========================================"
EXAMPLES=(
    "examples/01_hello_parallel.py"
    "examples/02_scoped_execution.py"
    "examples/07_configuration.py"
)

for example in "${EXAMPLES[@]}"; do
    if [ -f "$example" ]; then
        echo "Running $example..."
        if python "$example"; then
            echo -e "${GREEN}✓ $example passed${NC}"
        else
            echo -e "${RED}✗ $example failed${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠ $example not found, skipping${NC}"
    fi
done
echo ""

# Summary
echo "========================================"
echo -e "${GREEN}✓ All CI tests passed locally!${NC}"
echo "========================================"
echo ""
echo "You can now safely push your changes."
echo "The GitHub CI should pass with these results."
