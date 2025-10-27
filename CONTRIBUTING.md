# Contributing to VedaRT

Thank you for your interest in contributing to VedaRT! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)
- [Roadmap](#roadmap)

---

## Code of Conduct

Be respectful, inclusive, and professional. We aim to create a welcoming environment for all contributors.

---

## Getting Started

### Finding Issues to Work On

- Check [GitHub Issues](https://github.com/TIVerse/vedart/issues)
- Look for issues labeled `good-first-issue`
- Review the [Roadmap](docs/ROADMAP.md) for planned features
- Check [PROGRESS.md](docs/PROGRESS.md) for current status

### Before Starting

1. **Check existing issues** - Make sure someone isn't already working on it
2. **Create an issue** - For major changes, create an issue first to discuss
3. **Fork the repository** - Work in your own fork
4. **Create a branch** - Use a descriptive branch name

---

## Development Setup

### Prerequisites

- Python 3.10 or higher
- git
- virtualenv or similar tool

### Setup Steps

```bash
# 1. Fork and clone
git clone https://github.com/YOUR-USERNAME/vedart.git
cd vedart

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in development mode
pip install -e ".[dev]"

# 4. Install optional dependencies (if needed)
pip install -e ".[gpu,telemetry]"

# 5. Verify installation
python -c "import vedart as veda; print(veda.__version__)"

# 6. Run tests
pytest tests/unit -v
```

---

## Making Changes

### Branch Naming

Use descriptive branch names:
- `feature/add-distributed-executor`
- `fix/memory-leak-in-scope`
- `docs/improve-quickstart`
- `test/add-gpu-fallback-tests`

### Coding Style

We follow these conventions:

#### Python Style
- **PEP 8** with line length 88 (Black formatter)
- **Type hints** on all public APIs
- **Docstrings** for all public functions/classes (Google style)

#### Example

```python
def process_task(task: Task, timeout: float = 30.0) -> Result:
    """Process a task with optional timeout.
    
    Args:
        task: The task to process
        timeout: Maximum time to wait in seconds
        
    Returns:
        Result object containing task output
        
    Raises:
        TimeoutError: If task exceeds timeout
        ValueError: If task is invalid
    """
    pass
```

### Code Organization

- **Core functionality** → `src/vedart/core/`
- **Executors** → `src/vedart/executors/`
- **Utilities** → `src/vedart/utils/`
- **Tests** → `tests/unit/`, `tests/integration/`, `tests/stress/`
- **Examples** → `examples/`
- **Documentation** → `docs/`

---

## Testing

### Test Categories

#### Unit Tests (`tests/unit/`)
- Fast, isolated tests
- Mock external dependencies
- Test single components

```bash
pytest tests/unit -v
```

#### Integration Tests (`tests/integration/`)
- Test component interactions
- Use real dependencies
- Test end-to-end workflows

```bash
pytest tests/integration -v
```

#### Stress Tests (`tests/stress/`)
- Test under high load
- Long-running tests
- Resource limits

```bash
pytest tests/stress -v -m slow
```

### Writing Tests

```python
def test_feature_name():
    """Test description."""
    # Arrange
    data = range(10)
    
    # Act
    result = veda.par_iter(data).map(lambda x: x * 2).collect()
    
    # Assert
    assert len(result) == 10
    assert result[0] == 0
    assert result[9] == 18
```

### Test Coverage

We aim for >90% coverage on core modules:

```bash
pytest --cov=src/vedart --cov-report=html
# Open htmlcov/index.html to view coverage
```

---

## Code Quality

### Before Committing

Run these checks:

```bash
# 1. Format code
black src/vedart tests examples

# 2. Lint
ruff check src/vedart tests

# 3. Type check
mypy src/vedart --strict

# 4. Run tests
pytest tests/unit tests/integration -v
```

### Pre-commit Hook

Consider setting up a pre-commit hook:

```bash
# .git/hooks/pre-commit
#!/bin/bash
black src/vedart tests examples
ruff check src/vedart tests
mypy src/vedart --strict
pytest tests/unit -v
```

---

## Documentation

### Docstring Style

Use Google-style docstrings:

```python
def function(arg1: int, arg2: str = "default") -> bool:
    """Short description.
    
    Longer description with more details about what the
    function does and how to use it.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2, defaults to "default"
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg1 is negative
        
    Example:
        >>> function(42, "test")
        True
    """
    pass
```

### Documentation Files

- **README.md** - Project overview
- **docs/architecture.md** - System design
- **docs/guarantees.md** - Behavioral guarantees
- **docs/ROADMAP.md** - Development roadmap
- **docs/PROGRESS.md** - Implementation status

### Adding Examples

When adding features, include an example:

```python
# examples/XX_feature_name.py
"""Example demonstrating the new feature.

This example shows how to...
"""

import vedart as veda

def main():
    # Demonstration code
    pass

if __name__ == "__main__":
    main()
```

---

## Submitting Changes

### Commit Messages

Use clear, descriptive commit messages:

```
feat: add distributed executor support

- Implement Redis queue backend
- Add network-aware scheduling
- Include integration tests

Fixes #123
```

**Commit types:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation only
- `test:` - Adding/improving tests
- `refactor:` - Code restructuring
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

### Pull Request Process

1. **Create PR** with descriptive title
2. **Fill out template** completely
3. **Link related issues** using `Fixes #123`
4. **Ensure CI passes** (all checks green)
5. **Request review** from maintainers
6. **Address feedback** promptly
7. **Rebase if needed** to keep history clean

### PR Checklist

Before submitting:

- [ ] Tests added/updated
- [ ] All tests pass
- [ ] Documentation updated
- [ ] CHANGELOG.md updated (for significant changes)
- [ ] Code formatted (Black)
- [ ] Linting passes (Ruff)
- [ ] Type checking passes (MyPy)
- [ ] Examples added (if new feature)
- [ ] No new warnings

---

## Roadmap

Check the [Roadmap](docs/ROADMAP.md) for planned features and priorities.

### Current Focus Areas

**High Priority:**
- Increasing test coverage to >90%
- Fixing remaining mypy strict mode issues
- GPU CI integration

**Medium Priority:**
- Performance optimizations
- Additional examples
- Documentation improvements

**Future:**
- Distributed execution
- ML framework integration
- Timeline visualizer

---

## Development Tips

### Debugging

Use deterministic mode for reproducible debugging:

```python
with veda.deterministic(seed=42):
    result = buggy_function()
```

### Performance Testing

Run benchmarks before/after changes:

```bash
python benchmarks/compare_frameworks.py
```

### Local CI Simulation

Test like CI does:

```bash
# Run full test suite
./run_ci_tests.sh

# Or manually:
pytest tests/ -v --cov=src/vedart
ruff check src/vedart tests
black --check src/vedart tests
mypy src/vedart --strict
```

---

## Getting Help

- **Documentation:** Check `docs/` directory
- **Examples:** See `examples/` directory
- **Issues:** [GitHub Issues](https://github.com/TIVerse/vedart/issues)
- **Discussions:** [GitHub Discussions](https://github.com/TIVerse/vedart/discussions)

---

## Recognition

Contributors will be:
- Listed in `AUTHORS.md`
- Credited in release notes
- Mentioned in `CHANGELOG.md`

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to VedaRT! 🚀
