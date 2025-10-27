---
name: Bug Report
about: Report a bug in VedaRT
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of the bug.

## To Reproduce
Steps to reproduce the behavior:

```python
import vedart as veda

# Your code that triggers the bug
```

## Expected Behavior
A clear description of what you expected to happen.

## Actual Behavior
What actually happened.

## Error Message
```
If applicable, paste the full error traceback here
```

## Environment
- **VedaRT Version:** (e.g., 1.0.0)
- **Python Version:** (e.g., 3.11.6)
- **Operating System:** (e.g., Ubuntu 22.04, macOS 13, Windows 11)
- **Installation Method:** (e.g., pip, source)

## Additional Context
- Does this happen with deterministic mode? `with veda.deterministic(seed=42):`
- What executor type is being used? (threads/processes/async/GPU)
- How many tasks/workers are involved?
- Any custom configuration?

```python
# If using custom config, paste here
config = veda.Config.builder()....
```

## Minimal Reproducible Example
If possible, provide a minimal example that reproduces the issue:

```python
import vedart as veda

# Minimal code that reproduces the bug
```

## Workaround
If you've found a workaround, please share it here.
