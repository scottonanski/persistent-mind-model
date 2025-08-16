# Contributing to Persistent Mind Model (PMM)

Thank you for your interest in contributing. This document keeps contributions practical and focused on improving the software.

## How to contribute

### Ways to contribute
- Report bugs with clear reproduction steps and environment details
- Propose focused features with concrete use cases
- Improve documentation and examples
- Submit code changes that address a specific issue or enhancement

Before starting large changes, open an issue to discuss scope and approach.

## Development setup

Use Python 3.10+. Create a virtual environment and install dependencies:

## 🛠️ Development Setup

```bash
git clone https://github.com/scottonanski/persistent-mind-model.git
cd persistent-mind-model
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run tests and linters locally before opening a PR:

```bash
pytest -q
ruff check
black --check .
```

If formatting fails, run:

```bash
black .
```

## Pull requests

- Keep PRs small and scoped to one change when possible
- Include tests for new behavior and bug fixes
- Update documentation when behavior or APIs change
- Ensure CI passes (tests, lint, formatting)
- Write clear commit messages (imperative mood). Reference issues where relevant (e.g., "Fixes #123").

## Code style and quality

- Python 3.10+
- Formatting: Black
- Linting: Ruff
- Type hints where practical; prefer clear names and small functions

## Security and data

- Do not commit secrets or API keys
- Avoid including sensitive data in tests or fixtures
- Report security issues privately via email (see README)

---
**Questions?** Open an issue or start a discussion. Thank you for helping improve PMM.
