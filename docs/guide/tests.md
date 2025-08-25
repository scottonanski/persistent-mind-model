# Tests

Pytest is configured to discover tests in both the project root and `tests/`.

- Config: `pytest.ini`
- Root tests: `test_*.py` files in the repo root
- Folder tests: `tests/` suite with `conftest.py`

Run:
- `pytest` (fast)
- `pytest -q` (quieter)

Next step (optional): consolidate root tests under `tests/` once green.
