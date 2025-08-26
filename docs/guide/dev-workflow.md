# Development Workflow

- Use the [Makefile targets](/guide/makefile) or VS Code tasks to launch API + UI
- Hot reload: both uvicorn and Flutter support reload for fast iteration
- Tests: run with `pytest` (configured to discover in both `tests/` and project root)

## Repo layout
- `pmm/`: core library and API (`pmm/api/probe.py`)
- `ui/`: Flutter client
- `tests/`: centralized tests (existing root tests still run)
- `docs/`: this VitePress site
