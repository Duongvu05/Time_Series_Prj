# Python Development Rules

Project-wide rules for Python development to ensure consistency and quality.

## Tooling Requirements

- **Package Management**: Always use `uv` for dependency management and running scripts (e.g., `uv run`, `uv add`).
- **Static Analysis**: Use `pyright` for type checking.
- **Testing**: Use `pytest` for running automated tests.
- **Linting & Formatting**: Use `ruff` for linting and formatting code.

## Workflow

1. Before committing code, run `ruff check .` and `ruff format .`.
2. Ensure types are correct by running `pyright`.
3. Verify changes with `pytest`.
4. Execute scripts using `uv run script.py`.
