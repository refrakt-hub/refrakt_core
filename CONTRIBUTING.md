# Contributing to refrakt_core

Thank you for your interest in contributing to **refrakt_core**! We welcome contributions of all kinds, including bug fixes, new features, tests, and documentation improvements.

## Development Setup

To set up your development environment with all necessary tools, run:

```bash
pip install -e .[dev]
```

Or, use the provided helper script (recommended):

```bash
python scripts/dev_setup.py
```

This will:
- Install all runtime and development dependencies (testing, linting, formatting, type checking, code complexity analysis, etc.)
- Set up pre-commit hooks for code quality

### What gets installed with `[dev]`?
The `[dev]` extra includes:
- `pytest`, `pytest-cov` (testing & coverage)
- `black` (code formatting)
- `ruff` (linting)
- `isort` (import sorting)
- `mypy` (type checking)
- `pre-commit` (git hooks)
- `radon`, `lizard` (code complexity analysis)
- Type stubs for common libraries (see `pyproject.toml` for full list)

## Code Style & Quality

- **Formatting:** Run `black .` to auto-format code.
- **Import Sorting:** Run `isort .` to sort imports.
- **Linting:** Run `ruff check .` for fast linting.
- **Type Checking:** Run `mypy src/` to check types.
- **Complexity Analysis:**
  - Run `radon cc -s -a src/` to check cyclomatic complexity (aim for class A or B; C only if absolutely necessary).
  - Run `lizard src/` for additional complexity metrics.

Pre-commit hooks will automatically check formatting and linting before each commit.

## Testing

- Run all tests with:
  ```bash
  pytest
  ```
- To see coverage, use:
  ```bash
  pytest --cov=src/refrakt_core
  ```

## Making a Contribution

1. **Fork** the repository and create your branch from `main` or `staging`.
2. **Write clear, well-documented code** and add tests for new features or bug fixes.
3. **Ensure your code passes all pre-commit checks, tests, and complexity analysis**:
   - Run `pre-commit run --all-files`
   - Run `pytest`
   - Run `radon cc -s -a src/` and ensure your code is rated class A or B (C only if absolutely necessary, with justification).
   - Run `lizard src/` for further complexity checks.
4. **Open a pull request** with a clear description of your changes.
5. Be responsive to code review feedback and update your PR as needed.

## Additional Notes

- Please keep your pull requests focused and atomic (one feature or fix per PR).
- If you are adding a new dependency, please justify its necessity.
- For larger changes, consider opening an issue or discussion first.

---

Thank you for helping make **refrakt_core** better!
