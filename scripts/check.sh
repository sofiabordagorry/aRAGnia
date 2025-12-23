#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"

ruff check src tests
black --check src tests
mypy src/institutional_graphrag
pytest -v --cov=institutional_graphrag --cov-report=term-missing

echo "All checks passed!"
