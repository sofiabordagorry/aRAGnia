#!/usr/bin/env bash
set -euo pipefail

python -m pip install -e ".[dev]"

ruff check src tests
black --check src tests
mypy src/aragnia
pytest -v --cov=aragnia --cov-report=term-missing

echo "All checks passed!"
