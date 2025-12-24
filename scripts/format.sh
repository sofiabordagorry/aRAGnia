#!/usr/bin/env bash
set -euo pipefail

echo "Formatting with black..."
black src tests

echo "Sorting imports with ruff..."
ruff check --select I --fix src tests

echo
echo "Code formatted!"
