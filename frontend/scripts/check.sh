#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npx prettier --check "public/**/*.html" "css/**/*.css" "js/**/*.js" || \
npx prettier --check .
