#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npx prettier --write "public/**/*.html" "css/**/*.css" "js/**/*.js" || \
npx prettier --write .
