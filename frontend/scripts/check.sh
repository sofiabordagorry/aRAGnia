#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

npx prettier --check \
  "public/**/*.html" \
  "public/css/**/*.css" \
  "public/js/**/*.js"