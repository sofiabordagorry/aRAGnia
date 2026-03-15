@echo off
cd /d %~dp0\..

npx prettier --check ^
  "public/**/*.html" ^
  "public/css/**/*.css" ^
  "public/js/**/*.js"

pause