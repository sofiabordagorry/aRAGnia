cd /d %~dp0\..

npx prettier --write ^
  "public/**/*.html" ^
  "public/css/**/*.css" ^
  "public/js/**/*.js"