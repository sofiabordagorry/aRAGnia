@echo off
:: =============================================================================
:: run_local_eval.bat
:: Script para evaluar GraphRAG localmente en Windows (CMD Puro)
:: =============================================================================

SET "REPO_ROOT=%~dp0"
CD /D "%REPO_ROOT%"

echo ============================================================
echo [INFO] Iniciando evaluacion local en Windows (CMD)...
echo [INFO] Modelo: Qwen 2.5 3B (HuggingFace)
echo ============================================================

:: 1. Activar el entorno virtual local de tu proyecto (dentro de backend)
if exist backend\.venv\Scripts\activate.bat (
    echo [INFO] Activando entorno virtual .venv de backend...
    call backend\.venv\Scripts\activate.bat
    echo [INFO] Entorno virtual activado.
)

:: 2. Agregar la carpeta src al PYTHONPATH de Windows
SET "PYTHONPATH=%REPO_ROOT%backend\src;%PYTHONPATH%"

:: 3. Ejecutar el script mostrando todo en la terminal en tiempo real
python -u "evaluation\scripts\evaluate_retrieval.py" --max-questions 1 --local-judge mistral

echo ============================================================
echo [INFO] Evaluacion finalizada.
echo ============================================================
pause