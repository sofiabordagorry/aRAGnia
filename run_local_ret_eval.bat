@echo off
chcp 65001 > nul
:: =============================================================================
:: run_local_eval.bat
:: Script para evaluar GraphRAG localmente en Windows (CMD Puro)
:: =============================================================================

SET "REPO_ROOT=%~dp0"
CD /D "%REPO_ROOT%"

:: 1. Crear carpeta de logs si no existe
IF NOT EXIST evaluation\results\retrieval\logs MKDIR evaluation\results\retrieval\logs

SET "TIMESTAMP=%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%"
SET "LOG_FILE=evaluation\results\retrieval\logs\local_eval_%TIMESTAMP%.log"

echo ============================================================
echo [INFO] Iniciando evaluacion local en Windows (CMD)...
echo [INFO] Modelo: Qwen 2.5 3B (Ollama)
echo ============================================================

:: 2. Activar el entorno virtual
if exist .venv\Scripts\activate.bat (
    echo [INFO] Activando entorno virtual .venv de institutional-graphrag...
    call .venv\Scripts\activate.bat
    echo [INFO] Entorno virtual activado.
)

:: 3. Agregar la carpeta src al PYTHONPATH de Windows
SET "PYTHONPATH=%REPO_ROOT%backend\src;%PYTHONPATH%"
SET PYTHONIOENCODING=utf-8

:: 3. Ejecutar el script mostrando todo en la terminal en tiempo real
python -u "evaluation\scripts\evaluate_retrieval.py" --max-questions 2 --local-judge qwen2.5:3b-instruct 2>&1 | powershell -Command "$input | Tee-Object -FilePath '%LOG_FILE%'"

echo ============================================================
echo [INFO] Evaluacion finalizada.
echo ============================================================
pause