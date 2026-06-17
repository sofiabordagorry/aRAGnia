#!/bin/bash
# =============================================================================
# submit_generation.sh
# Script SLURM para evaluar la GENERACIÓN de respuestas en cluster.uy
#
# Corre evaluation/scripts/evaluate_generation.py: genera respuestas con varios
# LLMs de HuggingFace × variantes de prompt sobre el GT de validación, las juzga
# contra el answer de referencia con LLM-as-a-judge (API de Anthropic) y registra
# tiempos de espera. Produce JSON + gráficas + reporte HTML.
#
# Uso:
#   sbatch backend/scripts/cluster/submit_generation.sh
#
# Para monitorear:
#   squeue -u $USER
#   tail -f logs/gen_eval_<JOBID>.log
#
# Para cancelar:
#   scancel <JOBID>
#
# Requisitos:
#   - ANTHROPIC_API_KEY en backend/.env (o usar --no-judge editando la línea de abajo).
#   - HF_TOKEN exportado (o en .env) para los modelos gated (Llama, Gemma).
#   - matplotlib en el entorno conda (ver environment.yml).
# =============================================================================

# --- Configuración del job ---
#SBATCH --job-name=graphrag-gen-eval
#SBATCH --output=logs/gen_eval_%j.log
#SBATCH --error=logs/gen_eval_%j.log
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --time=1-00:00:00        # 1 día (la evaluación es más corta que la extracción)
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1         # GPU A40 (48GB). Los modelos >20B en bf16 pueden no
                                 # entrar: ver HF_QUANTIZATION=bnb4 más abajo.
#SBATCH --export=ALL             # Exportar el entorno del shell al job (HF_TOKEN, etc.)
#SBATCH --mail-type=ALL
# Descomentar y poner tu email para recibir notificaciones:
# #SBATCH --mail-user=tu_email@fing.edu.uy

# =============================================================================

set -e

# CUDA (requerido por la documentación de cluster.uy para jobs GPU)
export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64

# SLURM copia el script a /var/spool/, así que BASH_SOURCE no sirve para ubicar el repo.
# Usamos SLURM_SUBMIT_DIR (directorio desde donde se hizo sbatch) como raíz del repo.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ENV_NAME="graphrag"

echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Nodo:       $SLURMD_NODENAME"
echo "  Inicio:     $(date)"
echo "  Repo:       $REPO_ROOT"
echo "============================================================"

# Crear directorio de logs si no existe
mkdir -p "$REPO_ROOT/logs"

# Activar conda.
# SLURM no hereda el shell init del usuario, así que el PATH de conda no está disponible.
# Hay que hacer source directo del conda.sh de miniconda.
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# El sistema tiene libstdc++ viejo (GCC 4.8.5, CXXABI <= 1.3.8), numpy/torch pip wheels
# requieren CXXABI_1.3.9+. Forzar el uso del libstdc++ del entorno conda.
export LD_LIBRARY_PATH="$CONDA_BASE/envs/$ENV_NAME/lib:${LD_LIBRARY_PATH:-}"

echo "[INFO] Python: $(which python) — $(python --version)"
echo "[INFO] Entorno conda: $CONDA_DEFAULT_ENV"

# Cargar variables de entorno desde backend/.env (ANTHROPIC_API_KEY, HF_TOKEN, etc.)
if [ -f "$REPO_ROOT/backend/.env" ]; then
    set -a
    source "$REPO_ROOT/backend/.env"
    set +a
    echo "[INFO] Variables cargadas desde backend/.env"
fi

# Backend de generación: huggingface (modelos locales, sin servidor externo).
export LLM_BACKEND="huggingface"

# Directorio de caché para modelos (evita re-descargar entre jobs).
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

# Cuantización 4-bit (bitsandbytes) para que los modelos grandes (p. ej. Mistral 24B,
# que en bf16 no entra en los 48GB de la A40) corran. Descomentar si hace falta:
# export HF_QUANTIZATION="bnb4"

echo "[INFO] LLM_BACKEND:    $LLM_BACKEND"
echo "[INFO] HF_CACHE_DIR:   $HF_CACHE_DIR"
echo "[INFO] HF_QUANTIZATION: ${HF_QUANTIZATION:-none}"
echo "[INFO] Juez ANTHROPIC: $([ -n "${ANTHROPIC_API_KEY:-}" ] && echo 'configurado' || echo 'FALTA (usar --no-judge)')"

echo ""
echo "[INFO] Iniciando evaluación de generación..."
echo "============================================================"

# Los modelos a comparar se definen en la constante MODELS del script.
# Para una prueba rápida agregar:  --max-questions 3
# Para no juzgar (sin API key):    --no-judge
python "$REPO_ROOT/evaluation/scripts/evaluate_generation.py" \
    --judge-model claude-opus-4-8

echo ""
echo "============================================================"
echo "  Job finalizado: $(date)"
echo "  Resultados en:  $REPO_ROOT/evaluation/results/generation/"
echo "============================================================"
