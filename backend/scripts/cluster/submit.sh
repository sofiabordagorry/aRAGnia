#!/bin/bash
# =============================================================================
# submit.sh
# Script SLURM para correr el pipeline en el cluster.uy
#
# Uso:
#   sbatch cluster/submit.sh
#
# Para monitorear:
#   squeue -u $USER
#   tail -f logs/pipeline_<JOBID>.log
#
# Para cancelar:
#   scancel <JOBID>
# =============================================================================

# --- Configuración del job ---
#SBATCH --job-name=graphrag-pipeline
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.log
#SBATCH --time=08:00:00          # Tiempo máximo (HH:MM:SS) — ajustar según corpus
#SBATCH --mem=32G                # RAM total
#SBATCH --cpus-per-task=8        # CPUs para Docling y embeddings
#SBATCH --gres=gpu:1             # GPU para embeddings con sentence-transformers
                                 # Quitar esta línea si no hay GPU disponible

# Descomentar para especificar partición/cola:
# #SBATCH --partition=gpu

# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="graphrag"

echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Nodo:       $SLURMD_NODENAME"
echo "  Inicio:     $(date)"
echo "  Repo:       $REPO_ROOT"
echo "============================================================"

# Crear directorio de logs si no existe
mkdir -p "$REPO_ROOT/logs"

# Activar conda
# En cluster.uy suele estar en ~/.conda o ~/miniconda3
# Ajustar la ruta si es necesario
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[INFO] Python: $(which python) — $(python --version)"
echo "[INFO] Entorno conda: $CONDA_DEFAULT_ENV"

# Cargar variables de entorno desde archivo .env en la raíz del repo
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
    echo "[INFO] Variables cargadas desde .env"
fi

# Backend LLM: huggingface (sin servidor externo)
export LLM_BACKEND="huggingface"

# Modelo de HuggingFace a usar (se descarga automáticamente si no está en caché)
# Opciones recomendadas (de menor a mayor RAM/VRAM):
#   Qwen/Qwen2.5-3B-Instruct      (~6 GB VRAM)  ← default, equivalente al modelo Ollama actual
#   Qwen/Qwen2.5-7B-Instruct      (~14 GB VRAM)
#   meta-llama/Llama-3.2-3B-Instruct (~6 GB VRAM, requiere aceptar licencia en HF)
export HF_MODEL="${HF_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

# Directorio de caché para modelos (evita re-descargar entre jobs)
# Por defecto HuggingFace usa ~/.cache/huggingface — conviene apuntar a un dir con espacio
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"

echo "[INFO] LLM_BACKEND: $LLM_BACKEND"
echo "[INFO] HF_MODEL:    $HF_MODEL"
echo "[INFO] HF_CACHE_DIR: $HF_CACHE_DIR"

echo ""
echo "[INFO] Iniciando pipeline..."
echo "============================================================"

# Correr el pipeline completo
python "$REPO_ROOT/cluster/pipeline.py" \
    --data-dir "$REPO_ROOT/data" \
    --env-file "$REPO_ROOT/.env"

# Para saltear etapas ya procesadas, agregar flags:
#   --skip-docling
#   --skip-chunks
#   --skip-embeddings
#   --skip-extraction

# Para desactivar LLM (más rápido, sin Groq):
#   --no-llm-researchers
#   --no-llm-topics

echo ""
echo "============================================================"
echo "  Job finalizado: $(date)"
echo "============================================================"
