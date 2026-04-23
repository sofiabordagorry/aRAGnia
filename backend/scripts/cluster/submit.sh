#!/bin/bash
# =============================================================================
# submit.sh
# Script SLURM para correr el pipeline en el cluster.uy
#
# Uso:
#   sbatch backend/scripts/cluster/submit.sh
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
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --time=3-00:00:00        # Tiempo máximo: 3 días (máximo permitido para GPU)
#SBATCH --ntasks=1               # Un proceso principal
#SBATCH --mem=32G                # RAM total
#SBATCH --cpus-per-task=8        # CPUs para Docling y embeddings
#SBATCH --gres=gpu:a40:1         # GPU A40 (48GB) — más disponibles que A100
                                 # Alternativas: gpu:a100:1 (40GB, solo 2 en el cluster)
                                 #               gpu:1 (cualquier GPU, si el modelo entra en 12GB)
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

# --- Almacenamiento de alta velocidad (scratch) ---
# El home es NFS (lento para muchos archivos). /scratch es SSD local (300 GB).
# Copiamos los datos a scratch para leer/escribir rápido durante el job.
SCRATCH_DATA="/scratch/${USER}/graphrag_data"

echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Nodo:       $SLURMD_NODENAME"
echo "  Inicio:     $(date)"
echo "  Repo:       $REPO_ROOT"
echo "  Scratch:    $SCRATCH_DATA"
echo "============================================================"

# Crear directorio de logs si no existe
mkdir -p "$REPO_ROOT/logs"

# Copiar datos del corpus a scratch (solo lo que no esté ya copiado)
mkdir -p "$SCRATCH_DATA"
echo "[INFO] Copiando datos a scratch..."
rsync -a --info=progress2 "$REPO_ROOT/data/" "$SCRATCH_DATA/"
echo "[INFO] Datos copiados a scratch"

# Función para copiar resultados de vuelta al home al terminar (o si falla)
cleanup() {
    echo "[INFO] Copiando resultados de scratch a home..."
    rsync -a "$SCRATCH_DATA/" "$REPO_ROOT/data/"
    echo "[INFO] Resultados copiados a home"
}
trap cleanup EXIT

# Activar conda
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

# Correr el pipeline multi-modelo.
# --skip-docling y --skip-chunks: las etapas previas ya están hechas en una
# corrida anterior; los archivos están en data/docling/ y data/chunks/.
# Esto también evita importar docling, que está roto en este entorno por la
# upgrade de transformers a >=5 (necesario para Qwen3.5).
python "$REPO_ROOT/backend/scripts/cluster/multi_model_pipeline.py" \
    --data-dir "$SCRATCH_DATA" \
    --env-file "$REPO_ROOT/backend/.env"

# Para desactivar LLM (más rápido, sin Groq):
#   --no-llm-researchers
#   --no-llm-topics

echo ""
echo "============================================================"
echo "  Job finalizado: $(date)"
echo "============================================================"
