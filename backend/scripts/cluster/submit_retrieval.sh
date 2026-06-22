#!/bin/bash
# =============================================================================
# submit_retrieval.sh
# Script SLURM para evaluar la RECUPERACIÓN de GraphRAG en cluster.uy
# =============================================================================

# --- Configuración del job ---
#SBATCH --job-name=graphrag-ret-eval
#SBATCH --output=logs/ret_eval_%j.log
#SBATCH --error=logs/ret_eval_%j.log
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --time=1-00:00:00        
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1         
#SBATCH --export=ALL             
#SBATCH --mail-type=ALL

set -e

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/usr/local/cuda/lib64

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
ENV_NAME="graphrag"

echo "============================================================"
echo "  Job ID:     $SLURM_JOB_ID"
echo "  Nodo:       $SLURMD_NODENAME"
echo "  Inicio:     $(date)"
echo "  Repo:       $REPO_ROOT"
echo "============================================================"

mkdir -p "$REPO_ROOT/logs"

CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export LD_LIBRARY_PATH="$CONDA_BASE/envs/$ENV_NAME/lib:${LD_LIBRARY_PATH:-}"

if [ -f "$REPO_ROOT/backend/.env" ]; then
    set -a
    source "$REPO_ROOT/backend/.env"
    set +a
fi

export LLM_BACKEND="huggingface"
export HF_CACHE_DIR="${HF_CACHE_DIR:-$HOME/.cache/huggingface}"


echo "[INFO] Iniciando evaluación de recuperación (Cypher -> Neo4j)..."
echo "============================================================"

python "$REPO_ROOT/evaluation/scripts/evaluate_retrieval.py" \
    --max-questions 1 \
    --local-judge llama3.1

echo ""
echo "============================================================"
echo "  Job finalizado: $(date)"
echo "  Resultados en:  $REPO_ROOT/evaluation/results/retrieval/"
echo "============================================================"