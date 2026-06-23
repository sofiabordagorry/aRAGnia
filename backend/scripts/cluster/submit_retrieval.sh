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

TEMP_NEO4J="$REPO_ROOT/neo4j_temp"

rm -rf "$TEMP_NEO4J"
mkdir -p "$TEMP_NEO4J/data" "$TEMP_NEO4J/logs" "$TEMP_NEO4J/run"

echo "[INFO] Extrayendo configuración por defecto de la imagen..."
singularity exec "$REPO_ROOT/neo4j.simg" cp -r /var/lib/neo4j/conf "$TEMP_NEO4J/conf"
chmod -R 777 "$TEMP_NEO4J/conf"

export SINGULARITYENV_NEO4J_AUTH="neo4j/password"
export SINGULARITYENV_NEO4J_ACCEPT_LICENSE_AGREEMENT="yes"

echo "[INFO] Levantando Neo4j vacío en Singularity..."
singularity run \
    --bind "$TEMP_NEO4J/data:/data" \
    --bind "$TEMP_NEO4J/logs:/logs" \
    --bind "$TEMP_NEO4J/run:/var/lib/neo4j/run" \
    --bind "$TEMP_NEO4J/conf:/var/lib/neo4j/conf" \
    "$REPO_ROOT/neo4j.simg" > "$REPO_ROOT/logs/neo4j_db.log" 2>&1 &
NEO4J_PID=$!

echo "[INFO] Esperando 60 segundos a que el motor Neo4j inicie..."
sleep 60

export HOST="localhost"
export NEO4J_BOLT_PORT="7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

echo "============================================================"
echo "[INFO] Poblando el grafo con ground_truth_kg.json..."
python "$REPO_ROOT/backend/scripts/load_graph.py"

echo "============================================================"
echo "[INFO] Iniciando evaluación de recuperación (Cypher -> Neo4j)..."

python "$REPO_ROOT/evaluation/scripts/evaluate_retrieval.py" \
    --max-questions 1 \
    --no-judge
    # --local-judge llama3.1

echo ""
echo "============================================================"
echo "  Job finalizado: $(date)"
echo "  Resultados en:  $REPO_ROOT/evaluation/results/retrieval/"
echo "  Apagando base de datos Neo4j efímera (PID: $NEO4J_PID)..."
kill $NEO4J_PID
rm -rf "$SCRATCH_NEO4J"
echo "============================================================"