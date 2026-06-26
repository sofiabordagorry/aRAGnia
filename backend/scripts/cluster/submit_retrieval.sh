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
TEMP_QDRANT="$REPO_ROOT/qdrant_temp"
OLLAMA_DATA="$REPO_ROOT/ollama_data"

rm -rf "$TEMP_NEO4J" "$TEMP_QDRANT"
mkdir -p "$TEMP_NEO4J/data" "$TEMP_NEO4J/logs" "$TEMP_NEO4J/run"
mkdir -p "$TEMP_QDRANT/storage"
mkdir -p "$OLLAMA_DATA"

# --- Iniciar Neo4j ---
echo "[INFO] Extrayendo configuración por defecto de Neo4j..."
singularity exec "$REPO_ROOT/neo4j.simg" cp -r /var/lib/neo4j/conf "$TEMP_NEO4J/conf"
chmod -R 777 "$TEMP_NEO4J/conf"

unset NEO4J_USER NEO4J_PASSWORD NEO4J_HTTP_PORT NEO4J_BOLT_PORT

export SINGULARITYENV_NEO4J_AUTH="neo4j/password"
export SINGULARITYENV_NEO4J_ACCEPT_LICENSE_AGREEMENT="yes"

echo "[INFO] Levantando Neo4j vacío en Singularity..."
singularity run --cleanenv \
    --bind "$TEMP_NEO4J/data:/data" \
    --bind "$TEMP_NEO4J/logs:/logs" \
    --bind "$TEMP_NEO4J/run:/var/lib/neo4j/run" \
    --bind "$TEMP_NEO4J/conf:/var/lib/neo4j/conf" \
    "$REPO_ROOT/neo4j.simg" > "$REPO_ROOT/logs/neo4j_db.log" 2>&1 &
NEO4J_PID=$!

# --- Iniciar Qdrant ---
echo "[INFO] Levantando Qdrant..."
singularity run --cleanenv --writable-tmpfs --pwd /qdrant \
    --bind "$TEMP_QDRANT/storage:/qdrant/storage" \
    "$REPO_ROOT/qdrant.simg" > "$REPO_ROOT/logs/qdrant_db.log" 2>&1 &
QDRANT_PID=$!

# --- Iniciar Ollama ---
PUERTO_OLLAMA="${OLLAMA_PORT:-11434}"
echo "[INFO] Levantando Ollama en el puerto $PUERTO_OLLAMA (con soporte GPU)..."
export SINGULARITYENV_OLLAMA_HOST="127.0.0.1:$PUERTO_OLLAMA"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="0"

singularity run --cleanenv --nv \
    --bind "$OLLAMA_DATA:/root/.ollama" \
    --bind /usr/lib64/nvidia:/usr/lib/nvidia \
    "$REPO_ROOT/ollama.simg" > "$REPO_ROOT/logs/ollama_db.log" 2>&1 &
OLLAMA_PID=$!

echo "[INFO] Esperando 60 segundos a que inicien las bases de datos..."
sleep 60

export HOST="localhost"
export NEO4J_BOLT_PORT="7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"
export QDRANT_PORT="6333"
export OLLAMA_HOST="127.0.0.1:$PUERTO_OLLAMA"

echo "============================================================"
echo "[INFO] Descargando/Verificando modelo del juez local (mistral)..."
singularity exec "$REPO_ROOT/ollama.simg" ollama pull mistral

echo "============================================================"
echo "[INFO] Poblando el grafo con ground_truth_kg.json..."
python "$REPO_ROOT/backend/scripts/load_graph.py"

echo "[INFO] Poblando Qdrant (Few-Shot Store)..."
python "$REPO_ROOT/backend/scripts/load_fewshot_examples.py"

echo "============================================================"
echo "[INFO] Iniciando evaluación de recuperación (Cypher -> Neo4j)..."

python -u "$REPO_ROOT/evaluation/scripts/evaluate_retrieval.py" \
    --max-questions 1 \
    --local-judge mistral

echo ""
echo "============================================================"
echo "  Job finalizado: $(date)"
echo "  Resultados en:  $REPO_ROOT/evaluation/results/retrieval/"
echo "  Apagando contenedores..."
kill $NEO4J_PID
kill $QDRANT_PID
kill $OLLAMA_PID
rm -rf "$TEMP_NEO4J" "$TEMP_QDRANT"
echo "============================================================"