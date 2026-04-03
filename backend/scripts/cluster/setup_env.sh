#!/bin/bash
# =============================================================================
# setup_env.sh
# Script de configuración inicial del entorno conda en el cluster.
# Ejecutar UNA SOLA VEZ antes de usar submit.sh.
#
# Uso:
#   bash cluster/setup_env.sh
#
# Prerrequisito: tener conda/miniconda instalado.
# Ver: https://cluster.uy/ayuda/conda/
# =============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_ROOT/cluster/environment.yml"
ENV_NAME="graphrag"

echo "============================================================"
echo "  Setup entorno conda: $ENV_NAME"
echo "  Repo: $REPO_ROOT"
echo "============================================================"

# 1. Crear (o actualizar) el entorno conda
if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] El entorno '$ENV_NAME' ya existe. Actualizando..."
    conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
    echo "[INFO] Creando entorno '$ENV_NAME'..."
    conda env create -f "$ENV_FILE"
fi

# 2. Instalar el paquete local en modo editable
echo ""
echo "[INFO] Instalando paquete local (institutional-graphrag)..."
conda run -n "$ENV_NAME" pip install -e "$REPO_ROOT/backend" --no-deps

echo ""
echo "============================================================"
echo "  Setup completo."
echo ""
echo "  Para activar manualmente:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Para enviar el job al cluster:"
echo "    sbatch cluster/submit.sh"
echo "============================================================"
