# Ejecucion en ClusterUY

Guia completa para correr el pipeline multi-modelo de GraphRAG en [cluster.uy](https://cluster.uy).

Esta guia documenta **los pasos reales** que funcionaron, incluyendo workarounds por incompatibilidades del cluster (CentOS 7, GLIBC 2.17, GCC 4.8.5).

Referencia oficial: [Como ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)

---

## Indice

1. [Prerequisitos](#prerequisitos)
2. [Paso 1: Conectarse al cluster](#paso-1-conectarse-al-cluster)
3. [Paso 2: Clonar el repositorio (via PAT)](#paso-2-clonar-el-repositorio-via-pat)
4. [Paso 3: Subir datos](#paso-3-subir-datos)
5. [Paso 4: Instalar Miniconda](#paso-4-instalar-miniconda)
6. [Paso 5: Crear entorno e instalar dependencias (manual)](#paso-5-crear-entorno-e-instalar-dependencias-manual)
7. [Paso 6: Configurar acceso a HuggingFace (para Llama)](#paso-6-configurar-acceso-a-huggingface-para-llama)
8. [Paso 7: Enviar el job](#paso-7-enviar-el-job)
9. [Paso 8: Monitorear / relanzar](#paso-8-monitorear--relanzar)
10. [Evaluación de generación](#evaluacion-de-generacion)
11. [Evaluación de recuperación (Neo4j con Singularity)](#evaluacion-de-recuperacion-neo4j-con-singularity)
12. [Cambios hechos al codigo](#cambios-hechos-al-codigo)
13. [Troubleshooting: errores encontrados y soluciones](#troubleshooting-errores-encontrados-y-soluciones)

---

## Prerequisitos

- Cuenta en ClusterUY con clave SSH configurada ([Como conectarse](https://cluster.uy/ayuda/como_conectarse/))
- Personal Access Token de GitHub (el cluster no tiene la SSH key configurada por default)
- Cuenta de HuggingFace con licencia de Llama-3.1 aceptada (ver [Paso 6](#paso-6-configurar-acceso-a-huggingface-para-llama))

---

## Paso 1: Conectarse al cluster

```bash
ssh -p 10022 tu_usuario@cluster.uy
```

---

## Paso 2: Clonar el repositorio (via PAT)

El cluster no tiene SSH key configurada para GitHub, hay que usar un Personal Access Token.

**En GitHub**: Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token → permisos `repo`.

**En el cluster**:
```bash
cd ~
git clone https://github.com/sofiabordagorry/institutional-graphrag.git
# Cuando pida password, pegar el PAT (no la contrasena de GitHub)
cd institutional-graphrag
```

Para no tener que ingresarlo cada vez:
```bash
git config --global credential.helper 'store --file ~/.git-credentials'
```

---

## Paso 3: Subir datos

Los PDFs del corpus y los archivos de tablas deben estar en `data/corpus/` y `data/tables/`. Desde tu maquina local:

```bash
rsync -arvz -e "ssh -p 10022" data/corpus/ tu_usuario@cluster.uy:~/institutional-graphrag/data/corpus/
rsync -arvz -e "ssh -p 10022" data/tables/ tu_usuario@cluster.uy:~/institutional-graphrag/data/tables/
scp -P 10022 backend/.env tu_usuario@cluster.uy:~/institutional-graphrag/backend/.env
```

En windows:

```bash
scp -P 10022 -r data/corpus/* tu_usuario@cluster.uy:~/institutional-graphrag/data/corpus/
```

Tambien crear la carpeta `logs/` que SLURM necesita para escribir los logs:
```bash
# En el cluster:
mkdir -p ~/institutional-graphrag/logs
```

---

## Paso 4: Instalar Miniconda

> **IMPORTANTE**: No usar `Miniconda3-latest-Linux-x86_64.sh`. El instalador "latest" requiere GLIBC >= 2.28, pero el cluster tiene **GLIBC 2.17** (CentOS 7). Hay que usar una version vieja.

Pedir un nodo interactivo (no instalar en el nodo login):

```bash
srun -p normal -c 1 --time=02:00:00 --ntasks=1 --mem=4G --pty bash -l
```

En el nodo interactivo:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
bash Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash
source ~/.bashrc
```

---

## Paso 5: Crear entorno e instalar dependencias (manual)

> **IMPORTANTE**: `conda env create -f environment.yml` **no funciona** en este cluster. El solver de conda se queda colgado horas sin dar error (con 4 canales y muchas restricciones). Hay que instalar paquete por paquete con pip, forzando wheels binarios (`--only-binary :all:`) porque el GCC 4.8.5 del sistema no puede compilar numpy/pandas/scikit-learn modernos.

Desde el nodo interactivo (con conda ya instalado):

```bash
# 1. Crear entorno vacio con Python 3.11
conda create -n graphrag python=3.11 -y
conda activate graphrag

# 2. Instalar libstdc++ de conda (el del sistema es muy viejo, le falta CXXABI_1.3.9)
conda install -c conda-forge libstdcxx-ng -y

# 3. PyTorch 2.6.0 con CUDA 11.8
# torch >=2.7 requiere GLIBC 2.28; torch <=2.6 cu118 usa manylinux2014 (GLIBC 2.17 OK)
# torch 2.6.0 es necesario para soportar transformers 5.x (requerido por Qwen3.5-9B)
pip install --only-binary :all: \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 4. Numpy <2 (compatible con torch 2.6)
pip install --only-binary :all: "numpy<2"

# 5. Data stack (forzar wheels binarios; GCC 4.8 no compila estos)
pip install --only-binary :all: \
    pandas \
    scikit-learn \
    pyarrow \
    ijson \
    openpyxl

# 6. opencv <4.12 (opencv 4.13+ requiere numpy 2)
pip install --only-binary :all: "opencv-python-headless<4.12"

# 7. odfpy (pure Python, no hay wheel binario publicado pero no necesita compilar)
pip install --no-build-isolation odfpy

# 8. Docling + extras de chunking
pip install "docling>=2.0.0" "docling-core[chunking]" langchain-docling langchain-core langchain-text-splitters

# 9. Transformers y embeddings
# Instalar la version mas nueva disponible. transformers>=5 es necesario para modelos
# nuevos (Qwen3.5-9B, etc.) y funciona bien con docling a pesar del warning de pip
# sobre huggingface_hub>=1.0 "incompatible".
pip install "transformers>=5" accelerate "huggingface_hub>=1.0" hf-xet sentence-transformers

# 10. LLM client
pip install groq

# 11. Storage backends (psycopg2, neo4j, qdrant) — el paquete local los importa al arrancar
pip install --only-binary :all: psycopg2-binary neo4j qdrant-client

# 12. Utilidades
pip install python-dotenv requests

# 13. Instalar el paquete local en editable sin deps
pip install -e ~/institutional-graphrag/backend --no-deps
```

Verificar que todo importa:

```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
python -c "import docling; from docling.chunking import HybridChunker; print('docling OK')"
python -c "import aragnia; print('paquete OK')"
```

Salir del nodo interactivo:

```bash
exit
```

---

## Paso 6: Configurar acceso a HuggingFace (para Llama)

`meta-llama/Llama-3.1-8B-Instruct` es un modelo **gated** (gratis, pero hay que aceptar la licencia y autenticarse).

**En tu navegador**:
1. Crear cuenta en https://huggingface.co
2. Ir a https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct y click en "Agree and access repository"
3. Llenar el formulario (affiliation: "Universidad de la Republica - FING" u otro)
4. Esperar la aprobacion (suele ser rapido, minutos a horas)
5. Crear un token en https://huggingface.co/settings/tokens (permiso: `Read`)

**En el cluster**:
```bash
# Agregar el token al bashrc para que este siempre disponible
echo 'export HF_TOKEN="hf_xxxxxxxxxxxx"' >> ~/.bashrc
echo 'export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"' >> ~/.bashrc
source ~/.bashrc
```

Verificar:
```bash
python -c "from huggingface_hub import whoami; print(whoami())"
```

---

## Paso 7: Enviar el job

Desde la raiz del repositorio en el cluster:

```bash
cd ~/institutional-graphrag
sbatch backend/scripts/cluster/submit.sh
```

El job usa `--partition=besteffort --qos=besteffort_gpu --gres=gpu:a40:1` (ver [submit.sh](submit.sh)).

**Para recibir notificaciones por email** cuando el job termina (exitoso o fallido):
1. En [submit.sh](submit.sh), descomenta la línea `#SBATCH --mail-user=tu_email@fing.edu.uy` y reemplaza con tu email
2. Vuelve a ejecutar `sbatch backend/scripts/cluster/submit.sh`
3. Recibirás notificaciones al inicio, término y si el job falla

### Que hace el job

1. Copia `data/` del home (NFS lento) a `/scratch/$USER/graphrag_data/` (SSD local, 300 GB)
2. Activa el entorno conda `graphrag`
3. Fuerza `LD_LIBRARY_PATH` al `lib/` del entorno conda (para usar el libstdc++ nuevo)
4. Corre `multi_model_pipeline.py` que itera por los 3 modelos definidos en `LLM_MODELS`
5. Al terminar (exito o fallo), un `trap EXIT` copia los resultados de scratch de vuelta al home

---

## Paso 8: Monitorear / relanzar

```bash
# Ver jobs en cola
squeue -u $USER

# Seguir el log en vivo
tail -f ~/institutional-graphrag/logs/pipeline_<JOBID>.log

# Cancelar
scancel <JOBID>
```

**Si el job muere o es preemptado**, simplemente relanzar:

```bash
sbatch backend/scripts/cluster/submit.sh
```

El pipeline tiene **checkpointing** a varios niveles:
- **Docling**: cada PDF ya parseado tiene su JSON en `data/docling/` — no se reprocesa.
- **Chunks**: cada documento ya chunkeado tiene su `_chunks.json` — no se reprocesa.
- **Extraccion LLM**: el `llm_registry.json` por modelo trackea que documentos ya fueron procesados.

Solo usar `--skip-docling --skip-chunks` si querés ahorrar tiempo en una segunda corrida donde el docling ya esta hecho.

### Modelos parciales

Si un modelo falla (ej. Llama sin HF_TOKEN), el pipeline **continua con los siguientes modelos** (hay un try/except envolviendo el loop, ver [Cambios hechos al codigo](#cambios-hechos-al-codigo)). Al final imprime un resumen de cuales fallaron.

Los resultados parciales (ej. del primer Qwen) quedan guardados en `data/results/Qwen_Qwen2.5-7B-Instruct/entity_documents.json` aunque el siguiente modelo falle.

---

## Evaluacion de generacion

Job aparte para evaluar **qué LLM / prompt genera mejores respuestas** en lenguaje natural
sobre el GT de validación, y **registrar los tiempos de espera**. Usa
[`submit_generation.sh`](submit_generation.sh) → corre `evaluation/scripts/evaluate_generation.py`.

A diferencia de `submit.sh`, este job **no necesita el corpus ni scratch**: la entrada es
`evaluation/ground_truth/datasetQA_GT.json` (ya en el repo). Itera internamente por los
modelos definidos en la constante `MODELS` del script (no hace falta un pipeline aparte).

### Diferencias de dependencias

El entorno necesita dos paquetes extra respecto al de extracción (ya agregados al
`environment.yml`; en el cluster se instalan con pip por el flujo manual del [Paso 5](#paso-5-crear-entorno-e-instalar-dependencias-manual)):

```bash
conda activate graphrag
pip install matplotlib            # gráficas de los reportes
pip install bitsandbytes          # opcional: cuantización 4-bit (HF_QUANTIZATION=bnb4)
```

### El juez (API de Anthropic)

La comparación contra el `answer` de referencia la hace un LLM-as-a-judge vía la API de
Anthropic, así que el nodo de cómputo necesita:

1. `ANTHROPIC_API_KEY` en `backend/.env` (o exportada).
2. **Salida a internet** desde el nodo de cómputo (igual que para descargar modelos de HF).
   Si el nodo no tiene internet, correr con `--no-judge` (solo genera y mide latencias) y
   ejecutar el juicio después donde haya conexión, reusando `generation_details.json`.

### Enviar el job

```bash
cd ~/institutional-graphrag
sbatch backend/scripts/cluster/submit_generation.sh
```

Para una prueba rápida, editar la línea final de `submit_generation.sh` agregando
`--max-questions 3` (y `--no-judge` si todavía no configuraste la API key).

### Salidas

Quedan en `evaluation/results/generation/`:

- `generation_comparison_summary.json` — métricas agregadas por modelo×prompt
- `generation_details.json` — detalle por pregunta (respuesta de cada modelo + juicio)
- `images/*.png` y `generation_report.html` — gráficas y reporte

### Modelos grandes en la A40 (48GB)

Los modelos por defecto van de 4B a 24B. En bf16, un 24B (Mistral Small) ronda los 48GB y
**puede no entrar** con el overhead del KV cache. Si da OOM, descomentar
`export HF_QUANTIZATION="bnb4"` en `submit_generation.sh`. Si un modelo falla (gated sin
token, OOM, etc.) el script **sigue con los demás** y al final lista los que fallaron.

---

## Evaluacion de recuperacion (Neo4j con Singularity)

Para evaluar la etapa de *Retrieval* (qué tan bien el sistema genera queries Cypher y recupera la información relevante), el script necesita consultar una base de datos Neo4j viva. Como los nodos de ClusterUY no pueden acceder a tu entorno local (Docker Desktop), usamos **Singularity** (Apptainer) para levantar una base de datos efímera dentro del mismo nodo de cómputo asignado por SLURM.

Esta arquitectura garantiza latencia cero, ya que la base de datos, el LLM y el evaluador comparten la misma RAM y CPU/GPU durante el job, destruyéndose de forma segura al finalizar.

### 1. Descargar la imagen de Neo4j en el clúster

Singularity permite importar contenedores de Docker Hub directamente. Debes hacer esto **en un nodo interactivo**, no en el nodo de login.

```bash
# 1. Pedir nodo interactivo
srun -p normal -c 1 --time=00:30:00 --ntasks=1 --mem=4G --pty bash -l

# 2. Descargar la imagen estable de Neo4j v5, de qdrant y de ollama
cd ~/institutional-graphrag
singularity pull --name neo4j.simg docker://neo4j:5

singularity pull --name qdrant.simg docker://qdrant/qdrant:latest

singularity pull --name ollama.simg docker://ollama/ollama:latest

# 3. Salir del nodo
exit
```
### 2. Poblar el grafo con el ground truth y evaluar el retrieval

`submit_retrieval.sh` le pasa el `ground_truth_kg.json` a `load_graph.py` como argumento, no hay que editar nada.
```bash
cd ~/institutional-graphrag

conda activate graphrag

sbatch backend/scripts/cluster/submit_retrieval.sh
```

```bash
# Para ver los logs
# log de evaluación del retrieval:
cat ~/institutional-graphrag/logs/ret_eval_5546514.log
# log de neo4j:
cat ~/institutional-graphrag/logs/neo4j_db.log
# log de qdrant:
cat ~/institutional-graphrag/logs/qdrant_db.log
# log de ollama:
cat ~/institutional-graphrag/logs/ollama_db.log
```


## Cambios hechos al codigo

Durante el setup hubo que hacer los siguientes cambios al repo (ya commiteados en la rama `evaluation/cluster-scripts`):

### `backend/scripts/cluster/submit.sh`

1. **`REPO_ROOT` via `SLURM_SUBMIT_DIR`**: SLURM copia el script a `/var/spool/` antes de ejecutarlo, asi que `BASH_SOURCE` apunta a la copia, no al repo. Hay que usar `SLURM_SUBMIT_DIR` (el directorio desde donde se hizo `sbatch`):
   ```bash
   REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
   ```

2. **Activacion de conda hardcoded**: SLURM no hereda el shell init del usuario, asi que `conda` no esta en el PATH. Hay que hacer `source` directo del `conda.sh`:
   ```bash
   CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
   source "$CONDA_BASE/etc/profile.d/conda.sh"
   conda activate "$ENV_NAME"
   ```

3. **Forzar `LD_LIBRARY_PATH` al libstdc++ del conda env**: el sistema tiene GCC 4.8.5 (libstdc++ viejo, CXXABI <=1.3.8), pero los wheels de pip (numpy/torch) requieren CXXABI 1.3.9+. Anteponer el `lib/` del entorno conda:
   ```bash
   export LD_LIBRARY_PATH="$CONDA_BASE/envs/$ENV_NAME/lib:${LD_LIBRARY_PATH:-}"
   ```

4. **Llamar a `multi_model_pipeline.py`** en vez de `pipeline.py` (corre los 3 modelos en secuencia).

### `backend/scripts/cluster/multi_model_pipeline.py`

1. **Helper `_ensure_symlink`**: el `cleanup` trap en `submit.sh` copia todo de vuelta al home, incluyendo los symlinks que apuntan a `/scratch` — cuando el job siguiente arranca, esos symlinks estan rotos. `Path.exists()` sigue symlinks y devuelve `False` para los rotos, asi que el codigo viejo tiraba `FileExistsError` al intentar crearlos. Fix:
   ```python
   def _ensure_symlink(link: Path, target: Path) -> None:
       if link.is_symlink():
           link.unlink()
       if not link.exists():
           link.symlink_to(target)
   ```

2. **Lista `LLM_MODELS` actualizada**: `Qwen/Qwen3.5-9B` no existe en HuggingFace. Reemplazado por un modelo valido (ej. `Qwen/Qwen2.5-14B-Instruct` que entra en los 48 GB de la A40).

3. **Try/except alrededor del loop de modelos**: para que si un modelo falla (ej. Llama sin HF_TOKEN, modelo inexistente) los otros igual corran. Al final imprime un resumen de fallados.

---

## Troubleshooting: errores encontrados y soluciones

### Conda / instalacion

| Error | Causa | Solucion |
|---|---|---|
| `Miniconda installer requires GLIBC >= 2.28` | CentOS 7 tiene GLIBC 2.17 | Usar `Miniconda3-py310_23.3.1-0-Linux-x86_64.sh` (version vieja) |
| `conda env create` se queda colgado horas sin error | Solver de conda con 4 canales + muchas restricciones | No usar `environment.yml` en este cluster; instalar con pip manualmente |
| `Killed` durante `conda env create` (OOM) | Solver consume >32 GB RAM | Irrelevante (el solver se cuelga igual); saltar al flujo manual |

### Dependencias de Python

| Error | Causa | Solucion |
|---|---|---|
| `NumPy requires GCC >= 9.3` al compilar | GCC del sistema es 4.8.5 | `pip install --only-binary :all: numpy` (forzar wheel) |
| `pyarrow` compile error | Idem | `pip install --only-binary :all: pyarrow` |
| `ERROR: Could not find a version that satisfies the requirement odfpy` con `--only-binary :all:` | odfpy no publica wheel | `pip install --no-build-isolation odfpy` (es pure Python) |
| `libcurand.so.10: requires GLIBC_2.27` | torch >=2.5 requiere GLIBC nueva | Usar `torch==2.2.2 --index-url https://download.pytorch.org/whl/cu118` |
| `CXXABI_1.3.9 not found in libstdc++.so.6` | libstdc++ del sistema muy viejo | `conda install -c conda-forge libstdcxx-ng` + exportar `LD_LIBRARY_PATH=$CONDA_PREFIX/lib:...` |
| `_ARRAY_API not found` (al importar torch) | numpy 2.0 incompatible con torch 2.2 | `pip install "numpy<2"` |
| `opencv-python-headless requires numpy>=2` | opencv 4.13+ forzo numpy 2 | `pip install "opencv-python-headless<4.12"` |
| `ImportError: cannot import name 'RTDetrImageProcessor'` | transformers <4.42 (docling-ibm-models lo necesita) | `pip install "transformers>=4.45"` |
| `ModuleNotFoundError: docling_core.transforms.chunker.hybrid_chunker` | Falta extra `[chunking]` | `pip install "docling-core[chunking]"` |
| `ModuleNotFoundError: psycopg2` | El `__init__.py` del paquete importa `storage.database` al arrancar | `pip install --only-binary :all: psycopg2-binary neo4j qdrant-client` |

### SLURM

| Error | Causa | Solucion |
|---|---|---|
| `/var/spool/logs/pipeline_XXX.log: Permission denied` | `#SBATCH --output=logs/...` es relativo a donde arranca el script (`/var/spool/...`) | Crear `~/institutional-graphrag/logs/` antes de `sbatch` (SLURM lo resuelve relativo a `SLURM_SUBMIT_DIR`) |
| `conda: command not found` en el job | SLURM no hereda shell init | Hacer `source $HOME/miniconda3/etc/profile.d/conda.sh` explicitamente |
| `REPO_ROOT=/var/spool` | SLURM copia el script a `/var/spool/` | Usar `$SLURM_SUBMIT_DIR` en vez de `BASH_SOURCE` |
| `FileExistsError` en symlinks al relanzar | Cleanup trap copio symlinks rotos al home | Helper `_ensure_symlink` con `is_symlink()` check |

### HuggingFace / modelos

| Error | Causa | Solucion |
|---|---|---|
| `meta-llama/Llama-3.1-8B-Instruct: Repository not found (or gated)` | Modelo gated, requiere licencia aceptada + token | Ver [Paso 6](#paso-6-configurar-acceso-a-huggingface-para-llama) |
| `ValueError: model type 'xyz' not recognized by Transformers` | El modelo es demasiado nuevo para la version de transformers instalada | Upgradear: `pip install --upgrade transformers huggingface_hub`. Pip puede avisar incompatibilidad con docling pero docling sigue funcionando en runtime. |
| Warning `MultiScaleDeformableAttention CUDA kernel compile failed` | GCC 4.8 no tiene C++17 para compilar el kernel de transformers | **No es fatal**; transformers usa fallback en Python (mas lento pero anda) |

---

## Links utiles

- [Como ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)
- [Como conectarse](https://cluster.uy/ayuda/como_conectarse/)
- [Ambientes con miniconda](https://www.cluster.uy/ayuda/ambiente/)
- [Recursos disponibles](https://www.cluster.uy/ayuda/recursos_disponibles/)
- [Consejos y buenas practicas](https://www.cluster.uy/ayuda/tips/)
- [PyTorch previous versions (cu118)](https://pytorch.org/get-started/previous-versions/)
