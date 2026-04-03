# Ejecucion en ClusterUY

Guia paso a paso para ejecutar el pipeline de GraphRAG en [cluster.uy](https://cluster.uy).

Referencia oficial: [Como ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)

## Prerequisitos

- Tener cuenta en ClusterUY con clave SSH configurada ([Como conectarse](https://cluster.uy/ayuda/como_conectarse/))
- Tener Miniconda instalado en el cluster ([Ambientes con miniconda](https://www.cluster.uy/ayuda/ambiente/))

## Paso 1: Conectarse al cluster

```bash
ssh -p 10022 tu_usuario@cluster.uy
```

## Paso 2: Clonar el repositorio

```bash
cd ~
git clone <url-del-repo> institutional-graphrag
cd institutional-graphrag
```

Si ya lo tenias clonado:

```bash
cd ~/institutional-graphrag
git pull
```

## Paso 3: Subir los datos

Los PDFs del corpus deben estar en `data/corpus/`. Desde tu maquina local:

```bash
rsync -arvz -e "ssh -p 10022" data/corpus/ tu_usuario@cluster.uy:~/institutional-graphrag/data/corpus/
```

Tambien subir el archivo `.env` si es necesario:

```bash
scp -P 10022 backend/.env tu_usuario@cluster.uy:~/institutional-graphrag/backend/.env
```

## Paso 4: Configurar el entorno conda (solo la primera vez)

Pedir un nodo interactivo para instalar dependencias (no instalar en el nodo login):

```bash
srun -p normal -c 1 --time=01:00:00 --ntasks=1 --mem=4G --pty bash -l
```

Una vez en el nodo interactivo:

```bash
cd ~/institutional-graphrag
bash backend/scripts/cluster/setup_env.sh
```

Salir del nodo interactivo:

```bash
exit
```

## Paso 5: Enviar el job

Desde la raiz del repositorio:

```bash
cd ~/institutional-graphrag
sbatch backend/scripts/cluster/submit.sh
```

Esto envia un job de tipo **besteffort con GPU** (`--partition=besteffort --qos=besteffort_gpu`), que es la estrategia recomendada por la documentacion de ClusterUY para trabajos con GPU.

### Que significa besteffort?

Los jobs besteffort tienen **menor prioridad** y pueden ser interrumpidos (preemptados) por jobs de prioridad normal. A cambio, entran en cola mas rapido. El pipeline tiene **checkpointing**: si el job es interrumpido, al relanzar retoma desde donde quedo.

## Paso 6: Monitorear el job

```bash
# Ver estado del job
squeue -u $USER

# Ver logs en tiempo real
tail -f logs/pipeline_<JOBID>.log

# Cancelar el job
scancel <JOBID>
```

## Paso 7: Si el job fue interrumpido (preemptado)

Simplemente relanzar:

```bash
sbatch backend/scripts/cluster/submit.sh
```

El pipeline retoma automaticamente:
- **Docling**: cada PDF ya parseado genera un JSON individual, no se reprocesa
- **Chunks**: cada documento ya procesado genera un archivo `_chunks.json`, se saltea
- **Extraccion LLM**: un registro (`llm_registry.json`) trackea que documentos ya fueron procesados, y se guarda un checkpoint cada 5 documentos

## Configuracion del modelo LLM

Por defecto usa `Qwen/Qwen2.5-3B-Instruct` (~6 GB VRAM). Para cambiar el modelo, editar `submit.sh` o pasar la variable al enviar:

```bash
HF_MODEL="Qwen/Qwen2.5-7B-Instruct" sbatch backend/scripts/cluster/submit.sh
```

### Modelos recomendados y GPU necesaria

| Modelo | VRAM | GPU minima |
|--------|------|------------|
| `Qwen/Qwen2.5-3B-Instruct` | ~6 GB | P100 (12GB) - `gpu:1` |
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | A40 (48GB) o A100 (40GB) |

### GPUs disponibles en ClusterUY

| GPU | VRAM | Flag SLURM |
|-----|------|------------|
| NVIDIA Tesla P100 | 12 GB | `--gres=gpu:p100:1` |
| NVIDIA A100 | 40 GB | `--gres=gpu:a100:1` (solo 2 en el cluster) |
| NVIDIA A40 | 48 GB | `--gres=gpu:a40:1` |
| Cualquiera | - | `--gres=gpu:1` |

El script actual pide una A40. Para cambiarlo, editar la linea `--gres` en `submit.sh`.

## Almacenamiento de alta velocidad (scratch)

El home directory es NFS (lento para muchos archivos). El script automaticamente:

1. **Al iniciar el job**: copia `data/` a `/scratch/<usuario>/graphrag_data/` (SSD local, 300 GB)
2. **Durante el job**: lee y escribe en scratch (mucho mas rapido)
3. **Al terminar (o si falla)**: copia los resultados de vuelta al home via `trap EXIT`

Esto es transparente — no hay que hacer nada manual. Los datos en scratch son locales al nodo, asi que si el job se relanza en otro nodo, se vuelve a copiar desde el home (que tiene los resultados del job anterior gracias al trap).

## Estructura de archivos

```
backend/scripts/cluster/
  submit.sh         # Script SLURM para enviar el job
  setup_env.sh      # Setup del entorno conda (ejecutar una sola vez)
  environment.yml   # Dependencias conda/pip
  pipeline.py       # Pipeline: Docling -> Chunks -> Extraccion
```

## Links utiles

- [Como ejecutar un trabajo](https://www.cluster.uy/ayuda/como_ejecutar/)
- [Como conectarse](https://cluster.uy/ayuda/como_conectarse/)
- [Ambientes con miniconda](https://www.cluster.uy/ayuda/ambiente/)
- [Recursos disponibles](https://www.cluster.uy/ayuda/recursos_disponibles/)
- [Consejos y buenas practicas](https://www.cluster.uy/ayuda/tips/)
- [Comandos utiles](https://cluster.uy/ayuda/comandos_utiles/)
