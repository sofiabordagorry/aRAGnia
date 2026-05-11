# README - Opciones para usar modelos HuggingFace

---

# 1. Modelo normal (bf16)

## Descargas

```bash
python -m pip install transformers==4.51.3
```

## Modelo

```txt
mistralai/Mistral-Small-24B-Instruct-2501
```

## Variables

```bash
export HF_QUANTIZATION="bf16"
```

---

# 2. bitsandbytes 4-bit

En Cluster no disponible, version de pip vieja

## Descargas

```bash
python -m pip install transformers==4.51.3

python -m pip install "bitsandbytes>=0.43.2"
```

## Modelo

```txt
mistralai/Mistral-Small-24B-Instruct-2501
```

## Variables

```bash
export HF_QUANTIZATION="bnb4"
```

---

# 3. AWQ

## Descargas

```bash
python -m pip install transformers==4.51.3

python -m pip install autoawq autoawq-kernels

python -m pip install torchvision==0.18.1

python -m pip install "numpy==1.26.4"
```

## Modelo

```txt
stelterlab/Mistral-Small-24B-Instruct-2501-AWQ
```

## Variables

```bash
export HF_QUANTIZATION="awq"
```

## Uso

- MUCHÍSIMA menos VRAM
- muy recomendado para 24B
- el modelo ya viene cuantizado

---

# 4. GPTQ

## Descargas

```bash
python -m pip install transformers==4.51.3

python -m pip install optimum accelerate

python -m pip install gptqmodel --no-build-isolation
```

## Modelo

```txt
repo/modelo-GPTQ
```

## Variables

```bash
export HF_QUANTIZATION="gptq"
```

## Uso

- menos VRAM
- rápido
- muy compatible

---

# Cache HuggingFace

## Crear carpeta

```bash
mkdir -p "$HOME/hf_cache"
```

## Variable

```bash
export HF_CACHE_DIR="$HOME/hf_cache"
```

---

# Variables recomendadas

```bash
export HF_MAX_GPU_MEMORY="42GiB"

export HF_MAX_CPU_MEMORY="80GiB"
```

---

# Recomendación para tu caso

Usar:

```txt
stelterlab/Mistral-Small-24B-Instruct-2501-AWQ
```

con:

```bash
export HF_QUANTIZATION="awq"
```
