# README - Opciones para modelos HuggingFace

---

# 1. Modelo normal (bf16)

## Descargas

```bash
python -m pip install transformers==4.51.3
```

## Modelo de ejemplo

```txt
mistralai/Mistral-Small-24B-Instruct-2501
```

## Variables

```bash
export HF_QUANTIZATION="bf16"
```

---

# 2. bitsandbytes 4-bit

En Cluster no disponible, version de pip no compatible

## Descargas

```bash
python -m pip install transformers==4.51.3

python -m pip install "bitsandbytes>=0.43.2"
```

## Modelo de ejemplo

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
pip install autoawq==0.2.6

pip install transformers==4.46.3

pip install numpy==1.26.4
```

## Modelo de ejemplo

```txt
stelterlab/Mistral-Small-24B-Instruct-2501-AWQ
```

## Variables

```bash
export HF_QUANTIZATION="awq"
```
