# README - Opciones de carga de modelos HuggingFace

Este documento describe las distintas opciones disponibles para cargar modelos en HuggingFace dentro del proyecto.

---

# 1. Modelo normal (bf16 / fp16)

Carga el modelo original sin cuantización.

## Instalaciones necesarias

```bash
python -m pip install transformers==4.51.3
```

## Modelo de ejemplo

```txt
mistralai/Mistral-Small-24B-Instruct-2501
```

## Configuración `.env`

```bash
HF_QUANTIZATION="None"
```

---

# 2. Cuantización bitsandbytes 4-bit

Carga el modelo utilizando cuantización de 4 bits mediante `bitsandbytes`.

## Ventajas

- Mucho menor uso de VRAM
- Permite utilizar modelos más grandes en GPUs pequeñas

## Limitaciones

- Actualmente no disponible en Cluster por incompatibilidad de pip/bitsandbytes

## Instalaciones necesarias

```bash
python -m pip install transformers==4.51.3

python -m pip install "bitsandbytes>=0.43.2"
```

## Modelo de ejemplo

```txt
mistralai/Mistral-Small-24B-Instruct-2501
```

## Configuración `.env`

```bash
HF_QUANTIZATION="bnb4"
```

---

# 3. Cuantización AWQ

Carga un modelo previamente cuantizado con AWQ.

## Ventajas

- Menor uso de VRAM
- Inferencia más rápida
- En algunos casos mejor rendimiento que 4-bit estándar

## Importante

- Requiere modelos compatibles con AWQ
- Utilizar modelos que terminen en `-AWQ`

## Instalación

```bash
pip install autoawq==0.2.6

pip install transformers==4.46.3

pip install numpy==1.26.4
```

## Modelo de ejemplo

```txt
stelterlab/Mistral-Small-24B-Instruct-2501-AWQ
```

## Configuración `.env`

```bash
HF_QUANTIZATION="None"
```

---

# Resumen

| Modo         | Cuantización | Menor uso de VRAM | Requiere modelo especial |
| ------------ | ------------ | ----------------- | ------------------------ |
| Normal       | No           | No                | No                       |
| bitsandbytes | 4-bit        | Sí                | No                       |
| AWQ          | AWQ          | Sí                | Sí                       |
