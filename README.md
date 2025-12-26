# Institutional GraphRAG

Proyecto de grado de Ingeniería en Computación (FIng – Udelar) 2025-2026.

Este repositorio contiene el código de un pipeline incremental de GraphRAG para trabajar con informes de proyectos institucionales y permitir consultas longitudinales sobre los mismos.

## Requisitos

- Python 3.11
- pip

## Instalación

Se recomienda usar un entorno virtual.

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv/Scripts/activate

pip install -e ".[dev]"
```

## Ejecución

### Formatear código

```bash
./scripts/format.sh
```

### Ejecutar tests y verificaciones

```bash
./scripts/check.sh
```

Este comando ejecuta:
- Linter (ruff)
- Verificación de formato (black)
- Type checking (mypy)
- Tests con cobertura (pytest)

### Descargar corpus de PDFs

El proyecto incluye un script para descargar los PDFs desde una lista de URLs.

- La lista de URLs se encuentra en data/downloads_list.txt.

- Los PDFs descargados se guardan en la carpeta data/corpus.

# Ejecutar el script de descarga
```bash
python scripts/download_corpus.py
```
