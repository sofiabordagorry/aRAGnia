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

El proyecto incluye un script para descargar los PDF de la nube institucional a partir de una lista de URLs, las cuales corresponden a pares de documentos (Informe Final y Propuesta de Postulación). Además, se incorporaron tres documentos adicionales que contienen resultados o conclusiones de varios proyectos correspondientes a un año determinado.
- La lista de URLs se encuentra en data/downloads_list.txt.

- Los PDFs descargados se guardan en la carpeta data/corpus.

- El acceso a la nube se realiza mediante un enlace público de Nextcloud, cuyo token debe configurarse a través de una variable de entorno.

#### Configuración del token

Antes de ejecutar el script, es necesario crear un archivo `.env` en la raíz del proyecto con una variable de entorno llamada **FING_TOKEN**


#### Ejecutar el script de descarga
```bash
python scripts/download_corpus.py
```
