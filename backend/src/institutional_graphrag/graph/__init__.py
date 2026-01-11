"""Módulo de grafo."""

from .schema import (
    DE_DOCUMENTO,
    ES_DESCRITO_POR,
    EVIDENCIA_DE,
    INICIO_EN,
    PARTICIPO_EN,
    PRIMER_CHUNK,
    SIGUIENTE_CHUNK,
    TIENE_TOPICO,
    Anio,
    Chunk,
    Documento,
    Investigador,
    Proyecto,
    Topico,
)

__all__ = [
    # Entities
    "Anio",
    "Chunk",
    "Documento",
    "Investigador",
    "Proyecto",
    "Topico",
    # Relationships
    "PARTICIPO_EN",
    "TIENE_TOPICO",
    "ES_DESCRITO_POR",
    "INICIO_EN",
    "PRIMER_CHUNK",
    "SIGUIENTE_CHUNK",
    "DE_DOCUMENTO",
    "EVIDENCIA_DE",
]
