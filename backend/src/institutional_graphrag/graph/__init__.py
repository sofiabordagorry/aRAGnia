"""Módulo de grafo."""

from .schema import (
    Anio,
    Chunk,
    Documento,
    Investigador,
    Proyecto,
    Topico,
    PARTICIPO_EN,
    TIENE_TOPICO,
    ES_DESCRITO_POR,
    INICIO_EN,
    PRIMER_CHUNK,
    SIGUIENTE_CHUNK,
    DE_DOCUMENTO,
    EVIDENCIA_DE,
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
