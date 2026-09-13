"""
Reglas de negocio que el grafo debe satisfacer.

A diferencia de las restricciones del esquema, que son estructurales, estas reglas
expresan condiciones propias del dominio. Se verifican sobre el grafo ya cargado.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SanityCheck:
    """
    Regla de negocio, con la consulta que devuelve los casos que no la cumplen.

    La consulta debe devolver `count`, la cantidad de casos, y `examples`, algunos
    identificadores para poder revisarlos.
    """

    name: str
    description: str
    query: str

    def __str__(self) -> str:
        return self.description


SANITY_CHECKS: tuple[SanityCheck, ...] = (
    SanityCheck(
        name="proyecto_sin_topico",
        description="Un proyecto debe tener como mínimo un tópico asociado",
        query="""
        MATCH (p:Proyecto)
        WHERE NOT (p)-[:TIENE_TOPICO]->(:Topico)
        RETURN count(p) AS count, collect(p.id)[0..5] AS examples
        """,
    ),
    SanityCheck(
        name="chunks_consecutivos_de_distinto_documento",
        description=(
            "Si un chunk es el siguiente de otro, ambos deben pertenecer al mismo documento"
        ),
        query="""
        MATCH (a:Chunk)-[:SIGUIENTE_CHUNK]->(b:Chunk)
        MATCH (a)-[:DE_DOCUMENTO]->(documento_a:Documento)
        MATCH (b)-[:DE_DOCUMENTO]->(documento_b:Documento)
        WHERE documento_a <> documento_b
        RETURN count(*) AS count, collect(a.id + ' -> ' + b.id)[0..5] AS examples
        """,
    ),
    SanityCheck(
        name="chunk_inalcanzable",
        description=(
            "Todo chunk de un documento debe ser alcanzable desde el primer chunk "
            "siguiendo la relación SIGUIENTE_CHUNK"
        ),
        query="""
        MATCH (d:Documento)-[:PRIMER_CHUNK]->(primero:Chunk)
        MATCH (c:Chunk)-[:DE_DOCUMENTO]->(d)
        WHERE NOT (primero)-[:SIGUIENTE_CHUNK*0..]->(c)
        RETURN count(c) AS count, collect(c.id)[0..5] AS examples
        """,
    ),
    SanityCheck(
        name="ciclo_entre_chunks",
        description="No pueden existir ciclos en la relación SIGUIENTE_CHUNK",
        query="""
        MATCH (c:Chunk)
        WHERE (c)-[:SIGUIENTE_CHUNK*1..]->(c)
        RETURN count(c) AS count, collect(c.id)[0..5] AS examples
        """,
    ),
    SanityCheck(
        name="participacion_sin_evidencia",
        description=(
            "Si un investigador participa en un proyecto, debe existir al menos un chunk "
            "asociado a ese investigador que pertenezca a un documento que describa el proyecto"
        ),
        query="""
        MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)
        WHERE NOT EXISTS {
            MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)
            MATCH (c)-[:DE_DOCUMENTO]->(d:Documento)
            MATCH (p)-[:ES_DESCRITO_POR]->(d)
        }
        RETURN count(*) AS count, collect(i.id + ' -> ' + p.id)[0..5] AS examples
        """,
    ),
    SanityCheck(
        name="topico_sin_evidencia",
        description=(
            "Si un proyecto tiene un tópico asociado, debe existir al menos un chunk "
            "asociado a ese tópico que pertenezca a un documento que describa el proyecto"
        ),
        query="""
        MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico)
        WHERE NOT EXISTS {
            MATCH (c:Chunk)-[:EXTRAIDO_DE]->(t)
            MATCH (c)-[:DE_DOCUMENTO]->(d:Documento)
            MATCH (p)-[:ES_DESCRITO_POR]->(d)
        }
        RETURN count(*) AS count, collect(p.id + ' -> ' + t.id)[0..5] AS examples
        """,
    ),
    SanityCheck(
        name="titulo_de_documento_ajeno",
        description=(
            "Si un proyecto tiene una relación TITULO_EXTRAIDO_DE con un chunk, ese chunk "
            "debe pertenecer a un documento asociado a ese mismo proyecto"
        ),
        query="""
        MATCH (p:Proyecto)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
        WHERE NOT EXISTS {
            MATCH (c)-[:DE_DOCUMENTO]->(d:Documento)
            MATCH (p)-[:ES_DESCRITO_POR]->(d)
        }
        RETURN count(*) AS count, collect(p.id + ' -> ' + c.id)[0..5] AS examples
        """,
    ),
)
