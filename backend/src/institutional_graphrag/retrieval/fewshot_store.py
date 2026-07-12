"""Few-shot example store for Text2Cypher prompt augmentation."""

from __future__ import annotations

from typing import List, Tuple

from institutional_graphrag.ingest.embedder import E5Embedder
from institutional_graphrag.retrieval.vector_store import VectorStore

COLLECTION_NAME = "cypher_fewshot_examples"
EMBEDDING_DIM = 1024  # e5-large-v2


DEFAULT_EXAMPLES: List[Tuple[str, str]] = [
    # COUNT queries
    (
        "cuántos proyectos tiene el investigador gonzalez?",
        "MATCH (i:Investigador) WHERE toLower(i.nombre) CONTAINS 'gonzalez'\nMATCH (i)-[:PARTICIPO_EN]->(p:Proyecto)\nRETURN count(DISTINCT p) AS total",
    ),
    (
        "de cuántos proyectos fue responsable gonzalez?",
        "MATCH (i:Investigador) WHERE toLower(i.nombre) CONTAINS 'gonzalez'\nMATCH (i)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p:Proyecto)\nRETURN count(DISTINCT p) AS total",
    ),
    (
        "cuántos proyectos hay en 2018?",
        "MATCH (a:Anio)\nWHERE a.anio = '2018'\nMATCH (p:Proyecto)-[:INICIO_EN]->(a)\nRETURN count(p) AS total",
    ),
    (
        "cuántos proyectos de biotecnología hay?",
        "MATCH (t:Topico)\nWHERE t.valor = 'biotecnologia'\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nRETURN count(DISTINCT p) AS total",
    ),
    (
        "cuántos investigadores hay?",
        "MATCH (i:Investigador) RETURN count(i) AS total",
    ),
    (
        "cuántos proyectos hay de ciencias naturales?",
        "MATCH (s:Subcampo)\nWHERE s.valor = 'ciencias naturales'\nMATCH (t:Topico)-[:PERTENECE_A_SUBCAMPO]->(s)\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nRETURN count(DISTINCT p) AS total",
    ),
    (
        "cuántos proyectos hay del área tecnológica?",
        "MATCH (a:Area)\nWHERE a.valor = 'tecnologica'\nMATCH (p:Proyecto)-[:PERTENECE_A_AREA]->(a)\nRETURN count(DISTINCT p) AS total",
    ),
    (
        "¿cuántos proyectos investigan modelado, simulación y optimización?",
        "MATCH (t:Topico) WHERE t.valor = 'modelado, simulacion y optimizacion' MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t) RETURN count(DISTINCT p) AS total",
    ),
    # LIST queries
    (
        "qué proyectos hay de biotecnología?",
        "MATCH (t:Topico)\nWHERE t.valor = 'biotecnologia'\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN DISTINCT p, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "quiénes participaron en el proyecto co-simulacion en sistemas ciber fisicos?",
        "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)\nWHERE toLower(p.titulo) CONTAINS 'co-simulacion en sistemas ciber fisicos'\nOPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)\nRETURN i, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "quién fue el responsable del proyecto co-simulacion en sistemas ciber fisicos?",
        "MATCH (i:Investigador)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p:Proyecto)\nWHERE toLower(p.titulo) CONTAINS 'co-simulacion en sistemas ciber fisicos'\nOPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)\nRETURN i, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "qué proyectos hay de ciencias naturales?",
        "MATCH (s:Subcampo)\nWHERE s.valor = 'ciencias naturales'\nMATCH (t:Topico)-[:PERTENECE_A_SUBCAMPO]->(s)\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN DISTINCT p, s, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "busca el proyecto web warehouse de datos abiertos",
        "MATCH (p:Proyecto)\nWHERE toLower(p.titulo) CONTAINS 'web warehouse de datos abiertos'\nOPTIONAL MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN p, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "qué proyectos tiene el investigador con apellido perez?",
        "MATCH (i:Investigador) WHERE toLower(i.nombre) CONTAINS 'perez'\nMATCH (i)-[:PARTICIPO_EN]->(p:Proyecto)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN DISTINCT p, i, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "de qué proyectos fue responsable perez?",
        "MATCH (i:Investigador) WHERE toLower(i.nombre) CONTAINS 'perez'\nMATCH (i)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p:Proyecto)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN DISTINCT p, i, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "qué documentos tiene el proyecto proy_2020_513?",
        "MATCH (p:Proyecto {id: 'proy_2020_513'})\nMATCH (p)-[:ES_DESCRITO_POR]->(d:Documento)\nMATCH (c:Chunk)-[:DE_DOCUMENTO]->(d)\nRETURN d, p, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "describí el proyecto proy_2020_513",
        "MATCH (p:Proyecto {id: 'proy_2020_513'})\nOPTIONAL MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nOPTIONAL MATCH (p)-[:TIENE_TOPICO]->(t:Topico)\nOPTIONAL MATCH (maininv:Investigador)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p)\nOPTIONAL MATCH (inv:Investigador)-[:PARTICIPO_EN]->(p)\nOPTIONAL MATCH (p)-[:INICIO_EN]->(a:Anio)\nRETURN p, COLLECT(DISTINCT c) AS chunks, COLLECT(DISTINCT t) AS topics, COLLECT(DISTINCT inv) AS investigators, COLLECT(DISTINCT maininv) AS researchers_in_charge, a",
    ),
    (
        "qué proyectos iniciaron en 2018?",
        "MATCH (a:Anio {anio: '2018'})\nMATCH (p:Proyecto)-[:INICIO_EN]->(a)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN DISTINCT p, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "qué proyectos hay en el área básica?",
        "MATCH (a:Area)\nWHERE a.valor = 'basica'\nMATCH (p:Proyecto)-[:PERTENECE_A_AREA]->(a)\nRETURN DISTINCT p, a",
    ),
    # GRUPO queries (gi_* entities behave exactly like Proyecto)
    (
        "cuántos grupos de investigación hay?",
        "MATCH (g:Grupo) RETURN count(g) AS total",
    ),
    (
        "qué grupos hay del área tecnológica?",
        "MATCH (a:Area)\nWHERE a.valor = 'tecnologica'\nMATCH (g:Grupo)-[:PERTENECE_A_AREA]->(a)\nMATCH (g)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN DISTINCT g, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "quiénes integran el grupo gi_2014_133?",
        "MATCH (i:Investigador)-[:PARTICIPO_EN]->(g:Grupo {id: 'gi_2014_133'})\nOPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)\nRETURN i, COLLECT(DISTINCT c) AS chunks",
    ),
    # AGGREGATION
    (
        "cuáles son los 10 tópicos con más proyectos?",
        "MATCH (t:Topico)<-[:TIENE_TOPICO]-(p:Proyecto)\nRETURN t.valor AS topico, count(DISTINCT p) AS total ORDER BY total DESC LIMIT 10",
    ),
    (
        "qué investigadores participaron en más proyectos? top 10",
        "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)\nWITH i, count(DISTINCT p) AS num_proyectos\nRETURN i.nombre AS investigador, num_proyectos\nORDER BY num_proyectos DESC\nLIMIT 10",
    ),
    (
        "en qué año inició el proyecto proy_2020_513?",
        "MATCH (p:Proyecto {id: 'proy_2020_513'})\nOPTIONAL MATCH (p)-[:INICIO_EN]->(a:Anio)\nRETURN a.anio AS año, a",
    ),
    # TEMPORAL
    (
        "¿Qué subcampos aparecieron en proyectos iniciados a partir de 2016 y no estaban presentes antes de ese año?",
        "MATCH (p_nuevo:Proyecto)-[:INICIO_EN]->(a_nuevo:Anio) MATCH (p_nuevo)-[:TIENE_TOPICO]->(t_nuevo:Topico) MATCH (t_nuevo)-[:PERTENECE_A_SUBCAMPO]->(s_nuevo:Subcampo) WHERE toInteger(a_nuevo.anio) >= 2016 WITH collect(DISTINCT s_nuevo.valor) AS subcampos_desde_2016 MATCH (p_anterior:Proyecto)-[:INICIO_EN]->(a_anterior:Anio)\n"
        "MATCH (p_anterior)-[:TIENE_TOPICO]->(t_anterior:Topico) MATCH (t_anterior)-[:PERTENECE_A_SUBCAMPO]->(s_anterior:Subcampo) WHERE toInteger(a_anterior.anio) < 2016 WITH subcampos_desde_2016, collect(DISTINCT s_anterior.valor) AS subcampos_antes_2016\n"
        "RETURN [s IN subcampos_desde_2016 "
        "WHERE NOT s IN subcampos_antes_2016] AS subcampos_nuevos",
    ),
    # Preguntas fuera de alcance (no representables en el esquema)
    (
        "¿cuál fue el presupuesto total en dólares del proyecto x?",
        "NOT_IN_SCHEMA",
    ),
    (
        "¿cuál es el correo electrónico o teléfono de contacto de álvaro martín?",
        "NOT_IN_SCHEMA",
    ),
    (
        "¿cuántas horas semanales dedica cada investigador al proyecto x?",
        "NOT_IN_SCHEMA",
    ),
]


class FewShotStore:
    """Stores and retrieves (question, cypher_query) pairs via vector similarity."""

    def __init__(self) -> None:
        self._embedder = E5Embedder()
        self._store = VectorStore(
            collection_name=COLLECTION_NAME,
            embedding_dim=EMBEDDING_DIM,
        )

    def add_examples(self, examples: List[Tuple[str, str]]) -> None:
        """Add (question, cypher_query) pairs. Skips duplicates by question text."""
        questions = [q for q, _ in examples]
        existing = self._store.existing_payload_values("question", questions)
        new = [(q, c) for q, c in examples if q not in existing]
        if not new:
            return

        embeddings = [self._embedder.embed_query(q)[0].tolist() for q, _ in new]
        metadata = [{"question": q, "cypher": c} for q, c in new]
        self._store.add_documents(embeddings, metadata)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, str]]:
        """Return the top_k most similar (question, cypher_query) pairs.

        score_threshold is a calibration knob: with E5 (normalized) question->question
        matching, relevant pairs usually score high, so 0.5 is permissive. Raise it
        (e.g. 0.7-0.8) if irrelevant few-shots are leaking into the prompt.
        """
        embedding = self._embedder.embed_query(query)[0].tolist()
        results = self._store.search(embedding, top_k=top_k, score_threshold=0.5)
        return [
            (r[2]["question"], r[2]["cypher"])
            for r in results
            if "question" in r[2] and "cypher" in r[2]
        ]

    def count(self) -> int:
        return self._store.count_documents()

    def close(self) -> None:
        self._store.close()

    def clear(self) -> None:
        self._store.clear_collection()
