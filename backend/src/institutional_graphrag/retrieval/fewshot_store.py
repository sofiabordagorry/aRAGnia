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
        "cuántos proyectos tiene X?",
        "MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'lastname'\nMATCH (i)-[:PARTICIPO_EN]->(p:Proyecto)\nRETURN count(p) AS total",
    ),
    (
        "de cuántos proyectos fue responsable X?",
        "MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'lastname'\nMATCH (i)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p:Proyecto)\nRETURN count(p) AS total",
    ),
    (
        "cuántos proyectos hay en 2018?",
        "MATCH (a:Anio)\nWHERE a.year = '2018'\nMATCH (p:Proyecto)-[:INICIO_EN]->(a)\nRETURN count(p) AS total",
    ),
    (
        "cuántos proyectos de biotecnología hay?",
        "MATCH (t:Topico)\nWHERE t.value = 'biotecnologia'\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nRETURN count(p) AS total",
    ),
    (
        "cuántos investigadores hay?",
        "MATCH (i:Investigador) RETURN count(i) AS total",
    ),
    (
        "cuántos proyectos hay de ciencias naturales?",
        "MATCH (s:Subcampo)\nWHERE s.value = 'ciencias naturales'\nMATCH (t:Topico)-[:PERTENECE_A_SUBCAMPO]->(s)\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nRETURN count(DISTINCT p) AS total",
    ),
    (
        "cuántos proyectos hay del área tecnológica?",
        "MATCH (a:Area)\nWHERE a.value = 'tecnologica'\nMATCH (p:Proyecto)-[:PERTENECE_A_AREA]->(a)\nRETURN count(p) AS total",
    ),
    # LIST queries
    (
        "qué proyectos hay de biotecnología?",
        "MATCH (t:Topico)\nWHERE t.value = 'biotecnologia'\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN p, COLLECT(c) AS chunks",
    ),
    (
        "quiénes participaron en el proyecto proy_2020_513?",
        "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)\nWHERE p.id = 'proy_2020_513'\nOPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)\nRETURN i, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "quién fue el responsable del proyecto proy_2020_513?",
        "MATCH (i:Investigador)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p:Proyecto)\nWHERE p.id = 'proy_2020_513'\nOPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)\nRETURN i, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "qué proyectos hay de ciencias naturales?",
        "MATCH (s:Subcampo)\nWHERE s.value = 'ciencias naturales'\nMATCH (t:Topico)-[:PERTENECE_A_SUBCAMPO]->(s)\nMATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN DISTINCT p, s, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "busca el proyecto web warehouse de datos abiertos",
        "MATCH (p:Proyecto)\nWHERE toLower(p.value) CONTAINS 'web warehouse de datos abiertos'\nOPTIONAL MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN p, COLLECT(DISTINCT c) AS chunks",
    ),
    (
        "qué proyectos tiene el investigador con apellido perez?",
        "MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'perez'\nMATCH (i)-[:PARTICIPO_EN]->(p:Proyecto)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN p, i, COLLECT(c) AS chunks",
    ),
    (
        "de qué proyectos fue responsable perez?",
        "MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'perez'\nMATCH (i)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p:Proyecto)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN p, i, COLLECT(c) AS chunks",
    ),
    (
        "qué documentos tiene el proyecto proy_2020_513?",
        "MATCH (p:Proyecto {id: 'proy_2020_513'})\nMATCH (p)-[:ES_DESCRITO_POR]->(d:Documento)\nMATCH (c:Chunk)-[:DE_DOCUMENTO]->(d)\nRETURN d, p, COLLECT(c) AS chunks",
    ),
    (
        "describí el proyecto proy_2020_513",
        "MATCH (p:Proyecto {id: 'proy_2020_513'})\nOPTIONAL MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nOPTIONAL MATCH (p)-[:TIENE_TOPICO]->(t:Topico)\nOPTIONAL MATCH (maininv:Investigador)-[:PARTICIPO_EN {calidad: 'responsable'}]->(p)\nOPTIONAL MATCH (inv:Investigador)-[:PARTICIPO_EN]->(p)\nOPTIONAL MATCH (p)-[:INICIO_EN]->(a:Anio)\nRETURN p, COLLECT(DISTINCT c) AS chunks, COLLECT(DISTINCT t) AS topics, COLLECT(DISTINCT inv) AS investigators, COLLECT(DISTINCT maininv) AS researchers_in_charge, a LIMIT 1",
    ),
    (
        "qué proyectos iniciaron en 2018?",
        "MATCH (a:Anio {year: '2018'})\nMATCH (p:Proyecto)-[:INICIO_EN]->(a)\nMATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)\nRETURN p, COLLECT(c) AS chunks",
    ),
    (
        "qué proyectos hay en el área básica?",
        "MATCH (a:Area)\nWHERE a.value = 'basica'\nMATCH (p:Proyecto)-[:PERTENECE_A_AREA]->(a)\nRETURN DISTINCT p, a",
    ),
    # AGGREGATION
    (
        "cuáles son los 10 tópicos con más proyectos?",
        "MATCH (t:Topico)<-[:TIENE_TOPICO]-(p:Proyecto)\nRETURN t.value AS area, count(DISTINCT p) AS total ORDER BY total DESC LIMIT 10",
    ),
    (
        "qué investigadores participaron en más proyectos? top 10",
        "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)\nWITH i, count(DISTINCT p) AS num_proyectos\nRETURN i.name AS investigador, num_proyectos\nORDER BY num_proyectos DESC\nLIMIT 10",
    ),
    (
        "en qué año inició el proyecto proy_2020_513?",
        "MATCH (p:Proyecto {id: 'proy_2020_513'})\nOPTIONAL MATCH (p)-[:INICIO_EN]->(a:Anio)\nRETURN a.year AS año, a",
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
        """Return the top_k most similar (question, cypher_query) pairs."""
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
