"""Tests for GraphRAGRetriever._fix_relationship_directions."""

from unittest.mock import MagicMock, patch

import pytest

from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever


@pytest.mark.parametrize(
    ("original_query", "expected_query"),
    [
        pytest.param(
            (
                "MATCH (p:Proyecto {titulo: 'mineria de procesos y datos'})"
                "-[:PARTICIPO_EN]->(i:Investigador)\n"
                "RETURN i"
            ),
            (
                "MATCH (p:Proyecto {titulo: 'mineria de procesos y datos'})"
                "<-[:PARTICIPO_EN]-(i:Investigador)\n"
                "RETURN i"
            ),
            id="propiedad-en-nodo",
        ),
        pytest.param(
            (
                "MATCH (p:Proyecto)-"
                "[r:PARTICIPO_EN {calidad: 'responsable'}]->"
                "(i:Investigador) RETURN p"
            ),
            (
                "MATCH (p:Proyecto)<-"
                "[r:PARTICIPO_EN {calidad: 'responsable'}]-"
                "(i:Investigador) RETURN p"
            ),
            id="propiedad-en-relacion",
        ),
        pytest.param(
            (
                "MATCH (p:Proyecto)-[:PARTICIPO_EN]->"
                "(i:Investigador)-[:EXTRAIDO_DE]->(c:Chunk) "
                "RETURN p, i, c"
            ),
            (
                "MATCH (p:Proyecto)<-[:PARTICIPO_EN]-"
                "(i:Investigador)<-[:EXTRAIDO_DE]-(c:Chunk) "
                "RETURN p, i, c"
            ),
            id="relaciones-encadenadas",
        ),
        pytest.param(
            (
                "MATCH (p:Proyecto), (i:Investigador)\n"
                "MATCH (p)-[:PARTICIPO_EN]->(i)\n"
                "RETURN p, i"
            ),
            (
                "MATCH (p:Proyecto), (i:Investigador)\n"
                "MATCH (p)<-[:PARTICIPO_EN]-(i)\n"
                "RETURN p, i"
            ),
            id="tipos-inferidos",
        ),
        pytest.param(
            (
                "MATCH (c:Chunk)-[:EXTRAIDO_DE]->"
                "(i:Investigador)\n"
                "MATCH (a:Entidad)-[:RELACION_DESCONOCIDA]->"
                "(b:OtraEntidad)\n"
                "RETURN c, i, a, b"
            ),
            (
                "MATCH (c:Chunk)-[:EXTRAIDO_DE]->"
                "(i:Investigador)\n"
                "MATCH (a:Entidad)-[:RELACION_DESCONOCIDA]->"
                "(b:OtraEntidad)\n"
                "RETURN c, i, a, b"
            ),
            id="relaciones-correctas-o-desconocidas-no-cambian",
        ),
    ],
)
def test_fix_relationship_directions(
    original_query: str,
    expected_query: str,
) -> None:
    """Correct reversed relationships and preserve valid patterns."""
    result = GraphRAGRetriever._fix_relationship_directions(
        None,
        original_query,
    )

    assert result == expected_query


def test_fix_relationship_directions_on_retriever_instance() -> None:
    """The method must also work when called from a retriever instance."""
    original_query = (
        "MATCH (p:Proyecto {titulo: 'web warehouse'})"
        "-[:PARTICIPO_EN]->(i:Investigador)\n"
        "OPTIONAL MATCH (i)-[:EXTRAIDO_DE]->(c:Chunk)\n"
        "RETURN i, c"
    )

    expected_query = (
        "MATCH (p:Proyecto {titulo: 'web warehouse'})"
        "<-[:PARTICIPO_EN]-(i:Investigador)\n"
        "OPTIONAL MATCH (i)<-[:EXTRAIDO_DE]-(c:Chunk)\n"
        "RETURN i, c"
    )

    with (
        patch("institutional_graphrag.retrieval.graph_retriever.GraphDatabase"),
        patch("institutional_graphrag.retrieval.graph_retriever.get_llm_client"),
    ):
        retriever = GraphRAGRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            fewshot_store=MagicMock(),
        )

        result = retriever._fix_relationship_directions(original_query)

    assert result == expected_query
