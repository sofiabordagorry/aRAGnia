"""Tests for the GraphRAGRetriever query direction correction logic."""

from unittest.mock import MagicMock, patch
import pytest

from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever


def test_fix_relationship_directions_chunk_investigador():
    """
    Test that _fix_relationship_directions correctly fixes the direction of
    the (Chunk)-[:EXTRAIDO_DE]->(Investigador) relationship to
    (Investigador)-[:EXTRAIDO_DE]->(Chunk).
    """
    original_query = (
        "MATCH (i:Investigador)-[:PARTICIPO_EN {calidad:'responsable'}]->(p:Proyecto)\n"
        "WHERE toLower(p.titulo) CONTAINS 'web warehouse de datos abiertos de gobierno con gestion de calidad'\n"
        "OPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i:Investigador)"
    )

    expected_query = (
        "MATCH (i:Investigador)-[:PARTICIPO_EN {calidad:'responsable'}]->(p:Proyecto)\n"
        "WHERE toLower(p.titulo) CONTAINS 'web warehouse de datos abiertos de gobierno con gestion de calidad'\n"
        "OPTIONAL MATCH (i:Investigador)-[:EXTRAIDO_DE]->(c:Chunk)"
    )

    # 1. Test using the class unbound (passing None for self, which works as self is not used)
    result_unbound = GraphRAGRetriever._fix_relationship_directions(None, original_query)
    assert result_unbound == expected_query, f"Unbound method call failed.\nExpected: {expected_query}\nGot: {result_unbound}"

    # 2. Test using a mocked instance of GraphRAGRetriever
    with patch("institutional_graphrag.retrieval.graph_retriever.GraphDatabase") as mock_db, \
         patch("institutional_graphrag.retrieval.graph_retriever.get_llm_client") as mock_llm:
        
        # Instantiate retriever with dummy parameters, mocking out FewShotStore
        retriever = GraphRAGRetriever(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            fewshot_store=MagicMock()
        )
        
        result_bound = retriever._fix_relationship_directions(original_query)
        assert result_bound == expected_query, f"Bound method call failed.\nExpected: {expected_query}\nGot: {result_bound}"
