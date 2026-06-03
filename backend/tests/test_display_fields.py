import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from neo4j import GraphDatabase

from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever

env_path = Path(__file__).parents[1] / ".env"
load_dotenv(env_path)

# ==============================================================================
# PART 1: Test the Query Post-Processing Logic
# ==============================================================================


@pytest.mark.parametrize(
    "test_name, original_query, expected_final_query",
    [
        (
            "Standard Query",
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) WHERE toLower(i.name) CONTAINS 'garcia' RETURN i.name, p.value",
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) WHERE toLower(i.name) CONTAINS 'garcia' RETURN i.display_name, p.value",
        ),
        (
            "Multiple Variables",
            "MATCH (x:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)<-[:PARTICIPO_EN {calidad: 'responsable'}]-(y:Investigador) RETURN x.name AS investigador, y.name AS responsable",
            "MATCH (x:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)<-[:PARTICIPO_EN {calidad: 'responsable'}]-(y:Investigador) RETURN x.display_name AS investigador, y.display_name AS responsable",
        ),
        (
            "Query without Investigador (Should not modify)",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.name, t.value",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.name, t.value",
        ),
        (
            "Proyecto title should become display_title",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.title, t.value",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.display_title, t.value",
        ),
        (
            "Investigador and Proyecto display fields",
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) RETURN i.name AS investigador, p.title AS proyecto",
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) RETURN i.display_name AS investigador, p.display_title AS proyecto",
        ),
        (
            "Should not modify title before RETURN",
            "MATCH (p:Proyecto) WHERE toLower(p.title) CONTAINS 'datos abiertos' RETURN p.title AS proyecto",
            "MATCH (p:Proyecto) WHERE toLower(p.title) CONTAINS 'datos abiertos' RETURN p.display_title AS proyecto",
        ),
    ],
)
def test_use_display_fields_for_return(test_name, original_query, expected_final_query):
    """Test that the regex correctly replaces .name with .display_name only after RETURN."""

    # Llamamos al método
    processed = GraphRAGRetriever._use_display_fields_for_return(original_query)

    # Comparamos el string final completo
    assert (
        processed == expected_final_query
    ), f"[{test_name}] La query procesada no coincide con lo esperado."


# ==============================================================================
# PART 2: Test the Neo4j Database State
# ==============================================================================


def test_neo4j_database_display_name_populated():
    print("\n--- PART 2: Testing Neo4j Database ---")

    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session() as session:
            # Query 5 random researchers to check their properties
            query = """
            MATCH (i:Investigador) 
            RETURN i.id AS id, i.name AS name, i.display_name AS display_name 
            LIMIT 5
            """
            result = session.run(query)

            records = list(result)
            if not records:
                pytest.skip(
                    "No Investigador nodes found in the database to run validation against."
                )
            else:
                print("Found Researchers:")
                for record in records:
                    # Check standard properties exist
                    assert record["id"] is not None, "Found a node with a NULL id"
                    assert record["name"] is not None, f"Node {record['id']} has a NULL name"

                    # The critical check: display_name must be populated
                    assert (
                        record["display_name"] is not None
                    ), f"Node {record['id']} has a NULL display_name! Migration Cypher query might have failed."
                    assert isinstance(
                        record["display_name"], str
                    ), f"display_name for {record['id']} is not a string."

                    expected_format = record["display_name"].title()
                    assert (
                        record["display_name"] == expected_format
                    ), f"El display_name '{record['display_name']}' del nodo {record['id']} no respeta el formato .title() (Se esperaba: '{expected_format}')"

        driver.close()

    except Exception as e:
        print(f"Could not connect to Neo4j or execute query. Error: {e}")
        print(
            "Make sure your Neo4j container is running and the credentials in this script are correct."
        )


def test_neo4j_database_display_title_populated():
    print("\n--- Testing Neo4j Proyecto display_title ---")

    neo4j_host = os.getenv("HOST", "localhost")
    neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
    neo4j_user = os.getenv("NEO4J_USER")
    neo4j_password = os.getenv("NEO4J_PASSWORD")
    neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

        with driver.session() as session:
            query = """
            MATCH (p:Proyecto)
            RETURN p.id AS id, p.title AS title, p.display_title AS display_title
            LIMIT 5
            """
            result = session.run(query)

            records = list(result)
            if not records:
                pytest.skip("No Proyecto nodes found in the database to run validation against.")

            for record in records:
                assert record["id"] is not None, "Found a Proyecto node with a NULL id"
                assert record["title"] is not None, f"Proyecto {record['id']} has a NULL title"

                assert (
                    record["display_title"] is not None
                ), f"Proyecto {record['id']} has a NULL display_title! Migration Cypher query might have failed."

                assert isinstance(
                    record["display_title"], str
                ), f"display_title for {record['id']} is not a string."

        driver.close()

    except Exception as e:
        print(f"Could not connect to Neo4j or execute query. Error: {e}")
        print(
            "Make sure your Neo4j container is running and the credentials in this script are correct."
        )
        raise
