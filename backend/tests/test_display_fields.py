import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from neo4j import GraphDatabase

from aragnia.retrieval.graph_retriever import GraphRAGRetriever

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
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) WHERE toLower(i.nombre) CONTAINS 'garcia' RETURN i.nombre, p.valor",
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) WHERE toLower(i.nombre) CONTAINS 'garcia' RETURN i.nombre_de_despliegue, p.valor",
        ),
        (
            "Multiple Variables",
            "MATCH (x:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)<-[:PARTICIPO_EN {calidad: 'responsable'}]-(y:Investigador) RETURN x.nombre AS investigador, y.nombre AS responsable",
            "MATCH (x:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)<-[:PARTICIPO_EN {calidad: 'responsable'}]-(y:Investigador) RETURN x.nombre_de_despliegue AS investigador, y.nombre_de_despliegue AS responsable",
        ),
        (
            "Query without Investigador (Should not modify)",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.nombre, t.valor",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.nombre, t.valor",
        ),
        (
            "Proyecto title should become titulo_de_despliegue",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.titulo, t.valor",
            "MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t:Topico) RETURN p.titulo_de_despliegue, t.valor",
        ),
        (
            "Investigador and Proyecto display fields",
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) RETURN i.nombre AS investigador, p.titulo AS proyecto",
            "MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto) RETURN i.nombre_de_despliegue AS investigador, p.titulo_de_despliegue AS proyecto",
        ),
        (
            "Should not modify title before RETURN",
            "MATCH (p:Proyecto) WHERE toLower(p.titulo) CONTAINS 'datos abiertos' RETURN p.titulo AS proyecto",
            "MATCH (p:Proyecto) WHERE toLower(p.titulo) CONTAINS 'datos abiertos' RETURN p.titulo_de_despliegue AS proyecto",
        ),
    ],
)
def test_use_display_fields_for_return(test_name, original_query, expected_final_query):
    """Test that the regex correctly replaces .nombre with .nombre_de_despliegue only after RETURN."""

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
            RETURN i.id AS id, i.nombre AS nombre, i.nombre_de_despliegue AS nombre_de_despliegue
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
                    assert record["nombre"] is not None, f"Node {record['id']} has a NULL nombre"

                    # The critical check: nombre_de_despliegue must be populated
                    assert (
                        record["nombre_de_despliegue"] is not None
                    ), f"Node {record['id']} has a NULL nombre_de_despliegue! Migration Cypher query might have failed."
                    assert isinstance(
                        record["nombre_de_despliegue"], str
                    ), f"nombre_de_despliegue for {record['id']} is not a string."

                    expected_format = record["nombre_de_despliegue"].title()
                    assert (
                        record["nombre_de_despliegue"] == expected_format
                    ), f"El nombre_de_despliegue '{record['nombre_de_despliegue']}' del nodo {record['id']} no respeta el formato .title() (Se esperaba: '{expected_format}')"

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
            RETURN p.id AS id, p.titulo AS titulo, p.titulo_de_despliegue AS titulo_de_despliegue
            LIMIT 5
            """
            result = session.run(query)

            records = list(result)
            if not records:
                pytest.skip("No Proyecto nodes found in the database to run validation against.")

            for record in records:
                assert record["id"] is not None, "Found a Proyecto node with a NULL id"
                assert record["titulo"] is not None, f"Proyecto {record['id']} has a NULL titulo"

                assert (
                    record["titulo_de_despliegue"] is not None
                ), f"Proyecto {record['id']} has a NULL titulo_de_despliegue! Migration Cypher query might have failed."

                assert isinstance(
                    record["titulo_de_despliegue"], str
                ), f"titulo_de_despliegue for {record['id']} is not a string."

        driver.close()

    except Exception as e:
        print(f"Could not connect to Neo4j or execute query. Error: {e}")
        print(
            "Make sure your Neo4j container is running and the credentials in this script are correct."
        )
        raise
