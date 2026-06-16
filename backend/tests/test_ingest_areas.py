import os
from pathlib import Path

import pytest
from dotenv import load_dotenv
from neo4j import GraphDatabase

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")


@pytest.fixture(scope="module")
def neo4j_session():
    """Fixture to set up and tear down the Neo4j connection."""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Verify connection
    try:
        driver.verify_connectivity()
    except Exception as e:
        pytest.fail(f"Could not connect to Neo4j: {e}")

    with driver.session() as session:
        yield session

    driver.close()


def test_areas_were_ingested_correctly(neo4j_session):
    """Verifies that the specific Area nodes exist with correct IDs and values."""

    query = "MATCH (a:Area) RETURN a.id AS id, a.valor AS value"
    result = neo4j_session.run(query)

    areas_in_db = {record["id"]: record["value"] for record in result}

    expected_areas = {
        "area_basica": "basica",
        "area_salud": "salud",
        "area_agraria": "agraria",
        "area_social_y_artistica": "social y artistica",
        "area_tecnologica": "tecnologica",
    }

    # Assert that each expected area exists and matches
    for expected_id, expected_value in expected_areas.items():
        assert expected_id in areas_in_db, f"Missing Area ID in database: '{expected_id}'"

        actual_value = areas_in_db[expected_id]
        assert (
            actual_value == expected_value
        ), f"Value mismatch for '{expected_id}'. Expected '{expected_value}', got '{actual_value}'."
