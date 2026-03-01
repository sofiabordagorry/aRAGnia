"""Tests for the graph schema module."""

import pytest
import json
from pathlib import Path
from pydantic import TypeAdapter

from institutional_graphrag.graph.schema import (
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
    GraphSchema,
    Investigador,
    Proyecto,
    Relationship,
    Topico,
    validate_relationship_endpoints,
    AnioValue,
    DocumentoValue,
    InvestigadorValue,
)

VALUE_SCHEMAS = {
    "Anio": TypeAdapter(AnioValue),
    "Documento": TypeAdapter(DocumentoValue),
    "Investigador": TypeAdapter(InvestigadorValue),
}


# auxiliary function
def load_entities():
    current_file = Path(__file__).resolve()
    json_path = current_file.parents[2] / "data" / "entities_relations" / "entity_documents.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data["entities"]


class TestEntities:
    """Test entity classes."""

    def test_proyecto_creation(self):
        """Test creating a Proyecto entity."""
        proyecto = Proyecto(id="proyecto_123", value="GraphRAG Research")
        assert proyecto.id == "proyecto_123"
        assert proyecto.label == "Proyecto"
        assert proyecto.value == "GraphRAG Research"

    def test_investigador_creation(self):
        """Test creating an Investigador entity."""
        investigador = Investigador(id="inv_456", value="Dr. Smith")
        assert investigador.id == "inv_456"
        assert investigador.label == "Investigador"

    def test_topico_creation(self):
        """Test creating a Topico entity."""
        topico = Topico(id="topico_789", value="Machine Learning")
        assert topico.id == "topico_789"
        assert topico.label == "Topico"

    def test_documento_creation(self):
        """Test creating a Documento entity."""
        documento = Documento(id="doc_101", value="Propuesta 2024")
        assert documento.id == "doc_101"
        assert documento.label == "Documento"

    def test_chunk_creation(self):
        """Test creating a Chunk entity."""
        chunk = Chunk(id="chunk_202", value="Sample text")
        assert chunk.id == "chunk_202"
        assert chunk.label == "Chunk"

    def test_anio_creation(self):
        """Test creating an Anio entity."""
        anio = Anio(id="2024", value=2024)
        assert anio.id == "2024"
        assert anio.label == "Anio"
        assert anio.id == "2024"
        assert anio.label == "Anio"

    def test_entity_without_id_raises_error(self):
        """Test that creating an entity without ID raises ValueError."""
        with pytest.raises(ValueError):
            Proyecto(id="", value="Test Project")

    def test_entity_to_dict(self):
        """Test converting entity to dictionary."""
        proyecto = Proyecto(id="proyecto_123", value="Test Project")
        result = proyecto.to_dict()
        assert result["id"] == "proyecto_123"
        assert result["label"] == "Proyecto"
        assert result["value"] == "Test Project"

    def test_entity_schema(self):
        """
        Tests if entity data complies with the expected schema.
        """
        entities = load_entities()
        failures = []

        for entity_data in entities:
            entity_id = entity_data.get("id")
            label = entity_data.get("label")
            value = entity_data.get("value")

            # 1. Check label exists
            if label is None:
                failures.append(f"Entity {entity_id}: missing label")
                continue

            try:
                # 2. Get Class
                EntityClass = GraphSchema.get_entity_class(label)

                # 3. Validate
                EntityClass(id=entity_id, value=value)
                if label in VALUE_SCHEMAS:
                    schema_adapter = VALUE_SCHEMAS[label]
                    schema_adapter.validate_python(value)

            except Exception as e:
                failures.append(f"Entity {entity_id} ({label}) falló validación:\n{str(e)}")

        if failures:
            error_count = len(failures)
            formatted_errors = "\n" + "=" * 40 + "\n".join(failures)
            pytest.fail(f"Se encontraron {error_count} entidades invalidas: {formatted_errors}")

    # @pytest.mark.parametrize("entity_data", load_entities())
    # def test_individual_entity_schema(self, entity_data):
    #     """
    #     Tests one entity at a time. Pytest will generate a
    #     separate test case for every item in the list.
    #     """
    #     entity_id = entity_data.get("id")
    #     label = entity_data.get("label")
    #     value = entity_data.get("value")

    #     # 1. Check label exists
    #     assert label is not None, f"Entity {entity_id} missing label"

    #     # 2. Get Class
    #     EntityClass = GraphSchema.get_entity_class(label)

    #     # 3. Validate
    #     EntityClass(id=entity_id, value=value)
    #     if label in VALUE_SCHEMAS:
    #         schema_adapter = VALUE_SCHEMAS[label]
    #         try:
    #             schema_adapter.validate_python(value)
    #         except ValidationError as e:
    #             pytest.fail(f"Entity {entity_id} ({label}) falló validación de campo value:\n{e}")


class TestRelationships:
    """Test relationship classes and factories."""

    def test_participo_en_relationship(self):
        """Test PARTICIPO_EN relationship creation."""
        rel = PARTICIPO_EN("inv_1", "proj_1", {"rol": "director"})
        assert rel.type == "PARTICIPO_EN"
        assert rel.source_id == "inv_1"
        assert rel.target_id == "proj_1"
        assert rel.properties["rol"] == "director"

    def test_tiene_topico_relationship(self):
        """Test TIENE_TOPICO relationship creation."""
        rel = TIENE_TOPICO("proj_1", "topic_1")
        assert rel.type == "TIENE_TOPICO"
        assert rel.source_id == "proj_1"
        assert rel.target_id == "topic_1"

    def test_es_descrito_por_relationship(self):
        """Test ES_DESCRITO_POR relationship creation."""
        rel = ES_DESCRITO_POR("proj_1", "doc_1")
        assert rel.type == "ES_DESCRITO_POR"
        assert rel.source_id == "proj_1"
        assert rel.target_id == "doc_1"

    def test_inicio_en_relationship(self):
        """Test INICIO_EN relationship creation."""
        rel = INICIO_EN("proj_1", "2024")
        assert rel.type == "INICIO_EN"
        assert rel.source_id == "proj_1"
        assert rel.target_id == "2024"

    def test_primer_chunk_relationship(self):
        """Test PRIMER_CHUNK relationship creation."""
        rel = PRIMER_CHUNK("doc_1", "chunk_1")
        assert rel.type == "PRIMER_CHUNK"
        assert rel.source_id == "doc_1"
        assert rel.target_id == "chunk_1"

    def test_siguiente_chunk_relationship(self):
        """Test SIGUIENTE_CHUNK relationship creation."""
        rel = SIGUIENTE_CHUNK("chunk_1", "chunk_2", {"orden": 1})
        assert rel.type == "SIGUIENTE_CHUNK"
        assert rel.source_id == "chunk_1"
        assert rel.target_id == "chunk_2"
        assert rel.properties["orden"] == 1

    def test_de_documento_relationship(self):
        """Test DE_DOCUMENTO relationship creation."""
        rel = DE_DOCUMENTO("chunk_1", "doc_1", {"pagina": 5})
        assert rel.type == "DE_DOCUMENTO"
        assert rel.source_id == "chunk_1"
        assert rel.target_id == "doc_1"
        assert rel.properties["pagina"] == 5

    def test_evidencia_de_relationship(self):
        """Test EVIDENCIA_DE relationship creation."""
        rel = EVIDENCIA_DE("chunk_1", "proj_1", {"confianza": 0.95})
        assert rel.type == "EVIDENCIA_DE"
        assert rel.source_id == "chunk_1"
        assert rel.target_id == "proj_1"
        assert rel.properties["confianza"] == 0.95

    def test_relationship_without_source_raises_error(self):
        """Test that creating a relationship without source_id raises ValueError."""
        with pytest.raises(ValueError):
            Relationship(type="PARTICIPO_EN", source_id="", target_id="proj_1")

    def test_relationship_without_target_raises_error(self):
        """Test that creating a relationship without target_id raises ValueError."""
        with pytest.raises(ValueError):
            Relationship(type="PARTICIPO_EN", source_id="inv_1", target_id="")

    def test_relationship_to_dict(self):
        """Test converting relationship to dictionary."""
        rel = PARTICIPO_EN("inv_1", "proj_1", {"rol": "director"})
        result = rel.to_dict()
        assert result["type"] == "PARTICIPO_EN"
        assert result["source_id"] == "inv_1"
        assert result["target_id"] == "proj_1"
        assert result["properties"]["rol"] == "director"


class TestGraphSchema:
    """Test the GraphSchema registry."""

    def test_list_entities(self):
        """Test listing all entity types."""
        entities = GraphSchema.list_entities()
        assert "Proyecto" in entities
        assert "Investigador" in entities
        assert "Topico" in entities
        assert "Documento" in entities
        assert "Chunk" in entities
        assert "Anio" in entities

    def test_list_relationships(self):
        """Test listing all relationship types."""
        relationships = GraphSchema.list_relationships()
        assert "PARTICIPO_EN" in relationships
        assert "TIENE_TOPICO" in relationships
        assert "ES_DESCRITO_POR" in relationships
        assert "INICIO_EN" in relationships
        assert "PRIMER_CHUNK" in relationships
        assert "SIGUIENTE_CHUNK" in relationships
        assert "DE_DOCUMENTO" in relationships
        assert "EVIDENCIA_DE" in relationships

    def test_get_entity_class(self):
        """Test getting entity class by name."""
        ProyectoClass = GraphSchema.get_entity_class("Proyecto")
        assert ProyectoClass == Proyecto

    def test_get_entity_class_unknown_type(self):
        """Test that getting unknown entity type raises ValueError."""
        with pytest.raises(ValueError):
            GraphSchema.get_entity_class("UnknownEntity")

    def test_get_relationship_factory(self):
        """Test getting relationship factory by type."""
        factory = GraphSchema.get_relationship_factory("PARTICIPO_EN")
        assert factory == PARTICIPO_EN

    def test_get_relationship_factory_unknown_type(self):
        """Test that getting unknown relationship type raises ValueError."""
        with pytest.raises(ValueError):
            GraphSchema.get_relationship_factory("INVALID_TYPE")


class TestValidation:
    """Test relationship validation."""

    def test_validate_participo_en(self):
        """Test validating PARTICIPO_EN relationship."""
        investigador = Investigador(id="inv_1", value="Dr. Smith")
        proyecto = Proyecto(id="proj_1", value="Proyecto A")
        rel = PARTICIPO_EN("inv_1", "proj_1")

        assert validate_relationship_endpoints(rel, investigador, proyecto) is True

    def test_validate_tiene_topico(self):
        """Test validating TIENE_TOPICO relationship."""
        proyecto = Proyecto(id="proj_1", value="Proyecto A")
        topico = Topico(id="topic_1", value="IA")
        rel = TIENE_TOPICO("proj_1", "topic_1")

        assert validate_relationship_endpoints(rel, proyecto, topico) is True

    def test_validate_es_descrito_por(self):
        """Test validating ES_DESCRITO_POR relationship."""
        proyecto = Proyecto(id="proj_1", value="Proyecto A")
        documento = Documento(id="doc_1", value="Propuesta.pdf")
        rel = ES_DESCRITO_POR("proj_1", "doc_1")

        assert validate_relationship_endpoints(rel, proyecto, documento) is True

    def test_validate_inicio_en(self):
        """Test validating INICIO_EN relationship."""
        proyecto = Proyecto(id="proj_1", value="Proyecto A")
        anio = Anio(id="2024", value=2024)
        rel = INICIO_EN("proj_1", "2024")

        assert validate_relationship_endpoints(rel, proyecto, anio) is True

    def test_validate_primer_chunk(self):
        """Test validating PRIMER_CHUNK relationship."""
        documento = Documento(id="doc_1", value="Propuesta.pdf")
        chunk = Chunk(id="chunk_1", value="Primer párrafo...")
        rel = PRIMER_CHUNK("doc_1", "chunk_1")

        assert validate_relationship_endpoints(rel, documento, chunk) is True

    def test_validate_siguiente_chunk(self):
        """Test validating SIGUIENTE_CHUNK relationship."""
        chunk1 = Chunk(id="chunk_1", value="Primer párrafo...")
        chunk2 = Chunk(id="chunk_2", value="Segundo párrafo...")
        rel = SIGUIENTE_CHUNK("chunk_1", "chunk_2")

        assert validate_relationship_endpoints(rel, chunk1, chunk2) is True

    def test_validate_de_documento(self):
        """Test validating DE_DOCUMENTO relationship."""
        chunk = Chunk(id="chunk_1", value="Primer párrafo...")
        documento = Documento(id="doc_1", value="Propuesta.pdf")
        rel = DE_DOCUMENTO("chunk_1", "doc_1")

        assert validate_relationship_endpoints(rel, chunk, documento) is True

    def test_validate_evidencia_de_proyecto(self):
        """Test validating EVIDENCIA_DE relationship with Proyecto."""
        chunk = Chunk(id="chunk_1", value="Texto que menciona el proyecto...")
        proyecto = Proyecto(id="proj_1", value="Proyecto A")
        rel = EVIDENCIA_DE("chunk_1", "proj_1")

        assert validate_relationship_endpoints(rel, chunk, proyecto) is True

    def test_validate_evidencia_de_topico(self):
        """Test validating EVIDENCIA_DE relationship with Topico."""
        chunk = Chunk(id="chunk_1", value="Texto sobre el tópico...")
        topico = Topico(id="topic_1", value="IA")
        rel = EVIDENCIA_DE("chunk_1", "topic_1")

        assert validate_relationship_endpoints(rel, chunk, topico) is True

    def test_validate_evidencia_de_investigador(self):
        """Test validating EVIDENCIA_DE relationship with Investigador."""
        chunk = Chunk(id="chunk_1", value="Texto que menciona al investigador...")
        investigador = Investigador(id="inv_1", value="Dr. Smith")
        rel = EVIDENCIA_DE("chunk_1", "inv_1")

        assert validate_relationship_endpoints(rel, chunk, investigador) is True

    def test_validate_invalid_relationship(self):
        """Test validating relationship with wrong entity types."""
        # Try to create PARTICIPO_EN with wrong entity types
        topico = Topico(id="topic_1", value="IA")
        documento = Documento(id="doc_1", value="Doc.pdf")
        rel = PARTICIPO_EN("topic_1", "doc_1")  # Wrong: should be Investigador -> Proyecto

        assert validate_relationship_endpoints(rel, topico, documento) is False
