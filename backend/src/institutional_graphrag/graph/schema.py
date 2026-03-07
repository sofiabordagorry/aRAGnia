"""
Esquema de Grafo para Institutional GraphRAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
from typing_extensions import NotRequired, TypedDict

# Definiciones del campo Value


class AnioValue(TypedDict):
    year: str


class DocumentoValue(TypedDict):
    base_name: str
    is_group: str
    year_publisher: str
    sub_id: NotRequired[str]
    type: Literal["informe", "propuesta", "resumen", "tabla"]


class InvestigadorValue(TypedDict):
    name: str
    source: Literal["static", "llm"]


@dataclass
class Entity:
    """Clase base para entidades del grafo (nodos)."""

    id: str
    """Identificador único de la entidad."""

    value: Any
    """Valor de la entidad."""

    def __post_init__(self):
        if not self.id:
            raise ValueError(f"{self.__class__.__name__} debe tener un id no vacío")

    @property
    def label(self) -> str:
        return self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "value": self.value,
        }


@dataclass
class Relationship:
    """Clase base para relaciones del grafo (aristas)."""

    type: str
    """Tipo de la relación."""

    source_id: str
    """ID de la entidad origen."""

    target_id: str
    """ID de la entidad destino."""

    properties: Dict[str, Any] = field(default_factory=dict)
    """Propiedades adicionales asociadas a la relación."""

    def __post_init__(self):
        if not self.source_id or not self.target_id:
            raise ValueError("La relación debe tener source_id y target_id no vacíos")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "properties": self.properties,
        }


@dataclass
class Proyecto(Entity):
    value: str


@dataclass
class Anio(Entity):
    value: AnioValue


@dataclass
class Investigador(Entity):
    value: InvestigadorValue


@dataclass
class Topico(Entity):
    value: str


@dataclass
class Documento(Entity):
    value: DocumentoValue


@dataclass
class Chunk(Entity):
    pass


def PARTICIPO_EN(
    investigador_id: str, proyecto_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación PARTICIPO_EN.

    (Investigador)-[PARTICIPO_EN]->(Proyecto)
    """
    return Relationship(
        type="PARTICIPO_EN",
        source_id=investigador_id,
        target_id=proyecto_id,
        properties=properties or {},
    )


def TIENE_TOPICO(
    proyecto_id: str, topico_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación TIENE_TOPICO.

    (Proyecto)-[TIENE_TOPICO]->(Topico)
    """
    return Relationship(
        type="TIENE_TOPICO",
        source_id=proyecto_id,
        target_id=topico_id,
        properties=properties or {},
    )


def ES_DESCRITO_POR(
    proyecto_id: str, documento_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación ES_DESCRITO_POR.

    (Proyecto)-[ES_DESCRITO_POR]->(Documento)
    """
    return Relationship(
        type="ES_DESCRITO_POR",
        source_id=proyecto_id,
        target_id=documento_id,
        properties=properties or {},
    )


def INICIO_EN(
    proyecto_id: str, anio_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación INICIO_EN.

    (Proyecto)-[INICIO_EN]->(Anio)
    """
    return Relationship(
        type="INICIO_EN",
        source_id=proyecto_id,
        target_id=anio_id,
        properties=properties or {},
    )


def PRIMER_CHUNK(
    documento_id: str, chunk_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación PRIMER_CHUNK.

    (Documento)-[PRIMER_CHUNK]->(Chunk)
    """
    return Relationship(
        type="PRIMER_CHUNK",
        source_id=documento_id,
        target_id=chunk_id,
        properties=properties or {},
    )


def SIGUIENTE_CHUNK(
    chunk_id_from: str, chunk_id_to: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación SIGUIENTE_CHUNK.

    (Chunk)-[SIGUIENTE_CHUNK]->(Chunk)
    """
    return Relationship(
        type="SIGUIENTE_CHUNK",
        source_id=chunk_id_from,
        target_id=chunk_id_to,
        properties=properties or {},
    )


def DE_DOCUMENTO(
    chunk_id: str, documento_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación DE_DOCUMENTO.

    (Chunk)-[DE_DOCUMENTO]->(Documento)
    """
    return Relationship(
        type="DE_DOCUMENTO",
        source_id=chunk_id,
        target_id=documento_id,
        properties=properties or {},
    )


def EVIDENCIA_DE(
    chunk_id: str, entity_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación EVIDENCIA_DE.

    (Chunk)-[EVIDENCIA_DE]->(Proyecto|Topico|Investigador)
    """
    return Relationship(
        type="EVIDENCIA_DE",
        source_id=chunk_id,
        target_id=entity_id,
        properties=properties or {},
    )

def POTENCIAL_IGUALDAD(
    investigador_id: str, investigador2_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación POTENCIAL_IGUALDAD.

    (Investigador)-[POTENCIAL_IGUALDAD]->(PInvestigador)
    """
    return Relationship(
        type="POTENCIAL_IGUALDAD",
        source_id=investigador_id,
        target_id=investigador2_id,
        properties=properties or {},
    )

class GraphSchema:
    # Tipos de entidades
    ENTITIES = {
        "Proyecto": Proyecto,
        "Anio": Anio,
        "Investigador": Investigador,
        "Topico": Topico,
        "Documento": Documento,
        "Chunk": Chunk,
    }

    # Tipos de relaciones
    RELATIONSHIPS = {
        "PARTICIPO_EN": PARTICIPO_EN,
        "TIENE_TOPICO": TIENE_TOPICO,
        "ES_DESCRITO_POR": ES_DESCRITO_POR,
        "INICIO_EN": INICIO_EN,
        "PRIMER_CHUNK": PRIMER_CHUNK,
        "SIGUIENTE_CHUNK": SIGUIENTE_CHUNK,
        "DE_DOCUMENTO": DE_DOCUMENTO,
        "EVIDENCIA_DE": EVIDENCIA_DE,
        "POTENCIAL_IGUALDAD": POTENCIAL_IGUALDAD
    }

    @classmethod
    def get_entity_class(cls, entity_type: str) -> type[Entity]:
        """
        Obtener clase de entidad por nombre de tipo.
        """
        if entity_type not in cls.ENTITIES:
            raise ValueError(f"Tipo de entidad desconocido: {entity_type}")
        return cls.ENTITIES[entity_type]

    @classmethod
    def get_relationship_factory(cls, rel_type: str):
        """
        Obtener función factory de relación por tipo.
        """
        if rel_type not in cls.RELATIONSHIPS:
            raise ValueError(f"Tipo de relación desconocido: {rel_type}")
        return cls.RELATIONSHIPS[rel_type]

    @classmethod
    def list_entities(cls) -> List[str]:
        """Listar todos los tipos de entidades disponibles."""
        return list(cls.ENTITIES.keys())

    @classmethod
    def list_relationships(cls) -> List[str]:
        """Listar todos los tipos de relaciones disponibles."""
        return list(cls.RELATIONSHIPS.keys())


def validate_relationship_endpoints(
    relationship: Relationship, source_entity: Entity, target_entity: Entity
) -> bool:
    """
    Validar que una relación tiene los tipos de entidad origen y destino correctos.
    """
    # Definir combinaciones válidas
    valid_combinations = {
        "PARTICIPO_EN": ("Investigador", "Proyecto"),
        "TIENE_TOPICO": ("Proyecto", "Topico"),
        "ES_DESCRITO_POR": ("Proyecto", "Documento"),
        "INICIO_EN": ("Proyecto", "Anio"),
        "PRIMER_CHUNK": ("Documento", "Chunk"),
        "SIGUIENTE_CHUNK": ("Chunk", "Chunk"),
        "DE_DOCUMENTO": ("Chunk", "Documento"),
        "EVIDENCIA_DE": ("Chunk", ["Proyecto", "Topico", "Investigador"]),
        "POTENCIAL_IGUALDAD":("Investigador", "Investigador"),
    }

    expected = valid_combinations.get(relationship.type)
    if not expected:
        return False

    source_label = source_entity.label
    target_label = target_entity.label

    if relationship.type == "EVIDENCIA_DE":
        return source_label == expected[0] and target_label in expected[1]

    return source_label == expected[0] and target_label == expected[1]
