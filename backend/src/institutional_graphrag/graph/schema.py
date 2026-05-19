"""
Esquema de Grafo para Institutional GraphRAG.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

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
    display_name: str
    documento: str
    tipo_documento: str
    pais_documento: str
    sexo: str


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
class Grupo(Entity):
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
class Dominio(Entity):
    value: str


@dataclass
class Area(Entity):
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


def PERTENECE_A_DOMINIO(
    topico_id: str, dominio_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación PERTENECE_A_DOMINIO.

    (Topico)-[PERTENECE_A_DOMINIO]->(Dominio)
    """
    return Relationship(
        type="PERTENECE_A_DOMINIO",
        source_id=topico_id,
        target_id=dominio_id,
        properties=properties or {},
    )


def PERTENECE_A_AREA(
    proyecto_id: str, area_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación PERTENECE_A_AREA.

    (Proyecto)-[PERTENECE_A_AREA]->(Area)
    """
    return Relationship(
        type="PERTENECE_A_AREA",
        source_id=proyecto_id,
        target_id=area_id,
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


def EXTRAIDO_DE(
    chunk_id: str, entity_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación EXTRAIDO_DE.

    (Chunk)-[EXTRAIDO_DE]->(Topico|Investigador)
    """
    return Relationship(
        type="EXTRAIDO_DE",
        source_id=chunk_id,
        target_id=entity_id,
        properties=properties or {},
    )


def TITULO_EXTRAIDO_DE(
    proyect_id: str, chunk_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación TITULO_EXTRAIDO_DE.

    (Proyecto)-[TITULO_EXTRAIDO_DE]->(Chunk)
    """
    return Relationship(
        type="TITULO_EXTRAIDO_DE",
        source_id=proyect_id,
        target_id=chunk_id,
        properties=properties or {},
    )


class GraphSchema:
    # Tipos de entidades
    ENTITIES = {
        "Proyecto": Proyecto,
        "Grupo": Grupo,
        "Anio": Anio,
        "Investigador": Investigador,
        "Topico": Topico,
        "Dominio": Dominio,
        "Area": Area,
        "Documento": Documento,
        "Chunk": Chunk,
    }

    # Tipos de relaciones
    RELATIONSHIPS = {
        "PARTICIPO_EN": PARTICIPO_EN,
        "TIENE_TOPICO": TIENE_TOPICO,
        "PERTENECE_A_DOMINIO": PERTENECE_A_DOMINIO,
        "PERTENECE_A_AREA": PERTENECE_A_AREA,
        "ES_DESCRITO_POR": ES_DESCRITO_POR,
        "INICIO_EN": INICIO_EN,
        "PRIMER_CHUNK": PRIMER_CHUNK,
        "SIGUIENTE_CHUNK": SIGUIENTE_CHUNK,
        "DE_DOCUMENTO": DE_DOCUMENTO,
        "EXTRAIDO_DE": EXTRAIDO_DE,
        "TITULO_EXTRAIDO_DE": TITULO_EXTRAIDO_DE,
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
    PROJECT_TYPES = ("Proyecto", "Grupo")

    valid_combinations: Dict[str, tuple] = {
        "PARTICIPO_EN": ("Investigador", PROJECT_TYPES),
        "TIENE_TOPICO": (PROJECT_TYPES, "Topico"),
        "PERTENECE_A_DOMINIO": ("Topico", "Dominio"),
        "PERTENECE_A_AREA": ("Proyecto", "Area"),
        "ES_DESCRITO_POR": (PROJECT_TYPES, "Documento"),
        "INICIO_EN": (PROJECT_TYPES, "Anio"),
        "PRIMER_CHUNK": ("Documento", "Chunk"),
        "SIGUIENTE_CHUNK": ("Chunk", "Chunk"),
        "DE_DOCUMENTO": ("Chunk", "Documento"),
        "EXTRAIDO_DE": ("Chunk", ["Topico", "Investigador"]),
        "TITULO_EXTRAIDO_DE": (PROJECT_TYPES, "Chunk"),
    }

    expected = valid_combinations.get(relationship.type)
    if not expected:
        return False

    source_label = source_entity.label
    target_label = target_entity.label

    expected_source, expected_target = expected

    source_ok = (
        source_label in expected_source
        if isinstance(expected_source, tuple)
        else source_label == expected_source
    )
    target_ok = (
        target_label in expected_target
        if isinstance(expected_target, tuple)
        else target_label == expected_target
    )

    return source_ok and target_ok
