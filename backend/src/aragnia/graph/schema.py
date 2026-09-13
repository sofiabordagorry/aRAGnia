"""
Esquema de Grafo para aRAGnia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, get_args, get_origin, get_type_hints

from typing_extensions import NotRequired, Required, TypedDict


class AnioValue(TypedDict):
    anio: str


class DocumentoValue(TypedDict):
    nombre_base: str
    es_grupo: str
    anio_publicacion: str
    sub_id: NotRequired[str]
    tipo: Literal["informe", "propuesta", "resumen", "tabla"]


class InvestigadorValue(TypedDict):
    nombre: str
    nombre_de_despliegue: str
    documento: NotRequired[str]
    tipo_documento: NotRequired[str]
    pais_documento: NotRequired[str]
    sexo: NotRequired[str]


class FileValue(TypedDict):
    titulo: str
    titulo_de_despliegue: str
    palabras_clave: NotRequired[List[str]]
    descripcion: NotRequired[str]


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
    value: FileValue


@dataclass
class Grupo(Entity):
    value: FileValue


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
class Subcampo(Entity):
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


def PERTENECE_A_SUBCAMPO(
    topico_id: str, subcampo_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación PERTENECE_A_SUBCAMPO.

    (Topico)-[PERTENECE_A_SUBCAMPO]->(Subcampo)
    """
    return Relationship(
        type="PERTENECE_A_SUBCAMPO",
        source_id=topico_id,
        target_id=subcampo_id,
        properties=properties or {},
    )


def PERTENECE_A_AREA(
    proyecto_id: str, area_id: str, properties: Optional[Dict[str, Any]] = None
) -> Relationship:
    """
    Crear una relación PERTENECE_A_AREA.

    (Proyecto|Grupo)-[PERTENECE_A_AREA]->(Area)
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
    ENTITIES = {
        "Proyecto": Proyecto,
        "Grupo": Grupo,
        "Anio": Anio,
        "Investigador": Investigador,
        "Topico": Topico,
        "Subcampo": Subcampo,
        "Area": Area,
        "Documento": Documento,
        "Chunk": Chunk,
    }

    RELATIONSHIPS = {
        "PARTICIPO_EN": PARTICIPO_EN,
        "TIENE_TOPICO": TIENE_TOPICO,
        "PERTENECE_A_SUBCAMPO": PERTENECE_A_SUBCAMPO,
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
        "PERTENECE_A_SUBCAMPO": ("Topico", "Subcampo"),
        "PERTENECE_A_AREA": (PROJECT_TYPES, "Area"),
        "ES_DESCRITO_POR": (PROJECT_TYPES, "Documento"),
        "INICIO_EN": (PROJECT_TYPES, "Anio"),
        "PRIMER_CHUNK": ("Documento", "Chunk"),
        "SIGUIENTE_CHUNK": ("Chunk", "Chunk"),
        "DE_DOCUMENTO": ("Chunk", "Documento"),
        "EXTRAIDO_DE": ("Chunk", ("Topico", "Investigador")),
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


# Esquema de las propiedades de cada entidad. El tipo y la obligatoriedad se
# toman de los TypedDict declarados arriba, que son la especificación del modelo.
ENTITY_VALUE_SCHEMAS: Dict[str, Any] = {
    "Proyecto": FileValue,
    "Grupo": FileValue,
    "Anio": AnioValue,
    "Investigador": InvestigadorValue,
    "Documento": DocumentoValue,
}

# Entidades cuyo valor es un único dato escalar (se persiste como propiedad "valor").
SCALAR_VALUE_ENTITIES = ("Topico", "Subcampo", "Area")

# Chunk es una forma abierta: además de las propiedades declaradas conserva la
# metadata que devuelve el parser de documentos, que varía según el documento.
CHUNK_PROPERTY_TYPES: Dict[str, Any] = {"texto": str, "paginas": List[int]}
CHUNK_REQUIRED_PROPERTIES = ("texto",)


def _format_type(expected: Any) -> str:
    origin = get_origin(expected)
    if origin is Literal:
        return " | ".join(repr(arg) for arg in get_args(expected))
    if origin is list:
        args = get_args(expected)
        return f"list[{_format_type(args[0])}]" if args else "list"
    name = getattr(expected, "__name__", None)
    return str(name) if name is not None else str(expected)


def _matches_type(value: Any, expected: Any) -> bool:
    origin = get_origin(expected)
    if origin is Literal:
        return value in get_args(expected)
    if origin is list:
        if not isinstance(value, list):
            return False
        args = get_args(expected)
        return all(_matches_type(item, args[0]) for item in value) if args else True
    if expected is Any:
        return True
    if isinstance(expected, type):
        if expected is int and isinstance(value, bool):
            return False
        return isinstance(value, expected)
    return True


def _validate_properties(
    label: str,
    value: Any,
    property_types: Dict[str, Any],
    required: tuple[str, ...],
    closed: bool,
) -> List[str]:
    if not isinstance(value, dict):
        return [f"{label}: se esperaba un diccionario de propiedades"]

    violations: List[str] = []

    for prop in sorted(required):
        if value.get(prop) is None:
            violations.append(f"{label}.{prop}: falta una propiedad obligatoria")

    for prop, prop_value in value.items():
        expected = property_types.get(prop)
        if expected is None:
            if closed:
                violations.append(f"{label}.{prop}: propiedad no declarada en el esquema")
            continue
        if prop_value is None:
            continue
        if not _matches_type(prop_value, expected):
            violations.append(
                f"{label}.{prop}: se esperaba {_format_type(expected)} "
                f"y se recibió {type(prop_value).__name__}"
            )

    return violations


def _resolve_property_types(value_schema: Any) -> tuple[Dict[str, Any], tuple[str, ...]]:
    """
    Obtener el tipo de cada propiedad y cuáles son obligatorias.

    No se usa __required_keys__ porque el módulo pospone la evaluación de
    anotaciones (PEP 563) y en ese caso NotRequired no queda registrado ahí.
    """
    property_types: Dict[str, Any] = {}
    required: List[str] = []

    for prop, hint in get_type_hints(value_schema, include_extras=True).items():
        origin = get_origin(hint)
        if origin is NotRequired:
            property_types[prop] = get_args(hint)[0]
        elif origin is Required:
            property_types[prop] = get_args(hint)[0]
            required.append(prop)
        else:
            property_types[prop] = hint
            required.append(prop)

    return property_types, tuple(required)


def validate_entity(entity: Entity) -> List[str]:
    """
    Verificar que una entidad cumpla lo declarado en el esquema.

    Devuelve la lista de incumplimientos encontrados, vacía si la entidad cumple.
    """
    label = entity.label

    if label in SCALAR_VALUE_ENTITIES:
        if not isinstance(entity.value, str) or not entity.value.strip():
            return [f"{label}.valor: se esperaba una cadena no vacía"]
        return []

    if label == "Chunk":
        return _validate_properties(
            label,
            entity.value,
            CHUNK_PROPERTY_TYPES,
            CHUNK_REQUIRED_PROPERTIES,
            closed=False,
        )

    value_schema = ENTITY_VALUE_SCHEMAS.get(label)
    if value_schema is None:
        return [f"Tipo de entidad desconocido: {label}"]

    property_types, required = _resolve_property_types(value_schema)
    return _validate_properties(label, entity.value, property_types, required, closed=True)


@dataclass(frozen=True)
class RelationshipCardinality:
    """
    Cardinalidad declarada para una relación, contada desde la entidad de origen.

    Un máximo en None representa la cota superior abierta (*) de la notación del esquema.
    """

    source: str
    type: str
    targets: tuple[str, ...]
    minimum: int
    maximum: Optional[int] = None

    def __str__(self) -> str:
        maximum = "*" if self.maximum is None else str(self.maximum)
        targets = "|".join(self.targets)
        return f"({self.source})-[:{self.type}]->({targets}) [{self.minimum}..{maximum}]"

    @property
    def is_unrestricted(self) -> bool:
        """Indicar si la cardinalidad admite cualquier cantidad de relaciones ([0..*])."""
        return self.minimum == 0 and self.maximum is None


# Cardinalidades del esquema del grafo, transcritas del diagrama de la sección de diseño.
RELATIONSHIP_CARDINALITIES: tuple[RelationshipCardinality, ...] = (
    RelationshipCardinality("Proyecto", "ES_DESCRITO_POR", ("Documento",), 1, None),
    RelationshipCardinality("Grupo", "ES_DESCRITO_POR", ("Documento",), 1, None),
    RelationshipCardinality("Proyecto", "INICIO_EN", ("Anio",), 1, 1),
    RelationshipCardinality("Grupo", "INICIO_EN", ("Anio",), 1, 1),
    RelationshipCardinality("Proyecto", "PERTENECE_A_AREA", ("Area",), 0, 1),
    RelationshipCardinality("Grupo", "PERTENECE_A_AREA", ("Area",), 0, 1),
    RelationshipCardinality("Proyecto", "TIENE_TOPICO", ("Topico",), 1, None),
    RelationshipCardinality("Grupo", "TIENE_TOPICO", ("Topico",), 1, None),
    RelationshipCardinality("Proyecto", "TITULO_EXTRAIDO_DE", ("Chunk",), 0, None),
    RelationshipCardinality("Grupo", "TITULO_EXTRAIDO_DE", ("Chunk",), 0, None),
    # Un investigador participa al menos de un proyecto o de un grupo: la cardinalidad
    # se cuenta sobre el conjunto de ambos, no sobre cada uno por separado.
    RelationshipCardinality("Investigador", "PARTICIPO_EN", ("Proyecto", "Grupo"), 1, None),
    # Un nodo Topico se identifica por el nombre del tópico, por lo que puede agrupar
    # tópicos de OpenAlex con la misma denominación en español, provenientes de
    # subcampos distintos.
    RelationshipCardinality("Topico", "PERTENECE_A_SUBCAMPO", ("Subcampo",), 1, None),
    RelationshipCardinality("Documento", "PRIMER_CHUNK", ("Chunk",), 1, 1),
    RelationshipCardinality("Chunk", "DE_DOCUMENTO", ("Documento",), 1, 1),
    RelationshipCardinality("Chunk", "SIGUIENTE_CHUNK", ("Chunk",), 0, 1),
    RelationshipCardinality("Chunk", "EXTRAIDO_DE", ("Topico", "Investigador"), 0, None),
)
