"""GraphRAG retriever usando queries Cypher sobre Neo4j."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase
from neo4j.graph import Node

logger = logging.getLogger(__name__)


@dataclass
class GraphRAGChunk:
    """Chunk de contexto obtenido del grafo."""

    chunk_id: str
    text: str


@dataclass
class GraphRAGResult:
    """Resultado de GraphRAG."""

    answer: str
    chunks: List[GraphRAGChunk]
    cypher_query: str
    chunk_to_entities: Dict[
        str, List[tuple[str, str]]
    ]  # Mapeo chunk_id -> [(entity_id, entity_label)] - TRAZABILIDAD COMPLETA


class CypherQueryValidator:
    """Validador de queries Cypher para seguridad."""

    # Palabras clave peligrosas que no deben aparecer
    FORBIDDEN_KEYWORDS = [
        r"\bDELETE\b",
        r"\bREMOVE\b",
        r"\bSET\b",
        r"\bCREATE\b",
        r"\bMERGE\b",
        r"\bDROP\b",
        r"\bDETACH\b",
    ]

    @classmethod
    def is_safe(cls, query: str) -> tuple[bool, Optional[str]]:
        """Valida que la query sea segura (solo lectura)."""
        if not query or not query.strip():
            return False, "Query vacía"

        query_upper = query.upper()

        # Verificar palabras clave prohibidas
        for pattern in cls.FORBIDDEN_KEYWORDS:
            if re.search(pattern, query_upper):
                return False, f"Query contiene operación prohibida: {pattern}"

        # Debe contener MATCH o RETURN
        if "MATCH" not in query_upper and "RETURN" not in query_upper:
            return False, "Query debe contener MATCH o RETURN"

        return True, None


class GraphRAGRetriever:
    """Retriever que usa queries Cypher para obtener subgrafos."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        from institutional_graphrag.rag.generate import get_llm_client

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        # LLM para clasificación y Cypher
        self.cypher_llm_client = get_llm_client(provider=llm_provider, model="qwen2.5:3b-instruct")
        # LLM para respuestas finales
        self.answer_llm_client = get_llm_client(provider=llm_provider, model="llama3.2:3b")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def close(self):
        """Cerrar conexión a Neo4j."""
        self.driver.close()

    def _classify_query_intent(self, user_query: str) -> str:
        """
        Clasifica si la consulta del usuario necesita búsqueda en grafo o es conversacional.
        """
        prompt = f"""Classify if this user query needs a knowledge graph search or is just conversational.

Examples:
- "hola" -> CHAT
- "gracias" -> CHAT
- "qué tal?" -> CHAT
- "buenos días" -> CHAT
- "qué proyectos de biotecnología?" -> SEARCH
- "quién trabajó en ingeniería?" -> SEARCH
- "cuántos proyectos hay?" -> SEARCH
- "dime sobre gi_2014_133" -> SEARCH

Query: "{user_query}"

Classification (answer only SEARCH or CHAT):"""

        messages = [
            {"role": "system", "content": "You are a classifier. Answer only with SEARCH or CHAT."},
            {"role": "user", "content": prompt},
        ]

        response = (
            self.cypher_llm_client.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=10,
            )
            .strip()
            .upper()
        )

        # Parse response - debe ser SEARCH o CHAT
        if "SEARCH" in response:
            return "SEARCH"
        elif "CHAT" in response:
            return "CHAT"
        else:
            # Default a CHAT para evitar errores de Cypher con queries ambiguas
            logger.warning(f"Intent classification unclear: '{response}', defaulting to CHAT")
            return "CHAT"

    def _generate_conversational_response(self, user_query: str) -> str:
        """
        Genera respuesta conversacional amigable para queries tipo chat.
        """
        system_prompt = """Eres un asistente amigable de consultas académicas. Responde de forma breve, amigable y profesional en español.

Si te preguntan qué puedes hacer, explica que puedes buscar información sobre proyectos de investigación, investigadores, tópicos y documentos académicos."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ]

        response = self.answer_llm_client.generate(
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        logger.info("Respuesta conversacional generada")

        return response.strip()

    def generate_cypher_query(self, user_query: str) -> str:
        """
        Genera una query Cypher usando LLM a partir de la pregunta del usuario.
        """
        prompt = self._build_cypher_generation_prompt(user_query)

        messages = [
            {
                "role": "system",
                "content": "You are an expert in Neo4j and Cypher. Generate precise, safe, read-only queries. Always return chunks with entities. Connect all MATCH patterns via shared variables. NO aggregations.",
            },
            {"role": "user", "content": prompt},
        ]

        response = self.cypher_llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Extraer query entre tags <QUERY>
        query_match = re.search(r"<QUERY>(.*?)</QUERY>", response, re.DOTALL | re.IGNORECASE)

        if not query_match:
            raise ValueError("LLM no devolvió query entre tags <QUERY>...</QUERY>")

        cypher_query = query_match.group(1).strip()

        # Validar query
        is_safe, error = CypherQueryValidator.is_safe(cypher_query)
        if not is_safe:
            raise ValueError(f"Query generada no es segura: {error}")

        logger.info(f"Query Cypher generada: {cypher_query}")
        return cypher_query

    def _build_cypher_generation_prompt(self, user_query: str) -> str:
        """Construye el prompt para la generación de queries Cypher."""
        return f"""Generate a Cypher query for Neo4j to answer this question.

SCHEMA:
Nodes and their key properties:
- Proyecto     → id: 'gi_2014_133', value: 'Síntesis y evaluación biológica de nuevos quimioterápicos'
- Investigador → id: 'lastname_firstname', name: 'Firstname Lastname'
- Topico       → value: 'Biotechnology'      ← property is "value", NOT "id"
- Documento    → id: '...', base_name: 'gi_2014_133', type: 'informe'|'propuesta'|'resumen'|'tabla', year_publisher: '2014', is_group: 'false'
- Chunk        → id: '...', text: '...'
- Anio         → year: '2014'                ← property is "year", NOT "value" or "id"

Relationships:
- (Investigador)-[:PARTICIPO_EN]->(Proyecto)
- (Proyecto)-[:TIENE_TOPICO]->(Topico)
- (Proyecto)-[:ES_DESCRITO_POR]->(Documento)
- (Proyecto)-[:INICIO_EN]->(Anio)
- (Documento)-[:PRIMER_CHUNK]->(Chunk)
- (Chunk)-[:SIGUIENTE_CHUNK]->(Chunk)
- (Chunk)-[:DE_DOCUMENTO]->(Documento)
- (Chunk)-[:EVIDENCIA_DE]->(Investigador|Topico|Proyecto)

RULES:
1. Read-only (MATCH, RETURN only)
2. ALWAYS return chunks (c:Chunk) — they contain the actual text evidence
3. Connect patterns: every MATCH must use variables defined in previous MATCHes
4. Use [:EVIDENCIA_DE] to navigate from entities to their evidence chunks
5. Topics are stored in ENGLISH: 'Biotechnology', 'Engineering', 'Medicine', etc.
6. LIMIT 20-30 results
7. NEVER define relationship variables — use anonymous patterns only: -[:TYPE]-> NOT -[r:TYPE]->
8. Node variables must be unique and never reused for a different type

PATTERNS (use what fits best):

List projects by topic:
MATCH (t:Topico {{value: 'Biotechnology'}})
MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)
MATCH (c:Chunk)-[:EVIDENCIA_DE]->(p)
RETURN p, COLLECT(c) AS chunks LIMIT 20

List investigators of a project:
MATCH (p:Proyecto {{id: 'gi_2014_133'}})
MATCH (i:Investigador)-[:PARTICIPO_EN]->(p)
MATCH (c:Chunk)-[:EVIDENCIA_DE]->(i)
RETURN i, COLLECT(c) AS chunks LIMIT 25

Chunks of a specific topic (only when asked about the topic itself, not its projects):
MATCH (t:Topico {{value: 'Biotechnology'}})
MATCH (c:Chunk)-[:EVIDENCIA_DE]->(t)
RETURN c, t LIMIT 25

By researcher name:
MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'lastname'
MATCH (c:Chunk)-[:EVIDENCIA_DE]->(i)
RETURN c, i LIMIT 25

By project ID (chunks about a specific project):
MATCH (p:Proyecto {{id: 'gi_2014_133'}})
MATCH (c:Chunk)-[:EVIDENCIA_DE]->(p)
RETURN c, p LIMIT 25

Documents of a project:
MATCH (p:Proyecto {{id: 'gi_2014_133'}})
MATCH (p)-[:ES_DESCRITO_POR]->(d:Documento)
MATCH (c:Chunk)-[:DE_DOCUMENTO]->(d)
RETURN d, p, COLLECT(c) AS chunks LIMIT 20

Describe / full info about a specific project (name, year, topics, investigators):
MATCH (p:Proyecto {{id: 'gi_2014_133'}})
OPTIONAL MATCH (c:Chunk)-[:EVIDENCIA_DE]->(p)
OPTIONAL MATCH (p)-[:TIENE_TOPICO]->(t:Topico)
OPTIONAL MATCH (inv:Investigador)-[:PARTICIPO_EN]->(p)
OPTIONAL MATCH (p)-[:INICIO_EN]->(a:Anio)
RETURN p, COLLECT(DISTINCT c) AS chunks, COLLECT(DISTINCT t) AS topics, COLLECT(DISTINCT inv) AS investigators, a LIMIT 1

Projects by year:
MATCH (a:Anio {{year: '2014'}})
MATCH (p:Proyecto)-[:INICIO_EN]->(a)
MATCH (c:Chunk)-[:EVIDENCIA_DE]->(p)
RETURN p, COLLECT(c) AS chunks LIMIT 20

Count entities (how many X are there):
MATCH (i:Investigador) RETURN count(i) AS total
MATCH (p:Proyecto) RETURN count(p) AS total
MATCH (t:Topico) RETURN count(t) AS total

QUESTION: {user_query}

CRITICAL:
- Wrap your query in <QUERY> and </QUERY> tags
- Every variable in WITH/RETURN must be defined in a previous MATCH
- Use [:EVIDENCIA_DE]->(entity) — never [:EVIDENCIA_DE]->(var1|var2)
- Topico uses {{value: '...'}}, Anio uses {{year: '...'}}, all others use {{id: '...'}}
- NEVER name a relationship variable (never write -[r:TYPE]-> or -[rel:TYPE]->), always use -[:TYPE]->
- NEVER use a variable as both a relationship and a node

<QUERY>
"""

    def execute_cypher_query(self, cypher_query: str) -> List[Any]:
        """
        Ejecuta una query Cypher y retorna resultados (como neo4j Record objects).
        """
        # Validar nuevamente antes de ejecutar
        is_safe, error = CypherQueryValidator.is_safe(cypher_query)
        if not is_safe:
            raise ValueError(f"Query no pasó validación de seguridad: {error}")

        with self.driver.session() as session:
            result = session.run(cypher_query)
            records = list(result)

        logger.info(f"Query ejecutada, {len(records)} registros obtenidos")
        return records

    def _is_aggregation_query(self, cypher_query: str) -> bool:
        """Detecta si una query es de agregación (usa COUNT, SUM, AVG, etc.)"""
        query_upper = cypher_query.upper()
        aggregation_functions = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN(", "COLLECT("]
        return any(func in query_upper for func in aggregation_functions)

    def _build_aggregation_context(self, records: List[Any]) -> str:
        """Construye contexto a partir de resultados de agregación o nodos sin chunks."""
        if not records:
            return "No aggregation results found."

        context_parts = ["=== QUERY RESULTS ===\n"]

        for idx, record in enumerate(records, 1):
            values = []
            for key in record.keys():
                value = record[key]
                if isinstance(value, Node):
                    labels = list(value.labels)
                    label = labels[0] if labels else "Node"
                    props = dict(value)
                    display = props.get("value") or props.get("name") or props.get("id", str(props))
                    values.append(f"{key} ({label}): {display}")
                else:
                    values.append(f"{key}: {value}")

            context_parts.append(f"{idx}. {', '.join(values)}")

            if idx >= 50:
                context_parts.append(f"\n[Note: {len(records) - idx} more results omitted...]")
                break

        return "\n".join(context_parts)

    def _extract_chunks_and_entities_from_results(
        self, records: List[Any]
    ) -> tuple[List[GraphRAGChunk], Dict[tuple[str, str], dict], Dict[str, List[tuple[str, str]]]]:
        """Extrae chunks y evidencia de entidades desde los resultados de Cypher."""
        chunks_dict: Dict[str, GraphRAGChunk] = {}  # chunk_id -> GraphRAGChunk
        evidence_entities: Dict[tuple[str, str], dict] = {}  # (entity_id, label) -> props
        chunk_to_entities: Dict[str, List[tuple[str, str]]] = {}  # chunk_id -> [(entity_id, label)]

        for record in records:
            # Recopilar chunks y entidades de este record
            chunks_in_record = []
            # direct = vienen de MATCH directo (implica relación EVIDENCIA_DE)
            entities_direct = []
            # collected = vienen de COLLECT() — son del proyecto/consulta, no del chunk
            entities_collected = []

            # Iterar sobre los valores del record
            for key in record.keys():
                value = record[key]

                # Verificar si es un nodo de Neo4j (directo, no en lista)
                if isinstance(value, Node):
                    labels = list(value.labels)
                    if "Chunk" in labels:
                        chunks_in_record.append(value)
                    elif any(
                        label in labels
                        for label in ["Investigador", "Topico", "Proyecto", "Documento", "Anio"]
                    ):
                        entities_direct.append(value)

                # Si es una lista (resultado de COLLECT), marcar como collected
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, Node):
                            labels = list(item.labels)
                            if "Chunk" in labels:
                                chunks_in_record.append(item)
                            elif any(
                                label in labels
                                for label in [
                                    "Investigador",
                                    "Topico",
                                    "Proyecto",
                                    "Documento",
                                    "Anio",
                                ]
                            ):
                                entities_collected.append(item)

            def resolve_entity_id(entity_node):
                props = dict(entity_node)
                entity_labels = list(entity_node.labels)
                entity_label = next(
                    (
                        lbl
                        for lbl in entity_labels
                        if lbl in ["Investigador", "Topico", "Proyecto", "Documento", "Anio"]
                    ),
                    "",
                )
                entity_id = props.get("id", "")
                if entity_label == "Investigador":
                    entity_id = props.get("name", entity_id) or entity_id
                elif entity_label == "Proyecto":
                    raw_id = props.get("id", "")
                    title = props.get("value", "")
                    entity_id = f"{title} ({raw_id})" if title else raw_id
                elif entity_label == "Documento":
                    entity_id = props.get("id", entity_id) or entity_id
                elif entity_label == "Topico":
                    entity_id = props.get("value", entity_id) or entity_id
                elif entity_label == "Anio":
                    entity_id = props.get("year", entity_id) or entity_id
                return entity_id, entity_label, props

            # Agregar entidades collected a evidence_entities (para el contexto del LLM)
            # pero NO asociarlas a chunks individuales
            for entity_node in entities_collected:
                entity_id, entity_label, props = resolve_entity_id(entity_node)
                if entity_id and entity_label:
                    evidence_entities[(entity_id, entity_label)] = props

            # Procesar chunks encontrados
            for chunk_node in chunks_in_record:
                chunk_id = chunk_node.get("id", "")
                if not chunk_id:
                    continue

                # Agregar chunk si no existe
                if chunk_id not in chunks_dict:
                    text = chunk_node.get("text", "")
                    if text:
                        chunks_dict[chunk_id] = GraphRAGChunk(
                            chunk_id=chunk_id,
                            text=text,
                        )

                # Solo asociar entidades DIRECTAS al chunk (implican EVIDENCIA_DE)
                if entities_direct and chunk_id in chunks_dict:
                    for entity_node in entities_direct:
                        entity_id, entity_label, props = resolve_entity_id(entity_node)
                        if entity_id and entity_label:
                            entity_tuple = (entity_id, entity_label)
                            if chunk_id not in chunk_to_entities:
                                chunk_to_entities[chunk_id] = []
                            if entity_tuple not in chunk_to_entities[chunk_id]:
                                chunk_to_entities[chunk_id].append(entity_tuple)
                            evidence_entities[(entity_id, entity_label)] = props

        return list(chunks_dict.values()), evidence_entities, chunk_to_entities

    def build_entity_context(
        self, evidence_entities: Dict[tuple, dict], chunk_to_entities: Dict[str, List[tuple]]
    ) -> str:
        """Construye contexto de entidades para el LLM con todas sus propiedades."""
        if not evidence_entities:
            return "No se encontraron entidades relevantes en el grafo."

        label_display = {
            "Proyecto": "Proyectos",
            "Investigador": "Investigadores",
            "Topico": "Tópicos",
            "Documento": "Documentos",
            "Anio": "Año",
        }

        lines = ["=== ENTIDADES ENCONTRADAS EN EL GRAFO ==="]

        for label in ["Proyecto", "Investigador", "Topico", "Documento", "Anio"]:
            entries = [
                ((eid, lbl), props)
                for (eid, lbl), props in evidence_entities.items()
                if lbl == label
            ]
            if not entries:
                continue
            lines.append(f"\n{label_display[label]}:")
            for (eid, _), props in sorted(entries, key=lambda x: x[0][0]):
                props_str = " | ".join(
                    f"{k}: {v}" for k, v in props.items() if v is not None and k != "text"
                )
                lines.append(f"  - {props_str}")

        return "\n".join(lines)

    def build_messages_for_answer(
        self, user_query: str, entity_context: str
    ) -> List[Dict[str, str]]:
        """
        Construir mensajes para el LLM usando SOLO entidades y relaciones del grafo.
        El texto de los chunks va al frontend, no aqui.
        """
        system_prompt = """Eres un asistente de investigación académica.
Se te dan resultados de una búsqueda en un grafo de conocimiento.
Respondé en español con UNA frase introductoria que mencione qué se buscó (ej: "Estos son los investigadores que participaron en X:" o "Se encontraron los siguientes proyectos relacionados con Y:"), seguida de una lista simple. Sin análisis, sin interpretaciones, sin mencionar evidencias ni chunks. Máximo 3 oraciones en total."""

        user_prompt = f"""CONSULTA: {user_query}

RESULTADOS:
{entity_context}

Respondé la consulta con una frase introductoria y la lista:"""

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def query(self, user_query: str) -> GraphRAGResult:
        """
        Pipeline completo de GraphRAG:
        """
        if not user_query or not user_query.strip():
            raise ValueError("Query vacía")

        logger.info(f"Query recibida: '{user_query}'")

        # Clasificar intención del usuario
        intent = self._classify_query_intent(user_query)
        logger.info(f"Intención clasificada: {intent}")

        # Si es conversacional, generar respuesta directa sin búsqueda en grafo
        if intent == "CHAT":
            logger.info("Modo conversacional activado (usando llama)")
            conversational_answer = self._generate_conversational_response(user_query)
            return GraphRAGResult(
                answer=conversational_answer,
                chunks=[],
                cypher_query="",
                chunk_to_entities={},
            )

        cypher_query = self.generate_cypher_query(user_query)

        # Ejecutar query
        records = self.execute_cypher_query(cypher_query)

        # Extraer chunks y evidencia
        chunks, evidence_entities, chunk_to_entities = (
            self._extract_chunks_and_entities_from_results(records)
        )
        logger.info(f"Extraídos {len(chunks)} chunks con {len(evidence_entities)} entidades")

        if not chunks:
            logger.warning("No se encontraron chunks en los resultados del grafo")

            if records and (self._is_aggregation_query(cypher_query) or not chunks):
                logger.info("Sin chunks en resultados, usando registros directos")
                context = self._build_aggregation_context(records)
                messages = [
                    {
                        "role": "system",
                        "content": "Respondé en español basado en estos resultados del grafo. Sé directo y conciso.",
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\nPREGUNTA: {user_query}\n\nRespuesta:",
                    },
                ]
                answer = self.answer_llm_client.generate(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                logger.info("Respuesta generada")
                return GraphRAGResult(
                    answer=answer,
                    chunks=[],
                    cypher_query=cypher_query,
                    chunk_to_entities={},
                )

            return GraphRAGResult(
                answer="No se encontró información relevante en el grafo para responder esta pregunta.",
                chunks=[],
                cypher_query=cypher_query,
                chunk_to_entities={},
            )

        entity_context = self.build_entity_context(evidence_entities, chunk_to_entities)

        messages = self.build_messages_for_answer(user_query, entity_context)
        answer = self.answer_llm_client.generate(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        logger.info("Respuesta final generada correctamente")

        return GraphRAGResult(
            answer=answer,
            chunks=chunks,
            cypher_query=cypher_query,
            chunk_to_entities=chunk_to_entities,
        )
