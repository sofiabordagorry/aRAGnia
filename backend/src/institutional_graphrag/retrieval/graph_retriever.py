"""GraphRAG retriever usando queries Cypher sobre Neo4j."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
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
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        from institutional_graphrag.llm.llm_provider import get_llm_client

        backend_dir = Path(__file__).resolve().parents[3]
        load_dotenv(backend_dir / ".env")
        cypher_model = os.getenv("OLLAMA_MODEL_CYPHER")
        answer_model = os.getenv("OLLAMA_MODEL_ANSWER")
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        # LLM para Cypher
        self.cypher_llm_client = get_llm_client(model=cypher_model)
        # LLM para clasificación y respuestas finales
        self.answer_llm_client = get_llm_client(model=answer_model)
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
            self.answer_llm_client.generate(
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
        user_query = "".join(
            c
            for c in unicodedata.normalize("NFD", user_query.lower())
            if unicodedata.category(c) != "Mn" or c == "\u0303"
        )
        user_query = unicodedata.normalize("NFC", user_query)

        prompt = self._build_cypher_generation_prompt(user_query)

        messages = [
            {
                "role": "system",
                "content": "You are an expert in Neo4j and Cypher. Generate precise, safe, read-only queries. For COUNT questions (cuántos/how many), use count() aggregation. For LIST questions (cuáles/list/show me), return entities with COLLECT(c) AS chunks. Connect all MATCH patterns via shared variables.",
            },
            {"role": "user", "content": prompt},
        ]

        # Reintentar hasta 2 veces si el LLM no genera los tags correctamente
        MAX_TAG_RETRIES = 2
        cypher_query = None
        for attempt in range(MAX_TAG_RETRIES):
            response = self.cypher_llm_client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Extraer query entre tags <QUERY>
            query_match = re.search(r"<QUERY>(.*?)</QUERY>", response, re.DOTALL | re.IGNORECASE)

            if query_match:
                cypher_query = query_match.group(1).strip()
                break
            else:
                logger.warning(
                    f"LLM no devolvió query entre tags <QUERY>...</QUERY> (intento {attempt + 1}/{MAX_TAG_RETRIES})"
                )
                if attempt < MAX_TAG_RETRIES - 1:
                    # Regenerar el prompt completo para mantener el contexto
                    logger.info("Regenerando prompt completo para reintento...")
                    prompt_with_reminder = (
                        prompt
                        + "\n\nREMINDER: You MUST wrap your Cypher query between <QUERY> and </QUERY> tags."
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert in Neo4j and Cypher. Generate precise, safe, read-only queries. For COUNT questions (cuántos/how many), use count() aggregation. For LIST questions (cuáles/list/show me), return entities with COLLECT(c) AS chunks. Connect all MATCH patterns via shared variables.",
                        },
                        {"role": "user", "content": prompt_with_reminder},
                    ]

        if not cypher_query:
            raise ValueError(
                "LLM no devolvió query entre tags <QUERY>...</QUERY> después de múltiples intentos"
            )

        # Corregir dirección incorrecta de PARTICIPO_EN si el LLM la invirtió
        cypher_query = self._fix_relationship_directions(cypher_query)

        # Detectar caso UNSUPPORTED (pregunta fuera del alcance de una sola query)
        if cypher_query.upper() == "UNSUPPORTED":
            raise ValueError(
                "UNSUPPORTED: La pregunta requiere múltiples consultas y está fuera del "
                "alcance de esta solución. Por favor, reformule su pregunta de forma más específica."
            )

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
- Proyecto     → id: 'gi_2014_133', value: 'sintesis y evaluacion biologica de nuevos quimioterapicos'
- Investigador → id: 'lastname_firstname', name: 'firstname lastname', cedula: 'cedula' (optional), mail: 'email' (optional), afiliacion: 'institutional affiliation' (optional)
- Topico       → value: 'biotecnologia'
- Dominio      → value: 'ciencias naturales'
- Documento → id: '...', base_name: 'gi_2014_133', type: 'informe'|'propuesta'|'resumen'|'tabla', year_publisher: '2014', is_group: 'false'
- Chunk        → id: '...', text: '...'
- Anio         → year: '2014'           ← property is "year", NOT "value" or "id"

Relationships:
- (Investigador)-[:PARTICIPO_EN]->(Proyecto)
- (Investigador)-[:RESPONSABLE_DE]->(Proyecto)
- (Proyecto)-[:TIENE_TOPICO]->(Topico)
- (Topico)-[:PERTENECE_A_DOMINIO]->(Dominio)
- (Proyecto)-[:ES_DESCRITO_POR]->(Documento)
- (Proyecto)-[:INICIO_EN]->(Anio)
- (Documento)-[:PRIMER_CHUNK]->(Chunk)
- (Chunk)-[:SIGUIENTE_CHUNK]->(Chunk)
- (Chunk)-[:DE_DOCUMENTO]->(Documento)
- (Chunk)-[:EXTRAIDO_DE]->(Investigador|Topico)
- (Proyecto)-[:TITULO_EXTRAIDO_DE]->(Chunk)

RULES:

1. Read-only (MATCH, OPTIONAL MATCH, WHERE, RETURN)
2. Return chunks (c:Chunk) when listing or describing entities — they contain the actual text evidence. For COUNT queries, omit chunks and return only the aggregation result.
3. Connect patterns: every MATCH must use variables defined in previous MATCHes
4. Use [:EXTRAIDO_DE] for investigators/topics, [:TITULO_EXTRAIDO_DE] for projects to navigate to their evidence chunks
6. Only add LIMIT when the question explicitly asks for a specific number of results (e.g. "los 10 tópicos con más proyectos" → LIMIT 10). Otherwise, omit LIMIT entirely.
7. NEVER define relationship variables — use anonymous patterns only: -[:TYPE]-> NOT -[r:TYPE]->
8. Node variables must be unique and never reused for a different type
9. Generate EXACTLY ONE Cypher query — never split the answer into multiple separate queries
10. Every variable used in WITH or RETURN must have been defined in a preceding MATCH/OPTIONAL MATCH
11. If the question genuinely CANNOT be answered with a single query, respond with <QUERY>UNSUPPORTED</QUERY>
12. When the question asks "how many" / "cuántos" / "qué cantidad", use count() aggregation (e.g., RETURN count(p) AS total). Do NOT return individual entities unless the question explicitly asks to list them.
13. ALWAYS filter values using WHERE.
14. ALWAYS normalize text values: lowercase, no accents, never translate.
15. Topics are stored in Spanish, lowercase and without accents: 'biotecnologia', 'ingenieria', 'medicina', etc.
16. Domains are stored in Spanish, lowercase and without accents.
17 Search project titles/names with toLower(p.value) CONTAINS.
18. Convert Anio.year with toInteger() for numeric comparisons.

PATTERNS:

=== COUNT QUERIES (when user asks "cuántos", "cuántas", "how many") ===

Count projects by investigator (e.g., "cuántos proyectos tiene X?"):
MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'lastname'
MATCH (i)-[:PARTICIPO_EN]->(p:Proyecto)
RETURN count(p) AS total

Count projects by investigator responsable (e.g., "de cuántos proyectos fue responsable X?"):
MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'lastname'
MATCH (i)-[:RESPONSABLE_DE]->(p:Proyecto)
RETURN count(p) AS total

Count projects by year (e.g., "cuántos proyectos en 2018?"):
MATCH (a:Anio)
WHERE a.year = '2018'
MATCH (p:Proyecto)-[:INICIO_EN]->(a)
RETURN count(p) AS total

Count projects by topic (e.g., "cuántos proyectos de biotecnología?"):
MATCH (t:Topico)
WHERE t.value = 'biotecnologia'
MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)
RETURN count(p) AS total

Count total entities (e.g., "cuántos investigadores hay?"):
MATCH (i:Investigador) RETURN count(i) AS total
MATCH (p:Proyecto) RETURN count(p) AS total


Count projects by domain:
MATCH (d:Dominio)
WHERE d.value = 'ciencias naturales'
MATCH (t:Topico)-[:PERTENECE_A_DOMINIO]->(d)
MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)
RETURN count(DISTINCT p) AS total

Count projects by project title/name:
MATCH (p:Proyecto)
WHERE toLower(p.value) CONTAINS 'web warehouse de datos abiertos'
RETURN count(p) AS total


=== LIST QUERIES (when user asks "cuáles", "qué proyectos", "lista", "muéstrame") ===

List projects by topic:
MATCH (t:Topico)
WHERE t.value = 'biotecnologia'
MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)
MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
RETURN p, COLLECT(c) AS chunks

List investigators of a project (WHO participated):
MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)
WHERE p.id = 'gi_2014_133'
OPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)
RETURN i, COLLECT(DISTINCT c) AS chunks
-- CRITICAL: RETURN the investigators (i), NOT the project (p)

List researchers responsible for a project (WHO was IN CHARGE):
MATCH (i:Investigador)-[:RESPONSABLE_DE]->(p:Proyecto)
WHERE p.id = 'gi_2014_133'
OPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)
RETURN i, COLLECT(DISTINCT c) AS chunks


List projects by domain:
MATCH (d:Dominio)
WHERE d.value = 'ciencias naturales'
MATCH (t:Topico)-[:PERTENECE_A_DOMINIO]->(d)
MATCH (p:Proyecto)-[:TIENE_TOPICO]->(t)
MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
RETURN DISTINCT p, d, COLLECT(DISTINCT c) AS chunks

List project by title/name:
MATCH (p:Proyecto)
WHERE toLower(p.value) CONTAINS 'web warehouse de datos abiertos'
OPTIONAL MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
RETURN p, COLLECT(DISTINCT c) AS chunks


List domains of a topic:
MATCH (t:Topico)
WHERE t.value = 'inteligencia artificial'
MATCH (t)-[:PERTENECE_A_DOMINIO]->(d:Dominio)
RETURN t, d

Topics of a domain:
MATCH (d:Dominio)
WHERE d.value = 'ciencias naturales'
MATCH (t:Topico)-[:PERTENECE_A_DOMINIO]->(d)
RETURN d, COLLECT(DISTINCT t) AS topics

Investigators of project by ID:
MATCH (p:Proyecto)
WHERE p.id = 'gi_2014_133'
MATCH (i:Investigador)-[:PARTICIPO_EN]->(p)
OPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)
RETURN i, COLLECT(DISTINCT c) AS chunks


Investigators of project by title/name:
MATCH (p:Proyecto)
WHERE toLower(p.value) CONTAINS 'web warehouse de datos abiertos'
MATCH (i:Investigador)-[:PARTICIPO_EN]->(p)
OPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)
RETURN i, COLLECT(DISTINCT c) AS chunks


Responsible investigators by project ID:
MATCH (p:Proyecto)
WHERE p.id = 'gi_2014_133'
MATCH (i:Investigador)-[:RESPONSABLE_DE]->(p)
OPTIONAL MATCH (c:Chunk)-[:EXTRAIDO_DE]->(i)
RETURN i, COLLECT(DISTINCT c) AS chunks

-- CRITICAL: RETURN the researchers (i), NOT the project (p)

Projects by name of researcher in charge — note direction: Investigador -> Proyecto:
MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'lastname'
MATCH (i)-[:RESPONSABLE_DE]->(p:Proyecto)
MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
RETURN p, i, COLLECT(c) AS chunks

Projects by researcher name — note direction: Investigador -> Proyecto:
MATCH (i:Investigador) WHERE toLower(i.name) CONTAINS 'lastname'
MATCH (i)-[:PARTICIPO_EN]->(p:Proyecto)
MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
RETURN p, i, COLLECT(c) AS chunks

Chunks of a specific topic (only when asked about the topic itself, not its projects):
MATCH (t:Topico {{value: 'Biotechnology'}})
MATCH (c:Chunk)-[:EXTRAIDO_DE]->(t)
RETURN c, t

By project ID (chunks about a specific project):
MATCH (p:Proyecto {{id: 'gi_2014_133'}})
MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
RETURN c, p

Documents of a project:
MATCH (p:Proyecto {{id: 'gi_2014_133'}})
MATCH (p)-[:ES_DESCRITO_POR]->(d:Documento)
MATCH (c:Chunk)-[:DE_DOCUMENTO]->(d)
RETURN d, p, COLLECT(c) AS chunks

Describe / full info about a specific project (name, year, topics, investigators):
MATCH (p:Proyecto {{id: 'gi_2014_133'}})
OPTIONAL MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
OPTIONAL MATCH (p)-[:TIENE_TOPICO]->(t:Topico)
OPTIONAL MATCH (maininv:Investigador)-[:RESPONSABLE_DE]->(p)
OPTIONAL MATCH (inv:Investigador)-[:PARTICIPO_EN]->(p)
OPTIONAL MATCH (p)-[:INICIO_EN]->(a:Anio)
RETURN p, COLLECT(DISTINCT c) AS chunks, COLLECT(DISTINCT t) AS topics, COLLECT(DISTINCT inv) AS investigators, COLLECT(DISTINCT maininv) AS researchers_in_charge, a LIMIT 1

Projects by a specific year (LIST):
MATCH (a:Anio {{year: 'year_value'}})
MATCH (p:Proyecto)-[:INICIO_EN]->(a)
MATCH (p)-[:TITULO_EXTRAIDO_DE]->(c:Chunk)
RETURN p, COLLECT(c) AS chunks

=== ADVANCED AGGREGATION ===

Projects by area for ALL years (aggregation with grouping):
MATCH (a:Anio)
MATCH (p:Proyecto)-[:INICIO_EN]->(a)
MATCH (p)-[:TIENE_TOPICO]->(t:Topico)
RETURN a.year AS anio, t.value AS area, count(DISTINCT p) AS total_proyectos
ORDER BY anio DESC, total_proyectos DESC

Top N topics by project count (user asked for specific number, e.g. 10):
MATCH (t:Topico)<-[:TIENE_TOPICO]-(p:Proyecto)
RETURN t.value AS area, count(DISTINCT p) AS total ORDER BY total DESC LIMIT 10

Top N investigators by project count (e.g., "qué investigadores participaron en más proyectos? top 10"):
MATCH (i:Investigador)-[:PARTICIPO_EN]->(p:Proyecto)
WITH i, count(DISTINCT p) AS num_proyectos
RETURN i.name AS investigador, num_proyectos
ORDER BY num_proyectos DESC
LIMIT 10
-- CRITICAL: DO NOT add WHERE conditions filtering by name unless explicitly asked

=== SIMPLE VALUE QUERIES (year, name, single property) ===

Get year when a project started:
MATCH (p:Proyecto {{id: 'gi_2010_152'}})
OPTIONAL MATCH (p)-[:INICIO_EN]->(a:Anio)
RETURN a.year AS año, a


QUESTION: {user_query}

CRITICAL DECISION - COUNT vs LIST:
- If question asks "cuántos", "cuántas", "how many", "qué cantidad" → USE count() and RETURN count(x) AS total (NO chunks needed)
- If question asks "cuáles", "qué proyectos", "quiénes", "list", "muéstrame" → RETURN entities + COLLECT(c) AS chunks
- If question asks "quién/quiénes" (WHO) → RETURN investigators (i), NOT projects
- If question asks "qué año" (WHAT year) → RETURN year value directly (a.year or a)
- Analyze the question intent carefully before generating the query
RETURN RULES:
- "¿Quiénes participaron?" → RETURN investigadores (i), NOT proyecto (p)
- "¿En qué año?" → RETURN año (a.year AS año) or (a) with OPTIONAL MATCH for chunks
- "¿Cuántos proyectos?" → RETURN count(p) AS total
- "¿Qué investigadores con más proyectos?" → RETURN i.name, count(p) ORDER BY count(p) DESC LIMIT N

CRITICAL SYNTAX:
- Wrap your query in <QUERY> and </QUERY> tags
- Every variable in WITH/RETURN must be defined in a previous MATCH
- Use [:EXTRAIDO_DE]->(entity) for investigators/topics, [:TITULO_EXTRAIDO_DE]->(chunk) for projects
- Topico uses {{value: '...'}}, Anio uses {{year: '...'}}, all others use {{id: '...'}}
- NEVER name a relationship variable (never write -[r:TYPE]-> or -[rel:TYPE]->), always use -[:TYPE]->
- NEVER use a variable as both a relationship and a node
- NEVER generate paths like (a)-[:REL]->(b)-[:REL2]->(c).
- ALWAYS use WHERE for filtering
- ALWAYS use toLower(p.value) CONTAINS 'normalized project text' for project titles/names

<QUERY>
"""

    def _fix_relationship_directions(self, query: str) -> str:
        """
        Corrige las direcciones de las relaciones cuando el LLM las genera al revés.
        Schema correcto:
        - (Investigador)-[:PARTICIPO_EN]->(Proyecto)
        - (Investigador)-[:RESPONSABLE_DE]->(Proyecto)
        - (Proyecto)-[:TIENE_TOPICO]->(Topico)
        - (Proyecto)-[:ES_DESCRITO_POR]->(Documento)
        - (Proyecto)-[:INICIO_EN]->(Anio)
        - (Documento)-[:PRIMER_CHUNK]->(Chunk)
        - (Chunk)-[:SIGUIENTE_CHUNK]->(Chunk)
        - (Chunk)-[:DE_DOCUMENTO]->(Documento)
        - (Chunk)-[:EXTRAIDO_DE]->(Investigador|Topico)
        - (Proyecto)-[:TITULO_EXTRAIDO_DE]->(Chunk)
        """
        # Definir las relaciones correctas: (source_type, rel_type, target_type)
        correct_directions = [
            ("Investigador", "PARTICIPO_EN", "Proyecto"),
            ("Investigador", "RESPONSABLE_DE", "Proyecto"),
            ("Proyecto", "TIENE_TOPICO", "Topico"),
            ("Topico", "PERTENECE_A_DOMINIO", "Dominio"),
            ("Proyecto", "ES_DESCRITO_POR", "Documento"),
            ("Proyecto", "INICIO_EN", "Anio"),
            ("Documento", "PRIMER_CHUNK", "Chunk"),
            ("Chunk", "SIGUIENTE_CHUNK", "Chunk"),
            ("Chunk", "DE_DOCUMENTO", "Documento"),
            ("Chunk", "EXTRAIDO_DE", "Investigador"),
            ("Chunk", "EXTRAIDO_DE", "Topico"),
            ("Proyecto", "TITULO_EXTRAIDO_DE", "Chunk"),
        ]
        fixed = query
        corrections_made = []
        var_types = dict(re.findall(r"\((\w+)\s*:\s*(\w+)", fixed))

        for source_type, rel_type, target_type in correct_directions:
            # Patrón para detectar la dirección invertida
            pattern = re.compile(
                rf"""
                (?P<match_type>OPTIONAL\s+MATCH|MATCH)\s*
                \(\s*(?P<left>\w+)\s*(?::\s*(?P<left_label>\w+))?\s*\)
                \s*-\s*\[:{rel_type}\]\s*->\s*
                \(\s*(?P<right>\w+)\s*(?::\s*(?P<right_label>\w+))?\s*\)
                """,
                re.IGNORECASE | re.VERBOSE,
            )

            def repl(m: re.Match[str]) -> str:
                left = m.group("left")
                right = m.group("right")

                left_type = m.group("left_label") or var_types.get(left)
                right_type = m.group("right_label") or var_types.get(right)
                # Si está invertida: target -> source
                if left_type == target_type and right_type == source_type:
                    return (
                        f"{m.group('match_type')} "
                        f"({right}:{source_type})-[:{rel_type}]->({left}:{target_type})"
                    )

                return str(m.group(0))

            new_fixed = pattern.sub(repl, fixed)

            if new_fixed != fixed:
                corrections_made.append(f"{target_type}-[:{rel_type}]->{source_type}")
                fixed = new_fixed

        if corrections_made:
            logger.info(f"Direcciones corregidas automáticamente: {', '.join(corrections_made)}")

        return fixed

    def _fix_cypher_query(self, broken_query: str, syntax_error: str) -> str:
        """
        Pide al LLM que corrija una query Cypher con error de sintaxis.
        """
        prompt = f"""The following Cypher query produced a syntax error. Fix it.

SCHEMA:
Nodes: Proyecto(id, value), Investigador(id, name), Topico(value), Documento(id, base_name, type, year_publisher, is_group), Chunk(id, text), Anio(year)
Relationships:
- (Investigador)-[:PARTICIPO_EN]->(Proyecto)
- (Investigador)-[:RESPONSABLE_DE]->(Proyecto)
- (Proyecto)-[:TIENE_TOPICO]->(Topico)
- (Proyecto)-[:ES_DESCRITO_POR]->(Documento)
- (Proyecto)-[:INICIO_EN]->(Anio)
- (Documento)-[:PRIMER_CHUNK]->(Chunk)
- (Chunk)-[:SIGUIENTE_CHUNK]->(Chunk)
- (Chunk)-[:DE_DOCUMENTO]->(Documento)
- (Chunk)-[:EXTRAIDO_DE]->(Investigador|Topico)
- (Proyecto)-[:TITULO_EXTRAIDO_DE]->(Chunk)

BROKEN QUERY:
{broken_query}

ERROR:
{syntax_error}

RULES:
- Every variable used in WITH or RETURN must be defined in a prior MATCH/OPTIONAL MATCH
- Never define relationship variables: use -[:TYPE]-> not -[r:TYPE]->
- Generate EXACTLY ONE corrected query
- Read-only (MATCH, RETURN only)

Return ONLY the fixed query wrapped in <QUERY> and </QUERY> tags.

<QUERY>
"""
        messages = [
            {
                "role": "system",
                "content": "You are an expert in Neo4j Cypher. Fix the broken query and return it in <QUERY>...</QUERY> tags.",
            },
            {"role": "user", "content": prompt},
        ]

        # Reintentar hasta 2 veces si el LLM no genera los tags correctamente
        MAX_TAG_RETRIES = 2
        fixed_query = None

        for attempt in range(MAX_TAG_RETRIES):
            response = self.cypher_llm_client.generate(
                messages=messages,
                temperature=0.0,
                max_tokens=self.max_tokens,
            )

            query_match = re.search(r"<QUERY>(.*?)</QUERY>", response, re.DOTALL | re.IGNORECASE)

            if query_match:
                fixed_query = query_match.group(1).strip()
                break
            else:
                logger.warning(
                    f"LLM no devolvió query corregida entre tags <QUERY>...</QUERY> (intento {attempt + 1}/{MAX_TAG_RETRIES})"
                )
                if attempt < MAX_TAG_RETRIES - 1:
                    # Regenerar el prompt completo para mantener el contexto
                    logger.info("Regenerando prompt completo de corrección para reintento...")
                    prompt_with_reminder = (
                        prompt
                        + "\n\nREMINDER: You MUST wrap your corrected Cypher query between <QUERY> and </QUERY> tags."
                    )
                    messages = [
                        {
                            "role": "system",
                            "content": "You are an expert in Neo4j Cypher. Fix the broken query and return it in <QUERY>...</QUERY> tags.",
                        },
                        {"role": "user", "content": prompt_with_reminder},
                    ]

        if not fixed_query:
            raise ValueError(
                "LLM no devolvió query corregida entre tags <QUERY>...</QUERY> después de múltiples intentos"
            )

        is_safe, error = CypherQueryValidator.is_safe(fixed_query)
        if not is_safe:
            raise ValueError(f"Query corregida no es segura: {error}")

        logger.info(f"Query corregida por LLM: {fixed_query}")
        return fixed_query

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
        """Detecta si una query es de agregación (usa COUNT, SUM, AVG, etc.) o devuelve valores simples."""
        query_upper = cypher_query.upper()
        # Agregaciones numéricas
        aggregation_functions = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
        # También considerar queries que devuelven propiedades simples sin COLLECT
        has_aggregation = any(func in query_upper for func in aggregation_functions)
        # Si no tiene COLLECT ni chunks explícitos, probablemente es una query simple
        has_collect = "COLLECT(" in query_upper
        return has_aggregation or not has_collect

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

    def extract_chunks_and_entities_from_results(
        self, records: List[Any]
    ) -> tuple[List[GraphRAGChunk], Dict[tuple[str, str], dict], Dict[str, List[tuple[str, str]]]]:
        """Extrae chunks y evidencia de entidades desde los resultados de Cypher."""
        chunks_dict: Dict[str, GraphRAGChunk] = {}  # chunk_id -> GraphRAGChunk
        evidence_entities: Dict[tuple[str, str], dict] = {}  # (entity_id, label) -> props
        chunk_to_entities: Dict[str, List[tuple[str, str]]] = {}  # chunk_id -> [(entity_id, label)]

        for record in records:
            # Recopilar chunks y entidades de este record
            chunks_in_record = []
            # direct = vienen de MATCH directo (implica relación EXTRAIDO_DE/TITULO_EXTRAIDO_DE)
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

                # Solo asociar entidades DIRECTAS al chunk (implican EXTRAIDO_DE/TITULO_EXTRAIDO_DE)
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
                if label == "Proyecto":
                    pid = props.get("id", "")
                    title = props.get("value", "") or props.get("name", "")
                    if title and pid:
                        lines.append(f"  - {title} ({pid})")
                    else:
                        lines.append(f"  - {pid or title}")
                else:
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
        system_prompt = """Eres un asistente de investigación académica especializado en presentar resultados de búsquedas en grafos de conocimiento.

REGLAS ESTRICTAS:
1. Debes responder en ESPAÑOL
2. Debes incluir TODOS los elementos que aparecen en la sección RESULTADOS - no omitas ninguno
3. Formato: Lista completa de resultados
4. NO inventes información - solo usa lo que está en RESULTADOS
5. NO agregues análisis ni interpretaciones
6. Si hay una lista, reprodúcela COMPLETA

Ejemplo:
Si RESULTADOS muestra 5 proyectos, tu respuesta debe listar los 5 proyectos."""

        user_prompt = f"""CONSULTA DEL USUARIO:
{user_query}

RESULTADOS ENCONTRADOS EN EL GRAFO:
{entity_context}

IMPORTANTE: Usa TODOS los resultados mostrados arriba para generar tu respuesta. No omitas ningún elemento de la lista.

Tu respuesta (frase introductoria + lista completa):"""

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
        cypher_query: str = ""
        records: List[Any] = []
        invalidResult, records, cypher_query = self.generate_cypher_query_result(
            user_query=user_query
        )

        if not records:
            return invalidResult
        return self.generate_result(
            records=records, user_query=user_query, cypher_query=cypher_query
        )

    def generate_cypher_query_result(self, user_query) -> Tuple[GraphRAGResult, List[Any], str]:
        logger.info(f"Query recibida: '{user_query}'")

        intent = self._classify_query_intent(user_query)
        logger.info(f"Intención clasificada: {intent}")

        # Si es conversacional, generar respuesta directa sin búsqueda en grafo
        if intent == "CHAT":
            logger.info("Modo conversacional activado (usando llama)")
            conversational_answer = self._generate_conversational_response(user_query)
            return (
                GraphRAGResult(
                    answer=conversational_answer,
                    chunks=[],
                    cypher_query="",
                    chunk_to_entities={},
                ),
                [],
                "",
            )

        try:
            cypher_query = self.generate_cypher_query(user_query)
        except ValueError as e:
            if str(e).startswith("UNSUPPORTED"):
                return (
                    GraphRAGResult(
                        answer=(
                            "Esta pregunta requiere múltiples consultas para responderse y está "
                            "fuera del alcance de esta solución. Por favor, intente dividirla en "
                            "preguntas más específicas."
                        ),
                        chunks=[],
                        cypher_query="",
                        chunk_to_entities={},
                    ),
                    [],
                    "",
                )
            raise

        # Ejecutar query con reintentos en caso de error de sintaxis
        MAX_SYNTAX_RETRIES = 3
        _too_complex_result = GraphRAGResult(
            answer=(
                "La consulta es muy compleja y está teniendo problemas para resolverla. "
                "Por favor, intente reformular su pregunta de forma más específica."
            ),
            chunks=[],
            cypher_query=cypher_query,
            chunk_to_entities={},
        )
        records: List[Any] = []
        for attempt in range(MAX_SYNTAX_RETRIES):
            try:
                records = self.execute_cypher_query(cypher_query)
                break
            except CypherSyntaxError as exc:
                logger.warning(
                    f"Error de sintaxis Cypher (intento {attempt + 1}/{MAX_SYNTAX_RETRIES}): {exc}"
                )
                if attempt == MAX_SYNTAX_RETRIES - 1:
                    logger.error("Se agotaron los reintentos de corrección de sintaxis")
                    return _too_complex_result, [], ""
                try:
                    cypher_query = self._fix_cypher_query(cypher_query, str(exc))
                except ValueError as fix_err:
                    logger.error(f"No se pudo corregir la query: {fix_err}")
                    return _too_complex_result, [], ""
        return _too_complex_result, records, cypher_query

    def generate_result(self, records, user_query, cypher_query) -> GraphRAGResult:
        # Extraer chunks y evidencia
        chunks, evidence_entities, chunk_to_entities = (
            self.extract_chunks_and_entities_from_results(records)
        )
        logger.info(f"Extraídos {len(chunks)} chunks con {len(evidence_entities)} entidades")
        if not chunks:
            logger.warning("No se encontraron chunks en los resultados del grafo")

            # Caso 1: Query tiene resultados (agregación, valores simples, o nodos sin chunks)
            if records:
                logger.info("Sin chunks pero con resultados del grafo, procesando...")
                context = self._build_aggregation_context(records)
                logger.info(f"Contexto construido ({len(context)} chars)")

                # Detectar si es una query de conteo simple
                is_count = any(
                    key.lower() in ["total", "count", "cantidad"]
                    for record in records
                    for key in record.keys()
                )

                if is_count:
                    # Para queries de conteo, ser muy explícito
                    messages = [
                        {
                            "role": "system",
                            "content": "Respondé en español de forma DIRECTA y NUMÉRICA. Si los resultados muestran un número, respondé ese número exacto. No digas 'no se puede determinar' si el número está ahí.",
                        },
                        {
                            "role": "user",
                            "content": f"PREGUNTA: {user_query}\n\nRESULTADOS DEL GRAFO:\n{context}\n\nRESPONDE con el número exacto que aparece en los resultados. Ejemplo: Si los resultados muestran 'total: 3', respondé '3 proyectos iniciaron en ese año.'",
                        },
                    ]
                else:
                    # Para otros resultados sin chunks (años, nombres, etc.)
                    messages = [
                        {
                            "role": "system",
                            "content": "Respondé en español basado EXACTAMENTE en los resultados mostrados. Incluí TODOS los resultados sin omitir ninguno. Sé directo y completo.",
                        },
                        {
                            "role": "user",
                            "content": f"PREGUNTA: {user_query}\n\nRESULTADOS DEL GRAFO:\n{context}\n\nIMPORTANTE: Los resultados arriba contienen la respuesta. Úsalos TODOS. Si ves un valor de año, ese es el año. Si ves nombres, esos son los nombres. No digas que no hay información si los resultados muestran datos.",
                        },
                    ]

                answer = self.answer_llm_client.generate(
                    messages=messages,
                    temperature=0.1,  # Más determinístico para respuestas factuales
                    max_tokens=2048,
                )
                logger.info(f"Respuesta generada ({len(answer)} chars): {answer[:200]}...")
                return GraphRAGResult(
                    answer=answer,
                    chunks=[],
                    cypher_query=cypher_query,
                    chunk_to_entities={},
                )

            # Caso 2: Sin resultados en absoluto
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
            max_tokens=2048,  # Aumentado para permitir listas más largas
        )
        logger.info(f"Respuesta final generada ({len(answer)} chars): {answer[:200]}...")

        return GraphRAGResult(
            answer=answer,
            chunks=chunks,
            cypher_query=cypher_query,
            chunk_to_entities=chunk_to_entities,
        )
