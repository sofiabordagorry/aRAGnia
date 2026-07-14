"""GraphRAG retriever usando queries Cypher sobre Neo4j."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import CypherSyntaxError
from neo4j.graph import Node

from institutional_graphrag.llm.llm_provider import get_llm_client
from institutional_graphrag.retrieval.fewshot_store import FewShotStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GraphRAGChunk:
    """Chunk de contexto obtenido del grafo."""

    chunk_id: str
    text: str
    page: int


@dataclass
class GraphRAGResult:
    """Resultado de GraphRAG."""

    answer: str
    chunks: List[GraphRAGChunk]
    cypher_query: str
    chunk_to_entities: Dict[str, List[tuple[str, str]]]  # chunk_id -> [(entity_id, entity_label)]


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

        for pattern in cls.FORBIDDEN_KEYWORDS:
            if re.search(pattern, query_upper):
                return False, f"Query contiene operación prohibida: {pattern}"

        if "MATCH" not in query_upper and "RETURN" not in query_upper:
            return False, "Query debe contener MATCH o RETURN"

        return True, None


class GraphRAGRetriever:
    """Retriever que usa queries Cypher para obtener la información relevante."""

    def __init__(
        self,
        neo4j_uri: str,
        neo4j_user: str,
        neo4j_password: str,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        fewshot_store: Optional[Any] = None,
    ):
        backend_dir = Path(__file__).resolve().parents[3]
        load_dotenv(backend_dir / ".env")

        backend = os.getenv("LLM_BACKEND", "ollama").lower()

        if backend == "huggingface":
            cypher_model = os.getenv("HF_RETRIEVAL_MODEL")
            answer_model = os.getenv("HF_GENERATION_MODEL")
        else:
            cypher_model = os.getenv("OLLAMA_MODEL_CYPHER")
            answer_model = os.getenv("OLLAMA_MODEL_ANSWER")

        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.cypher_llm_client = get_llm_client(model=cypher_model)
        self.answer_llm_client = get_llm_client(model=answer_model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._schema_cache: Optional[str] = None
        self._fewshot: Optional[Any] = (
            fewshot_store if fewshot_store is not None else self._init_fewshot()
        )

    def _init_fewshot(self) -> Optional[Any]:
        try:
            store = FewShotStore()
            if store.count() == 0:
                logger.info("Colección few-shot vacía, se usarán ejemplos estáticos")
                store.close()
                return None
            logger.info(f"Few-shot store cargado ({store.count()} ejemplos)")
            return store
        except Exception as e:
            logger.warning(f"Few-shot store no disponible, se usarán ejemplos estáticos: {e}")
            return None

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

            query_match = re.search(r"<QUERY>(.*?)</QUERY>", response, re.DOTALL | re.IGNORECASE)

            if query_match:
                cypher_query = query_match.group(1).strip()
                break
            else:
                logger.warning(
                    f"LLM no devolvió query entre tags <QUERY>...</QUERY> (intento {attempt + 1}/{MAX_TAG_RETRIES})"
                )
                logger.info(f"Respuesta fallida cruda del LLM al corregir: {response}")
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

        cypher_query = self._fix_relationship_directions(cypher_query)

        cypher_query = self._use_display_fields_for_return(cypher_query)

        if cypher_query.upper() == "NOT_IN_SCHEMA":
            raise ValueError(
                "NOT_IN_SCHEMA: La información solicitada no existe en el esquema del grafo."
            )
        if cypher_query.upper() == "UNSUPPORTED":
            raise ValueError(
                "UNSUPPORTED: La pregunta requiere múltiples consultas y está fuera del "
                "alcance de esta solución. Por favor, reformule su pregunta de forma más específica."
            )

        is_safe, error = CypherQueryValidator.is_safe(cypher_query)
        if not is_safe:
            logger.warning(
                f"La query generada falló la validación de seguridad. Motivo {error} | Query: {cypher_query}"
            )
            raise ValueError(f"Query generada no es segura: {error}")

        logger.info(f"Query Cypher generada: {cypher_query}")
        return cypher_query

    def _fetch_schema(self) -> str:
        """Obtiene labels/propiedades de nodos y relaciones de Neo4j. Se cachea tras la primera llamada."""
        if self._schema_cache is not None:
            return self._schema_cache

        node_props: Dict[str, List[str]] = {}
        rels_props: Dict[tuple[str, str, str], set[str]] = {}

        with self.driver.session() as session:
            records = list(session.run("""
                    MATCH (n)
                    UNWIND labels(n) AS lbl
                    UNWIND keys(n) AS prop
                    RETURN lbl AS label, collect(DISTINCT prop) AS properties
                    ORDER BY label
                    """))
            for r in records:
                if r["label"]:
                    node_props[r["label"]] = r["properties"]

            records = list(session.run("""
                    MATCH (a)-[r]->(b)
                    RETURN labels(a)[0] AS source, type(r) AS rel, labels(b)[0] AS target, keys(r) AS props
                    """))
            for r in records:
                if r["source"] and r["rel"] and r["target"]:
                    key = (r["source"], r["rel"], r["target"])
                    rels_props.setdefault(key, set()).update(r["props"] or [])

        lines = ["Nodes and their key properties:"]
        for label, props in sorted(node_props.items()):
            lines.append(f"- {label:<15} → {', '.join(props)}")

        lines.append("\nRelationships:")
        for (source, rel, target), rel_props in sorted(rels_props.items()):
            props_str = " {" + ", ".join(sorted(rel_props)) + "}" if rel_props else ""
            lines.append(f"- ({source})-[:{rel}{props_str}]->({target})")

        self._schema_cache = "\n".join(lines)
        logger.info("Schema cargado desde Neo4j y cacheado")
        return self._schema_cache

    def _fetch_fewshot_examples(self, user_query: str) -> str:
        """Retorna un bloque formateado de ejemplos similares (pregunta, cypher)."""
        if self._fewshot:
            try:
                examples = self._fewshot.search(user_query, top_k=3)
                if examples:
                    formatted_examples = [f"Pregunta: {q}\nQuery cypher:\n{c}" for q, c in examples]
                    logger.info(
                        "Few-shot examples retrieved: \n %s", "\n\n".join(formatted_examples)
                    )
                    lines = ["SIMILAR EXAMPLES (use as reference patterns):"]
                    for q, c in examples:
                        lines.append(f"\nQuestion: {q}\n<QUERY>\n{c}\n</QUERY>")
                    return "\n".join(lines)
            except Exception as e:
                logger.warning(f"Few-shot search failed, skipping: {e}")
        return ""

    def _build_cypher_generation_prompt(self, user_query: str) -> str:
        """Construye el prompt para la generación de queries Cypher."""
        schema = self._fetch_schema()
        fewshot_block = self._fetch_fewshot_examples(user_query)
        return f"""Generate a Cypher query for Neo4j to answer this question.
SCHEMA:
{schema}

FEW-SHOT EXAMPLES:
{fewshot_block}

OUTPUT CONTRACT:
- Return only one query wrapped exactly as <QUERY>...</QUERY>.
- Do not write explanations, markdown, comments, or multiple alternatives.
- If the requested information is not represented in the schema, return <QUERY>NOT_IN_SCHEMA</QUERY>.
- If it is related to the schema but cannot be answered with one Cypher query, return <QUERY>UNSUPPORTED</QUERY>.

GRAPH FACTS THAT MUST BE FOLLOWED:
- Relationship directions:
  - (Investigador)-[:PARTICIPO_EN]->(Proyecto|Grupo)
  - (Proyecto|Grupo)-[:TIENE_TOPICO]->(Topico)
  - (Topico)-[:PERTENECE_A_SUBCAMPO]->(Subcampo)
  - (Proyecto|Grupo)-[:PERTENECE_A_AREA]->(Area)
  - (Proyecto|Grupo)-[:INICIO_EN]->(Anio)
  - (Proyecto|Grupo)-[:TITULO_EXTRAIDO_DE]->(Chunk)
  - (Chunk)-[:EXTRAIDO_DE]->(Investigador|Topico)
  - (Chunk)-[:DE_DOCUMENTO]->(Documento)
- Never reverse these directions.
- Never invent relationships between Chunk and Proyecto except (Proyecto)-[:TITULO_EXTRAIDO_DE]->(Chunk).
- Never use (c:Chunk)-[:EXTRAIDO_DE]->(p:Proyecto): EXTRAIDO_DE is only for Investigador or Topico.
- Never write COLLECT(c:Chunk), COLLECT(c)-[:REL]->(), or any pattern inside RETURN. MATCH/OPTIONAL MATCH first, then RETURN COLLECT(DISTINCT c).
- Anio.anio is a string. Use a.anio = '2018'. For ranges use toInteger(a.anio) >= 2018.
- Proyecto.titulo, Investigador.nombre, Topico.valor, Subcampo.valor and Area.valor are already normalized: lowercase, no accents.
- Search project titles with WHERE toLower(p.titulo) CONTAINS 'normalized title fragment'. Prefer p.titulo over c.texto.
- Search researchers with WHERE toLower(i.nombre) CONTAINS 'normalized name fragment'. Prefer partial names over exact equality.
- Search Topico/Subcampo/Area by valor using normalized Spanish values.
- PARTICIPO_EN.calidad values: 'responsable', 'integrante', 'otros'. Add {{calidad: 'responsable'}} only when the question asks for responsables/liderado/responsable/directos. For general equipo/participaron/figura, do not filter by calidad.

INTENT RULES:
- "cuántos", "cuántas", "qué cantidad" => RETURN count(...) AS total; do not return chunks unless explicitly requested.
- "quién/quiénes" => return Investigador nodes or names.
- "qué proyectos/lista/muestra/nombra" => return Proyecto nodes.
- "qué tópicos/temas" => return Topico values or nodes.
- "qué subcampo" => return Subcampo.
- "qué área" => return Area.
- "en qué año" => return a.anio AS año.

RETRIEVAL-FIRST PATTERNS:
- Project by title:
  MATCH (p:Proyecto)
  WHERE toLower(p.titulo) CONTAINS 'fragmento normalizado'
- Then extend from p using the schema:
  responsables: MATCH (i:Investigador)-[:PARTICIPO_EN {{calidad: 'responsable'}}]->(p)
  equipo/participantes: MATCH (i:Investigador)-[:PARTICIPO_EN]->(p)
  año: MATCH (p)-[:INICIO_EN]->(a:Anio)
  tópicos: MATCH (p)-[:TIENE_TOPICO]->(t:Topico)
  subcampo: MATCH (p)-[:TIENE_TOPICO]->(:Topico)-[:PERTENECE_A_SUBCAMPO]->(s:Subcampo)
  área: MATCH (p)-[:PERTENECE_A_AREA]->(a:Area)
- Evidence chunks are optional:
  OPTIONAL MATCH (p)-[:TITULO_EXTRAIDO_DE]->(pc:Chunk)
  OPTIONAL MATCH (ic:Chunk)-[:EXTRAIDO_DE]->(i)
  OPTIONAL MATCH (tc:Chunk)-[:EXTRAIDO_DE]->(t)
- Return chunks only after matching them:
  RETURN ..., COLLECT(DISTINCT pc) + COLLECT(DISTINCT ic) + COLLECT(DISTINCT tc) AS chunks

SAFETY AGAINST COMMON FAILURES:
- Do not filter project title on c.texto when p.titulo exists.
- Do not add extra filters on p.descripcion unless the question explicitly asks about the description text.
- Do not use full long titles if a shorter distinctive fragment is enough; long exact fragments often over-filter.
- Use CONTAINS, not equality, for names and project titles.
- Use WHERE after the MATCH whose variables are already defined.
- Avoid disconnected comma patterns that create cartesian products. Use sequential MATCH clauses.
- Every variable in RETURN/WITH must already be bound.
- Use DISTINCT when listing entities.

QUESTION: {user_query}
<QUERY>

QUERY DESIGN CHECKLIST TO APPLY INTERNALLY BEFORE OUTPUT:
1. Identify target entity: Investigador / Proyecto / Topico / Subcampo / Area / Anio / count.
2. Identify anchor entity from the user question:
   - project title => anchor on Proyecto.titulo
   - researcher name => anchor on Investigador.nombre
   - topic/subfield/area => anchor on valor
   - year => anchor on Anio.anio
3. Build the shortest valid path using only schema directions.
4. Add role filter only if requested.
5. Add optional evidence chunks only after the main result is matched.
6. Verify no reversed relationship, no undefined variable, no pattern in RETURN, no over-specific c.texto filter.
7. Output the final query only.
"""

    def _fix_relationship_directions(self, query: str) -> str:
        """
        Corrige relaciones dirigidas cuando fueron generadas al revés.
        La corrección cambia únicamente las flechas y conserva:
        - El orden de los nodos.
        - Las variables.
        - Las etiquetas.
        - Las propiedades de ambos nodos.
        - La variable y las propiedades de la relación.
        - Las relaciones encadenadas dentro de un mismo MATCH.

        Esquema correcto:
        - (Investigador)-[:PARTICIPO_EN]->(Proyecto|Grupo)
        - (Proyecto|Grupo)-[:TIENE_TOPICO]->(Topico)
        - (Topico)-[:PERTENECE_A_SUBCAMPO]->(Subcampo)
        - (Proyecto|Grupo)-[:ES_DESCRITO_POR]->(Documento)
        - (Proyecto|Grupo)-[:INICIO_EN]->(Anio)
        - (Proyecto|Grupo)-[:PERTENECE_A_AREA]->(Area)
        - (Documento)-[:PRIMER_CHUNK]->(Chunk)
        - (Chunk)-[:SIGUIENTE_CHUNK]->(Chunk)
        - (Chunk)-[:DE_DOCUMENTO]->(Documento)
        - (Chunk)-[:EXTRAIDO_DE]->(Investigador|Topico)
        - (Proyecto|Grupo)-[:TITULO_EXTRAIDO_DE]->(Chunk)
        """
        if not query or not query.strip():
            return query

        # Una relación puede admitir más de una combinación de etiquetas.
        correct_directions: Dict[str, List[Tuple[str, str]]] = {
            "PARTICIPO_EN": [("Investigador", "Proyecto"), ("Investigador", "Grupo")],
            "TIENE_TOPICO": [("Proyecto", "Topico"), ("Grupo", "Topico")],
            "PERTENECE_A_SUBCAMPO": [("Topico", "Subcampo")],
            "ES_DESCRITO_POR": [("Proyecto", "Documento"), ("Grupo", "Documento")],
            "INICIO_EN": [("Proyecto", "Anio"), ("Grupo", "Anio")],
            "PERTENECE_A_AREA": [("Proyecto", "Area"), ("Grupo", "Area")],
            "PRIMER_CHUNK": [("Documento", "Chunk")],
            "SIGUIENTE_CHUNK": [("Chunk", "Chunk")],
            "DE_DOCUMENTO": [("Chunk", "Documento")],
            "EXTRAIDO_DE": [("Chunk", "Investigador"), ("Chunk", "Topico")],
            "TITULO_EXTRAIDO_DE": [("Proyecto", "Chunk"), ("Grupo", "Chunk")],
        }

        var_types: Dict[str, str] = {}

        node_declaration_pattern = re.compile(
            r"""\(\s*(?P<var>[A-Za-z_]\w*)\s*:\s*(?P<label>[A-Za-z_]\w*)""",
            re.IGNORECASE | re.VERBOSE,
        )

        for match in node_declaration_pattern.finditer(query):
            variable = match.group("var")
            label = match.group("label")
            var_types.setdefault(variable, label)

        node_properties = r"""(?:\s*\{(?:[^{}]|\{[^{}]*\})*\})?"""
        left_node = rf"""(?P<left_node>\(\s*(?:(?P<left>[A-Za-z_]\w*)\s*)?(?::\s*(?P<left_label>[A-Za-z_]\w*)(?:\s*:\s*[A-Za-z_]\w*)*)?{node_properties}\s*\))"""
        right_node = rf"""(?P<right_node>\(\s*(?:(?P<right>[A-Za-z_]\w*)\s*)?(?::\s*(?P<right_label>[A-Za-z_]\w*)(?:\s*:\s*[A-Za-z_]\w*)*)?{node_properties}\s*\))"""
        relationship = r"""(?P<relationship>\[\s*(?:(?P<rel_var>[A-Za-z_]\w*)\s*)?:\s*(?P<rel_type>[A-Za-z_]\w*)[^\]]*\])"""
        edge_pattern = re.compile(
            rf"""(?={left_node}\s*(?P<left_arrow><-|-)\s*{relationship}\s*(?P<right_arrow>->|-)\s*{right_node})""",
            re.IGNORECASE | re.VERBOSE,
        )

        edits: List[Tuple[int, int, str]] = []
        corrections_made: List[str] = []
        registered_edits: Set[Tuple[int, int, str]] = set()

        for match in edge_pattern.finditer(query):
            left_arrow = match.group("left_arrow")
            right_arrow = match.group("right_arrow")

            if (left_arrow, right_arrow) not in {("-", "->"), ("<-", "-")}:
                continue

            rel_type_original = match.group("rel_type")
            rel_type = rel_type_original.upper()

            expected_pairs = correct_directions.get(rel_type)
            if not expected_pairs:
                continue

            left_variable = match.group("left")
            right_variable = match.group("right")

            left_type = match.group("left_label")
            if left_type is None and left_variable is not None:
                left_type = var_types.get(left_variable)

            right_type = match.group("right_label")
            if right_type is None and right_variable is not None:
                right_type = var_types.get(right_variable)

            if left_type is None or right_type is None:
                continue

            expected_normalized = {
                (source.casefold(), target.casefold()) for source, target in expected_pairs
            }

            if (left_arrow, right_arrow) == ("-", "->"):
                actual_source = left_type
                actual_target = right_type
            else:
                actual_source = right_type
                actual_target = left_type

            actual_pair = (
                actual_source.casefold(),
                actual_target.casefold(),
            )

            if actual_pair in expected_normalized:
                continue

            reversed_pair = (
                actual_pair[1],
                actual_pair[0],
            )

            if reversed_pair not in expected_normalized:
                continue

            if (left_arrow, right_arrow) == ("-", "->"):
                new_left_arrow = "<-"
                new_right_arrow = "-"
            else:
                new_left_arrow = "-"
                new_right_arrow = "->"

            left_edit = (
                match.start("left_arrow"),
                match.end("left_arrow"),
                new_left_arrow,
            )
            right_edit = (
                match.start("right_arrow"),
                match.end("right_arrow"),
                new_right_arrow,
            )

            if left_edit not in registered_edits:
                edits.append(left_edit)
                registered_edits.add(left_edit)

            if right_edit not in registered_edits:
                edits.append(right_edit)
                registered_edits.add(right_edit)

            corrections_made.append(
                "{actual_source}-[:{rel_type_original}]->{actual_target} "
                f"se corrigió a "
                f"{actual_target}-[:{rel_type_original}]->{actual_source}"
            )

        fixed = query

        for start, end, replacement in sorted(edits, key=lambda edit: edit[0], reverse=True):
            fixed = fixed[:start] + replacement + fixed[end:]

        if corrections_made:
            logger.info(
                "Direcciones corregidas automáticamente: %s",
                " | ".join(corrections_made),
            )

        return fixed

    @staticmethod
    def _use_display_fields_for_return(cypher_query: str) -> str:
        """
        Reemplaza en la cláusula RETURN los campos normalizados por campos de visualización:
        - Investigador.nombre -> Investigador.nombre_de_despliegue
        - Proyecto.titulo -> Proyecto.titulo_de_despliegue
        """

        parts = re.split(r"\b(RETURN)\b", cypher_query, maxsplit=1, flags=re.IGNORECASE)

        if len(parts) != 3:
            return cypher_query

        before_return = parts[0]
        return_keyword = parts[1]
        after_return = parts[2]

        investigador_vars = set(
            re.findall(r"\(\s*(\w+)\s*:\s*Investigador\b", before_return, re.IGNORECASE)
        )

        proyecto_vars = set(
            re.findall(r"\(\s*(\w+)\s*:\s*Proyecto\b", before_return, re.IGNORECASE)
        )

        for var in investigador_vars:
            after_return = re.sub(
                rf"\b{var}\.nombre\b", f"{var}.nombre_de_despliegue", after_return
            )

        for var in proyecto_vars:
            after_return = re.sub(
                rf"\b{var}\.titulo\b", f"{var}.titulo_de_despliegue", after_return
            )

        return before_return + return_keyword + after_return

    def _fix_cypher_query(self, broken_query: str, syntax_error: str) -> str:
        """
        Pide al LLM que corrija una query Cypher con error de sintaxis.
        """
        prompt = f"""The following Cypher query produced a syntax error. Fix it.

SCHEMA:
{self._fetch_schema()}

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

                logger.info(f"Respuesta fallida cruda del LLM al corregir: {response}")

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

        fixed_query = self._use_display_fields_for_return(fixed_query)

        fixed_query = self._fix_relationship_directions(fixed_query)

        is_safe, error = CypherQueryValidator.is_safe(fixed_query)
        if not is_safe:
            logger.warning(
                f"La query corregida falló la validación de seguridad. Motivo {error} | Query: {fixed_query}"
            )
            raise ValueError(f"Query corregida no es segura: {error}")

        logger.info(f"Query corregida por LLM: {fixed_query}")
        return fixed_query

    def execute_cypher_query(self, cypher_query: str) -> List[Any]:
        """
        Ejecuta una query Cypher y retorna resultados (como neo4j Record objects).
        """
        is_safe, error = CypherQueryValidator.is_safe(cypher_query)
        if not is_safe:
            logger.warning(
                f"La query falló la validación de seguridad. Motivo {error} | Query: {cypher_query}"
            )
            raise ValueError(f"Query no pasó validación de seguridad: {error}")

        with self.driver.session() as session:
            result = session.run(cypher_query)
            records = list(result)

        logger.info(f"Query ejecutada, {len(records)} registros obtenidos")
        return records

    def _build_aggregation_context(self, records: List[Any]) -> str:
        """Construye contexto a partir de resultados de agregación o nodos sin chunks."""

        def _format_node(node: Node) -> str:
            labels = list(node.labels)
            label = labels[0] if labels else "Node"
            props = dict(node)
            display = (
                props.get("valor")
                or props.get("nombre_de_despliegue")
                or props.get("titulo_de_despliegue")
                or props.get("nombre")
                or props.get("titulo")
                or props.get("id", "")
            )
            return f"{display} ({label})"

        if not records:
            return "No aggregation results found."

        context_parts = ["=== QUERY RESULTS ===\n"]

        for idx, record in enumerate(records, 1):
            values = []
            for key in record.keys():
                value = record[key]
                if isinstance(value, Node):
                    values.append(f"{key}: {_format_node(value)}")
                elif isinstance(value, list):
                    items = [_format_node(v) if isinstance(v, Node) else str(v) for v in value]
                    values.append(f"{key}: [{', '.join(items)}]")
                else:
                    values.append(f"{key}: {value}")
            context_parts.append(f"{idx}. {', '.join(values)}")

        return "\n".join(context_parts)

    def extract_chunks_and_entities_from_results(
        self, records: List[Any]
    ) -> tuple[List[GraphRAGChunk], Dict[tuple[str, str], dict], Dict[str, List[tuple[str, str]]]]:
        print("RECORD", records)

        """Extrae chunks y evidencia de entidades desde los resultados de Cypher."""
        chunks_dict: Dict[str, GraphRAGChunk] = {}  # chunk_id -> GraphRAGChunk
        evidence_entities: Dict[tuple[str, str], dict] = {}  # (entity_id, label) -> props
        chunk_to_entities: Dict[str, List[tuple[str, str]]] = {}  # chunk_id -> [(entity_id, label)]

        for record in records:
            chunks_in_record = []
            # direct = vienen de MATCH directo (implica relación EXTRAIDO_DE/TITULO_EXTRAIDO_DE)
            entities_direct = []
            # collected = vienen de COLLECT() — son del proyecto/consulta, no del chunk
            entities_collected = []

            for key in record.keys():
                value = record[key]

                if isinstance(value, Node):
                    labels = list(value.labels)
                    if "Chunk" in labels:
                        chunks_in_record.append(value)
                    elif any(
                        label in labels
                        for label in [
                            "Investigador",
                            "Topico",
                            "Proyecto",
                            "Grupo",
                            "Documento",
                            "Anio",
                            "Area",
                        ]
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
                                    "Grupo",
                                    "Documento",
                                    "Anio",
                                    "Area",
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
                        if lbl
                        in [
                            "Investigador",
                            "Topico",
                            "Proyecto",
                            "Grupo",
                            "Documento",
                            "Anio",
                            "Area",
                        ]
                    ),
                    "",
                )
                entity_id = props.get("id", "")
                if entity_label == "Investigador":
                    entity_id = props.get("nombre", entity_id) or entity_id
                elif entity_label in ("Proyecto", "Grupo"):
                    raw_id = props.get("id", "")
                    title = props.get("titulo", "")
                    entity_id = f"{title} ({raw_id})" if title else raw_id
                elif entity_label == "Documento":
                    entity_id = props.get("id", entity_id) or entity_id
                elif entity_label in ("Topico", "Area"):
                    entity_id = props.get("valor", entity_id) or entity_id
                elif entity_label == "Anio":
                    entity_id = props.get("anio", entity_id) or entity_id
                return entity_id, entity_label, props

            # Agregar entidades collected a evidence_entities (para el contexto del LLM)
            # pero NO asociarlas a chunks individuales
            for entity_node in entities_collected:
                entity_id, entity_label, props = resolve_entity_id(entity_node)
                if entity_id and entity_label:
                    evidence_entities[(entity_id, entity_label)] = props

            for chunk_node in chunks_in_record:
                chunk_id = chunk_node.get("id", "")
                if not chunk_id:
                    continue

                if chunk_id not in chunks_dict:
                    text = chunk_node.get("texto", "")
                    page_numbers = chunk_node.get("paginas")
                    page = int(page_numbers[0]) if page_numbers else 1
                    if text:
                        chunks_dict[chunk_id] = GraphRAGChunk(
                            chunk_id=chunk_id,
                            text=text,
                            page=page,
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
            "Grupo": "Grupos",
            "Investigador": "Investigadores",
            "Topico": "Tópicos",
            "Documento": "Documentos",
            "Anio": "Año",
            "Area": "Áreas",
        }

        lines = ["=== ENTIDADES ENCONTRADAS EN EL GRAFO ==="]

        for label in ["Proyecto", "Grupo", "Investigador", "Topico", "Documento", "Anio", "Area"]:
            entries = [
                ((eid, lbl), props)
                for (eid, lbl), props in evidence_entities.items()
                if lbl == label
            ]
            if not entries:
                continue
            lines.append(f"\n{label_display[label]}:")
            for (eid, _), props in sorted(entries, key=lambda x: x[0][0]):
                if label in ("Proyecto", "Grupo"):
                    pid = props.get("id", "")
                    title = props.get("titulo", "") or props.get("nombre", "")
                    if title and pid:
                        lines.append(f"  - {title} ({pid})")
                    else:
                        lines.append(f"  - {pid or title}")
                else:
                    props_str = " | ".join(
                        f"{k}: {v}" for k, v in props.items() if v is not None and k != "texto"
                    )
                    lines.append(f"  - {props_str}")

        return "\n".join(lines)

    def build_messages_for_answer(
        self, user_query: str, entity_context: str
    ) -> List[Dict[str, str]]:
        """
        Construir mensajes para el LLM usando SOLO entidades y relaciones del grafo.
        Usa el prompt "grounded" seleccionado en la calibración.
        El texto de los chunks va al frontend, no aqui.
        """
        system_prompt = (
            "Eres un asistente académico riguroso. Respondé en ESPAÑOL usando ÚNICAMENTE "
            "los datos presentes en los resultados del grafo. Está PROHIBIDO inventar, "
            "inferir o completar información que no aparezca literalmente. Citá los valores "
            "tal como figuran en los resultados e incluí TODOS sin excepción. Si un dato no "
            "está en los resultados, no lo menciones."
        )

        user_prompt = (
            f"PREGUNTA: {user_query}\n\nRESULTADOS DEL GRAFO:\n{entity_context}\n\n"
            "Respondé usando solo los valores que aparecen arriba, copiándolos literalmente. "
            "No agregues nada que no esté en los resultados."
        )

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
            if str(e).startswith("NOT_IN_SCHEMA"):
                return (
                    GraphRAGResult(
                        answer=(
                            "La consulta solicitada está fuera del alcance del esquema actual del grafo."
                        ),
                        chunks=[],
                        cypher_query="",
                        chunk_to_entities={},
                    ),
                    [],
                    "",
                )
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
                if not records:
                    _too_complex_result.answer = (
                        "La consulta no puede responderse con la información del grafo."
                    )
                break
            except CypherSyntaxError as exc:
                logger.warning(
                    f"Error de sintaxis Cypher (intento {attempt + 1}/{MAX_SYNTAX_RETRIES}): {exc}"
                )
                logger.info(f"Query original que provocó el error de sintaxis:\n{cypher_query}")
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

                # Se usa el prompt "grounded" seleccionado en la calibración,
                # igual que para las respuestas sobre entidades.
                messages = self.build_messages_for_answer(user_query, context)

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
                answer="No se encontró ningún elemento que cumpla con los criterios de la consulta.",
                chunks=[],
                cypher_query=cypher_query,
                chunk_to_entities={},
            )

        if evidence_entities:
            entity_context = self.build_entity_context(evidence_entities, chunk_to_entities)
        else:
            entity_context = self._build_aggregation_context(records)

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
