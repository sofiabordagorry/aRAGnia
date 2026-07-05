import json
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
PROMPTS_JSON_PATH = EVAL_DIR / "prompts_variants.json"

baseline_template = """Generate a Cypher query for Neo4j to answer this question.
SCHEMA:
{schema}

SCHEMA NOTES:
- Anio uses property "anio" (NOT "valor" or "id"): Anio.anio = '2014'
- Investigador.id follows '{pais}_{tipo_documento}_{documento}'; search by Investigador.nombre (lowercase, no accents)
- PARTICIPO_EN has a required property "calidad" with values: 'responsable', 'integrante', 'otros'. ONLY filter by calidad when the question asks for a specific role (e.g. "responsable de", "integrantes del proyecto X"): -[:PARTICIPO_EN {calidad: 'responsable'}]->. For general "who participated / quiénes participaron" questions, use plain -[:PARTICIPO_EN]-> WITHOUT filtering.
- Proyecto.titulo contains the project title; Proyecto.id follows 'proy_2020_513'
- Grupo.titulo contains the group title; Grupo.id follows 'gi_2014_133'
- Use Proyecto label for project entities (proy_* IDs) and Grupo label for group entities (gi_* IDs)
- Topico.valor, Subcampo.valor and Area.valor are in Spanish, lowercase, no accents: 'biotecnologia', 'ciencias naturales'
- Documento.tipo is one of: 'informe', 'propuesta', 'resumen', 'tabla'

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
11. If the information requested does NOT exist in the schema, respond with: <QUERY>NOT_IN_SCHEMA</QUERY>
12. If the question cannot be answered with a single query but IS related to the schema, respond with: <QUERY>UNSUPPORTED</QUERY>
12. When the question asks "how many" / "cuántos" / "qué cantidad", use count() aggregation (e.g., RETURN count(p) AS total). Do NOT return individual entities unless the question explicitly asks to list them.
13. ALWAYS filter values using WHERE.
14. ALWAYS normalize text values: lowercase, no accents, never translate.
15. Topics are stored in Spanish, lowercase and without accents: 'biotecnologia', 'ingenieria', 'medicina', etc.
16. Subfields are stored in Spanish, lowercase and without accents.
17 Search project titles/names with toLower(p.titulo) CONTAINS.
18. Convert Anio.anio with toInteger() for numeric comparisons.

{fewshot}

QUESTION: {user_query}

CRITICAL DECISION - COUNT vs LIST:
- If question asks "cuántos", "cuántas", "how many", "qué cantidad" → USE count() and RETURN count(x) AS total (NO chunks needed)
- If question asks "cuáles", "qué proyectos", "quiénes", "list", "muéstrame" → RETURN entities + COLLECT(c) AS chunks
- If question asks "quién/quiénes" (WHO) → RETURN investigators (i), NOT projects
- If question asks "qué año" (WHAT year) → RETURN year value directly (a.anio or a)
- Analyze the question intent carefully before generating the query
RETURN RULES:
- "¿Quiénes participaron?" → RETURN investigadores (i), NOT proyecto (p)
- "¿En qué año?" → RETURN año (a.anio AS año) or (a) with OPTIONAL MATCH for chunks
- "¿Cuántos proyectos?" → RETURN count(p) AS total
- "¿Qué investigadores con más proyectos?" → RETURN i.nombre, count(p) ORDER BY count(p) DESC LIMIT N
- NEVER return p.id unless the user explicitly asks for the project identifier

CRITICAL SYNTAX:
- Wrap your query in <QUERY> and </QUERY> tags
- Every variable in WITH/RETURN must be defined in a previous MATCH
- Use [:EXTRAIDO_DE]->(entity) for investigators/topics, [:TITULO_EXTRAIDO_DE]->(chunk) for projects
- Topico uses {valor: '...'}, Anio uses {anio: '...'}, all others use {id: '...'}
- NEVER name a relationship variable (never write -[r:TYPE]-> or -[rel:TYPE]->), always use -[:TYPE]->
- NEVER use a variable as both a relationship and a node
- NEVER generate paths like (a)-[:REL]->(b)-[:REL2]->(c).
- ALWAYS use WHERE for filtering
- ALWAYS use toLower(p.titulo) CONTAINS 'normalized project text' for project titles/names
- If the user asks for information not represented in the schema
  (for example salaries, emails if not stored, countries, universities, budgets, etc.),
  respond with <QUERY>NOT_IN_SCHEMA</QUERY>
<QUERY>"""

concise_template = """Generate a precise Neo4j Cypher query to answer the user's question.

SCHEMA:
{schema}

{fewshot}

CRITICAL RULES:
1. Read-only queries only (MATCH, OPTIONAL MATCH, WHERE, RETURN).
2. Query MUST be enclosed exactly between <QUERY> and </QUERY> tags.
3. For text search on project titles, use: `toLower(p.titulo) CONTAINS 'text'`.
4. If asking 'how many' (cuántos), use `count()` aggregation.
5. If listing entities (who, which, list), return the entities and `COLLECT(c) AS chunks` (where c:Chunk).
6. Do NOT define relationship variables (use `-[:TYPE]->` not `-[r:TYPE]->`).
7. Ensure all node variables in RETURN/WITH are defined in a prior MATCH.

QUESTION: {user_query}
<QUERY>"""

prompts = {
    "baseline": baseline_template,
    "concise": concise_template
}

with open(PROMPTS_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(prompts, f, ensure_ascii=False, indent=4)

print(f"¡JSON generado exitosamente en: {PROMPTS_JSON_PATH}!")