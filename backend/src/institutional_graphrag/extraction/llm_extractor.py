"""Extracción de investigadores usando LLMs."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import regex

from institutional_graphrag.graph.schema import (
    EXTRAIDO_DE,
    PARTICIPO_EN,
    PERTENECE_A_DOMINIO,
    Dominio,
    Entity,
    Investigador,
    Relationship,
    Topico,
)
from institutional_graphrag.llm.llm_provider import get_llm_client

logger = logging.getLogger(__name__)
TOPICS_PATH = Path(__file__).parents[4] / "data" / "openalex_topics.json"


@dataclass
class ResearcherMention:
    """Mención de investigador en un chunk."""

    name: str
    evidence: str
    chunk_id: str


@dataclass
class TopicMention:
    """Mención de tópico en un chunk."""

    topic: str
    evidence: str
    chunk_id: str


@dataclass
class LLMExtractionResult:
    """Resultado de extracción LLM."""

    researchers: List[ResearcherMention]
    topics: List[TopicMention]
    errors: List[Dict[str, Any]]


class LLMEntityExtractor:
    """Extractor de investigadores usando LLMs."""

    def __init__(
        self,
        *,
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        available_topics: Optional[List[str]] = None,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_client = get_llm_client(model=llm_model)
        subfields, subfields_map = self.get_topics(TOPICS_PATH)
        self.subfields_map = subfields_map
        if available_topics is None:
            available_topics = subfields

        self.available_topics = available_topics

    def get_topics(self, file_path: Path) -> tuple[list, dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            fields_dict = json.load(f)
        """Crea un mapeo de fields y subfields para búsqueda rápida."""
        subfield_to_field = {}  # Mapeo de subfields a fields
        subfields_list = []  # Lista de subfields
        for field, details in fields_dict.items():
            for subfield in details.get("subfields", []):
                subfield_normalized = subfield.lower().strip()
                subfield_to_field[subfield_normalized] = field  # Mapear subfield a field
                subfields_list.append(subfield)  # Agregar subfield a la lista
        return subfields_list, subfield_to_field

    def extract_researchers_from_chunk(self, chunk_text: str, chunk_id: str) -> LLMExtractionResult:
        """Extraer investigadores de un chunk."""
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente especializado en análisis de documentos académicos.",
            },
            {"role": "user", "content": self._build_prompt(chunk_text)},
        ]

        try:
            response = self.llm_client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self._parse_response(response, chunk_id, chunk_text)

        except Exception as e:
            logger.error(f"Error en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                researchers=[],
                topics=[],
                errors=[{"type": "LLMExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def extract_topics_from_chunk(self, chunk_text: str, chunk_id: str) -> LLMExtractionResult:
        """Extraer tópicos de un chunk."""
        messages = [
            {
                "role": "system",
                "content": "Eres un asistente especializado en análisis de documentos académicos.",
            },
            {"role": "user", "content": self._build_topic_prompt(chunk_text)},
        ]

        try:
            response = self.llm_client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self._parse_topic_response(response, chunk_id, chunk_text)

        except Exception as e:
            logger.error(f"Error en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                researchers=[],
                topics=[],
                errors=[{"type": "LLMExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def _build_prompt(self, chunk_text: str) -> str:
        """Construir prompt para extracción de investigadores."""
        return f"""Extract researchers who PARTICIPATED in this research project from the text below.

RULES:
- Extract ONLY project team members (NOT cited authors from references)
- One person per entry
- Evidence must be actual text containing the person's name

TEXT:
{chunk_text}

IMPORTANT: You MUST wrap your response with <JSON> and </JSON> tags.

OUTPUT FORMAT:
<JSON>
{{
  "researchers": [
    {{"name": "Person Name", "evidence": "actual text mentioning Person Name"}}
  ]
}}
</JSON>

If no researchers found:
<JSON>{{"researchers": []}}</JSON>
"""

    def _build_topic_prompt(self, chunk_text: str) -> str:
        """Construir prompt para extracción de tópicos."""
        if self.available_topics:
            topics_list = "\n".join(f"- {topic}" for topic in self.available_topics)

            return f"""Match research topics from Spanish text to this English topic list.

ALLOWED TOPICS:
{topics_list}

TEXT (Spanish):
{chunk_text}

RULES:
- ONLY use topics from the list above
- DO NOT invent new topics
- Match by research field, not literal translation

IMPORTANT: You MUST wrap your response with <JSON> and </JSON> tags.

OUTPUT FORMAT:
<JSON>
{{
  "topics": [
    {{"topic": "Topic from list", "evidence": "Spanish text fragment"}}
  ]
}}
</JSON>

If no match:
<JSON>{{"topics": []}}</JSON>"""
        else:
            return f"""You are an information extraction system.

    Task:
    Identify the main research topics mentioned in the text.

    TEXT:
    {chunk_text}

    Rules:
    - Be concise and specific
    - Use simple, clear topic names
    - Extract only topics that are clearly mentioned in the text

    OUTPUT FORMAT - CRITICAL:
    - Wrap your JSON response in <JSON> tags
    - Format: <JSON>{{...}}</JSON>
    - Do NOT add any text before <JSON> or after </JSON>
    - Do NOT add explanations or comments
    - The JSON inside must be valid

    Exact format to follow:
    <JSON>
    {{
    "topics": [
        {{
        "topic": "Topic name",
        "evidence": "Exact text fragment mentioning it"
        }}
    ]
    }}
    </JSON>

    If no topics are identified:
    <JSON>{{"topics": []}}</JSON>
    
    Remember: Always use <JSON> tags around your response.
    """

    def _parse_response(
        self, response: str, chunk_id: str, chunk_text: str = ""
    ) -> LLMExtractionResult:
        """Parsear respuesta del LLM."""
        errors = []
        researchers = []

        # Intentar extraer JSON de tags <JSON>...</JSON>
        json_match = re.search(r"<JSON>\s*(\{.*?\})\s*</JSON>", response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                errors.append(
                    {
                        "type": "JSONDecodeError",
                        "chunk_id": chunk_id,
                        "message": f"JSON dentro de tags inválido: {str(e)}",
                    }
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)
        else:
            # Fallback: buscar JSON sin tags (comportamiento anterior)
            json_start = response.find("{")

            if json_start == -1:
                errors.append(
                    {
                        "type": "InvalidJSON",
                        "chunk_id": chunk_id,
                        "message": "No hay JSON en respuesta (falta <JSON> tags)",
                    }
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)

            # Intentar encontrar el primer objeto JSON válido
            data = None
            for json_end in range(len(response), json_start, -1):
                candidate = response[json_start:json_end]
                if candidate.rstrip().endswith("}"):
                    try:
                        data = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        continue

            if data is None:
                errors.append(
                    {
                        "type": "JSONDecodeError",
                        "chunk_id": chunk_id,
                        "message": "No se pudo parsear JSON válido",
                    }
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)

        try:
            researchers_data = data.get("researchers", [])

            if not isinstance(researchers_data, list):
                errors.append(
                    {
                        "type": "InvalidJSON",
                        "chunk_id": chunk_id,
                        "message": "'researchers' no es lista",
                    }
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)

            for item in researchers_data:
                if not isinstance(item, dict):
                    continue

                # Manejar casos donde name puede ser lista, None, u otro tipo
                raw_name = item.get("name", "")
                if isinstance(raw_name, list):
                    raw_name = raw_name[0] if raw_name else ""
                name = str(raw_name).strip() if raw_name else ""

                raw_evidence = item.get("evidence", "")
                if isinstance(raw_evidence, list):
                    raw_evidence = raw_evidence[0] if raw_evidence else ""
                evidence = (
                    str(raw_evidence).strip() if raw_evidence else f"Mencionado en {chunk_id}"
                )

                # Filtrar nombres genéricos sin sentido
                invalid_patterns = [
                    "nombre completo",
                    "nombre del investigador",
                    "no se mencionan",
                    "no se menciona",
                    "investigador",
                    "researcher",
                    "name",
                    "et al",
                    "and collaborators",
                    "y colaboradores",
                    "responsables del grupo",
                    "miembros del equipo",
                    "equipo de investigación",
                    "grupo de investigación",
                    "el equipo",
                    "los investigadores",
                    "el grupo",
                    "nuestro grupo",
                    "grupo del proyecto",
                    "casos similares",
                ]

                # Validar que no sea una lista de múltiples nombres
                if ";" in name or " and " in name.lower():
                    errors.append(
                        {
                            "type": "MultipleNamesInOne",
                            "chunk_id": chunk_id,
                            "message": f"Múltiples nombres en una entidad: '{name}' (debe ser un nombre por entrada)",
                        }
                    )
                    continue

                # Validar que no sea una referencia bibliográfica (patrones comunes)
                bibliographic_patterns = [
                    (r"\bet al\b", 0),  # et al.
                    (r"\d{4}\)", 0),  # año entre paréntesis como (2020)
                    (r"[A-Z]\.\s*[A-Z]\.", re.IGNORECASE),  # iniciales como J. K.
                    (r"^[A-Z]+,\s*[A-Z]\.$", re.IGNORECASE),  # MAHLER, G. o SUESCUN, L.
                    (r"^[A-Z]\s+[A-Z]+$", re.IGNORECASE),  # J BREM (inicial + apellido sin puntos)
                    (
                        r"^[A-Z]\.\s+[A-Z]+",
                        re.IGNORECASE,
                    ),  # G. SERRA / C. FAGUNDEZ (inicial + apellido)
                    (
                        r"^[A-Z]\.\s+[A-Z]\.\s+[A-Z]+",
                        re.IGNORECASE,
                    ),  # J. M. SMITH (dos iniciales + apellido)
                ]
                is_bibliographic = False
                for pattern, flags in bibliographic_patterns:
                    if re.search(pattern, name, flags):
                        is_bibliographic = True
                        break

                if is_bibliographic:
                    errors.append(
                        {
                            "type": "BibliographicReference",
                            "chunk_id": chunk_id,
                            "message": f"Posible referencia bibliográfica, no participante: '{name}'",
                        }
                    )
                    continue

                # Validar que no sea solo apellido(s) sin nombre
                # Detectar: "DEL PUERTO GARCÍA", "NOBOA ALDECOA", etc.
                name_parts = name.split()
                if len(name_parts) >= 2 and all(part.isupper() for part in name_parts):
                    # Si todos son mayúsculas y son 2-3 palabras, podría ser solo apellidos
                    # Verificar que al menos una parte no sea preposición común
                    prepositions = {
                        "DE",
                        "DEL",
                        "LA",
                        "LAS",
                        "LOS",
                        "Y",
                        "E",
                        "DA",
                        "DI",
                        "VON",
                        "VAN",
                    }
                    non_prep_parts = [p for p in name_parts if p not in prepositions]

                    # Si solo hay 2 partes no-preposición, probablemente son solo apellidos
                    if len(non_prep_parts) == 2 and len(name_parts) <= 3:
                        errors.append(
                            {
                                "type": "IncompleteNameSurnameOnly",
                                "chunk_id": chunk_id,
                                "message": f"Nombre incompleto (solo apellidos): '{name}'",
                            }
                        )
                        continue

                # Validar que no contenga caracteres corruptos
                if re.search(r"[\{\}\[\]\u51fd\u9601\ufffd]", name):
                    errors.append(
                        {
                            "type": "CorruptedCharacters",
                            "chunk_id": chunk_id,
                            "message": f"Nombre con caracteres corruptos: '{name}'",
                        }
                    )
                    continue

                # Validar que no sea especie biológica
                # Patrón: letra mayúscula + punto + palabra (C. elegans, E. granulosus)
                if re.match(r"^[A-Z]\.[\s]?[a-z]+", name):
                    errors.append(
                        {
                            "type": "BiologicalSpecies",
                            "chunk_id": chunk_id,
                            "message": f"Especie biológica, no investigador: '{name}'",
                        }
                    )
                    continue

                # Validar que no sea compuesto químico
                # Patrones: termina con letra mayúscula sola, contiene números/símbolos químicos
                chemical_patterns = [
                    r"\b[A-Z]$",  # termina con letra sola como "Aeruciclamida B"
                    r"^[A-Z]{2,}$",  # siglas como "DAST"
                    r"DIELS.*ALDER",  # reacciones químicas
                ]
                if any(re.search(pattern, name, re.IGNORECASE) for pattern in chemical_patterns):
                    # Excepción: si contiene espacios y palabras normales, podría ser nombre real
                    if not (len(name_parts) >= 2 and any(len(p) > 3 for p in name_parts)):
                        errors.append(
                            {
                                "type": "Invalid Name",
                                "chunk_id": chunk_id,
                                "message": f"Nombre inválido: '{name}'",
                            }
                        )
                        continue

                # Validar que no sea institución/organización
                institution_keywords = [
                    "CSIC",
                    "ANII",
                    "DICYT",
                    "LABORATORIO",
                    "FACULTAD",
                    "UNIVERSIDAD",
                    "INSTITUTO",
                    "CENTRO",
                    "DEPARTAMENTO",
                    "ACCELERATOR",
                    "PROGRAMA",
                    "POLO TECNOLÓGICO",
                    "ESTUDIANTE",
                    "GRUPO I+D",
                ]
                if any(keyword in name.upper() for keyword in institution_keywords):
                    errors.append(
                        {
                            "type": "InstitutionOrOrganization",
                            "chunk_id": chunk_id,
                            "message": f"Institución u organización, no investigador: '{name}'",
                        }
                    )
                    continue

                # Validar que no sea técnica/método
                if len(name) > 30 or ("DE " in name.upper() and name.count(" ") > 5):
                    # Frases largas o con muchas preposiciones son títulos/técnicas
                    errors.append(
                        {
                            "type": "TechniqueOrSection",
                            "chunk_id": chunk_id,
                            "message": f"Título de sección o técnica, no investigador: '{name[:60]}...'",
                        }
                    )
                    continue

                # Validar que no sea dato/estadística
                if re.search(r"\d+\s*%|^[A-Z]\.\s*\d+", name):
                    errors.append(
                        {
                            "type": "DataOrStatistic",
                            "chunk_id": chunk_id,
                            "message": f"Dato estadístico, no investigador: '{name}'",
                        }
                    )
                    continue

                # Validar que no sea mes/fecha
                months = {
                    "ENERO",
                    "FEBRERO",
                    "MARZO",
                    "ABRIL",
                    "MAYO",
                    "JUNIO",
                    "JULIO",
                    "AGOSTO",
                    "SEPTIEMBRE",
                    "OCTUBRE",
                    "NOVIEMBRE",
                    "DICIEMBRE",
                }
                if name.upper() in months:
                    errors.append(
                        {
                            "type": "MonthOrDate",
                            "chunk_id": chunk_id,
                            "message": f"Mes/fecha, no investigador: '{name}'",
                        }
                    )
                    continue

                # VALIDACIÓN: Verificar que evidencia y nombre estén en el chunk
                if evidence and "Mencionado en" not in evidence and chunk_text:
                    # Normalizar texto: minúsculas y limpiar caracteres de control
                    chunk_normalized = re.sub(r"[\t\r\n]+", " ", chunk_text.lower())
                    chunk_normalized = re.sub(r"\s+", " ", chunk_normalized)

                    evidence_normalized = re.sub(r"[\t\r\n]+", " ", evidence.lower())
                    evidence_normalized = re.sub(r"\s+", " ", evidence_normalized)

                    name_normalized = name.lower()

                    # 1. Verificar que la evidencia esté en el chunk
                    evidence_words = [
                        w
                        for w in re.findall(r"\b\w+\b", evidence_normalized)
                        if len(w) > 3 and not w.isdigit()
                    ]

                    if len(evidence_words) >= 3:
                        words_in_chunk = sum(1 for w in evidence_words if w in chunk_normalized)
                        match_ratio = words_in_chunk / len(evidence_words)

                        if match_ratio < 0.7:
                            errors.append(
                                {
                                    "type": "EvidenceNotInChunk",
                                    "chunk_id": chunk_id,
                                    "message": f"La evidencia '{evidence[:80]}...' no está en el chunk (solo {match_ratio:.0%} de palabras coinciden)",
                                }
                            )
                            continue

                    # 2. Verificar que el nombre esté en el chunk
                    name_parts = [p.strip() for p in name_normalized.split() if len(p.strip()) > 2]

                    # Cantidad de errores (substitute, insert, delete) tolerados
                    # Para valores especificos para cada uno en vez de "e<=3" en el pattern usar "s<=3,i<=3,d<=3"
                    allowed_errors = 3

                    patterns = []
                    name_in_chunk = []
                    failed_check = False

                    for part in name_parts:
                        pattern = f"({part}){{e<={allowed_errors}}}"
                        patterns.append(pattern)

                    if patterns:
                        for pattern in patterns:
                            match = regex.search(pattern, chunk_normalized, regex.BESTMATCH)
                            if match is not None:
                                name_in_chunk.append(match.group())
                            else:
                                errors.append(
                                    {
                                        "type": "NameNotInChunk",
                                        "chunk_id": chunk_id,
                                        "message": f"El nombre '{name}' no aparece en el chunk",
                                    }
                                )
                                failed_check = True
                                break
                        if failed_check:
                            continue
                        else:
                            name = " ".join(name_in_chunk)

                # Si llegamos hasta acá, pasó todas las validaciones
                if name and not any(pattern in name.lower() for pattern in invalid_patterns):
                    researchers.append(
                        ResearcherMention(name=name, evidence=evidence, chunk_id=chunk_id)
                    )

        except json.JSONDecodeError as e:
            errors.append({"type": "JSONDecodeError", "chunk_id": chunk_id, "message": str(e)})

        return LLMExtractionResult(researchers=researchers, topics=[], errors=errors)

    def _parse_topic_response(
        self, response: str, chunk_id: str, chunk_text: str = ""
    ) -> LLMExtractionResult:
        """Parsear respuesta del LLM para tópicos."""
        errors = []
        topics = []

        # Intentar extraer JSON de tags <JSON>...</JSON>
        json_match = re.search(r"<JSON>\s*(\{.*?\})\s*</JSON>", response, re.DOTALL)

        if json_match:
            json_str = json_match.group(1)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                errors.append(
                    {
                        "type": "JSONDecodeError",
                        "chunk_id": chunk_id,
                        "message": f"JSON dentro de tags inválido: {str(e)}",
                    }
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)
        else:
            # Fallback: buscar JSON sin tags (comportamiento anterior)
            json_start = response.find("{")

            if json_start == -1:
                errors.append(
                    {
                        "type": "InvalidJSON",
                        "chunk_id": chunk_id,
                        "message": "No hay JSON en respuesta (falta <JSON> tags)",
                    }
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)

            # Intentar encontrar el primer objeto JSON válido
            data = None
            for json_end in range(len(response), json_start, -1):
                candidate = response[json_start:json_end]
                if candidate.rstrip().endswith("}"):
                    try:
                        data = json.loads(candidate)
                        break
                    except json.JSONDecodeError:
                        continue

            if data is None:
                errors.append(
                    {
                        "type": "JSONDecodeError",
                        "chunk_id": chunk_id,
                        "message": "No se pudo parsear JSON válido",
                    }
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)

        try:
            topics_data = data.get("topics", [])

            if not isinstance(topics_data, list):
                errors.append(
                    {"type": "InvalidJSON", "chunk_id": chunk_id, "message": "'topics' no es lista"}
                )
                return LLMExtractionResult(researchers=[], topics=[], errors=errors)

            for item in topics_data:
                if not isinstance(item, dict):
                    continue

                # Manejar casos donde topic puede ser lista, None, u otro tipo
                raw_topic = item.get("topic", "")
                if isinstance(raw_topic, list):
                    raw_topic = raw_topic[0] if raw_topic else ""
                topic = str(raw_topic).strip() if raw_topic else ""

                raw_evidence = item.get("evidence", "")
                if isinstance(raw_evidence, list):
                    raw_evidence = raw_evidence[0] if raw_evidence else ""
                evidence = (
                    str(raw_evidence).strip() if raw_evidence else f"Mencionado en {chunk_id}"
                )

                if topic:
                    # Validar que el tópico esté en la lista permitida
                    if self.available_topics:
                        # Comparación case-insensitive
                        topic_lower = topic.lower()
                        available_lower = [t.lower() for t in self.available_topics]

                        if topic_lower not in available_lower:
                            errors.append(
                                {
                                    "type": "InvalidTopic",
                                    "chunk_id": chunk_id,
                                    "message": f"Tópico '{topic}' no está en la lista permitida (inventado por LLM)",
                                }
                            )
                            continue

                    # VALIDACIÓN: La evidencia debe estar en el chunk original
                    if (
                        chunk_text
                        and evidence
                        and "Mencionado en" not in evidence
                        and len(evidence) > 15
                    ):
                        # Normalizar texto: minúsculas y limpiar caracteres de control
                        chunk_normalized = re.sub(r"[\t\r\n]+", " ", chunk_text.lower())
                        chunk_normalized = re.sub(r"\s+", " ", chunk_normalized)

                        evidence_normalized = re.sub(r"[\t\r\n]+", " ", evidence.lower())
                        evidence_normalized = re.sub(r"\s+", " ", evidence_normalized)

                        # Extraer palabras significativas de la evidencia
                        evidence_words = [
                            w
                            for w in re.findall(r"\b\w+\b", evidence_normalized)
                            if len(w) > 3 and not w.isdigit()
                        ]

                        # Verificar que al menos el 70% de palabras estén en el chunk
                        if evidence_words:
                            words_in_chunk = sum(1 for w in evidence_words if w in chunk_normalized)
                            match_ratio = words_in_chunk / len(evidence_words)

                            if match_ratio < 0.7:
                                errors.append(
                                    {
                                        "type": "EvidenceNotInChunk",
                                        "chunk_id": chunk_id,
                                        "message": f"La evidencia del tópico '{topic}': '{evidence[:80]}...' tiene solo {match_ratio:.0%} de palabras en el chunk",
                                    }
                                )
                                continue

                    topics.append(TopicMention(topic=topic, evidence=evidence, chunk_id=chunk_id))

        except json.JSONDecodeError as e:
            errors.append({"type": "JSONDecodeError", "chunk_id": chunk_id, "message": str(e)})

        return LLMExtractionResult(researchers=[], topics=topics, errors=errors)

    def extract_researchers_from_chunks(
        self, chunks: List[Dict[str, Any]], max_chunks: Optional[int] = None
    ) -> LLMExtractionResult:
        """Extraer investigadores desde lista de chunks."""
        all_researchers = []
        all_errors = []

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for i, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            chunk_text = chunk.get("text", "")

            if not chunk_text.strip():
                continue

            logger.debug(f"  Procesando chunk {i}/{len(chunks_to_process)}: {chunk_id}")
            result = self.extract_researchers_from_chunk(chunk_text, chunk_id)
            all_researchers.extend(result.researchers)
            all_errors.extend(result.errors)
            if result.researchers:
                logger.debug(
                    f"    → Encontrados: {', '.join([r.name for r in result.researchers])}"
                )

        return LLMExtractionResult(researchers=all_researchers, topics=[], errors=all_errors)

    def extract_topics_from_chunks(
        self, chunks: List[Dict[str, Any]], max_chunks: Optional[int] = None
    ) -> LLMExtractionResult:
        """Extraer tópicos desde lista de chunks."""
        all_topics = []
        all_errors = []

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for i, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            chunk_text = chunk.get("text", "")

            if not chunk_text.strip():
                continue

            logger.debug(f"  Procesando chunk {i}/{len(chunks_to_process)}: {chunk_id}")
            result = self.extract_topics_from_chunk(chunk_text, chunk_id)
            all_topics.extend(result.topics)
            all_errors.extend(result.errors)
            if result.topics:
                logger.debug(f"    → Encontrados: {', '.join([t.topic for t in result.topics])}")

        return LLMExtractionResult(researchers=[], topics=all_topics, errors=all_errors)

    def create_topics_from_llm_extraction(
        self,
        llm_result: LLMExtractionResult,
        existing_topic_ids: Optional[set[str]] = None,
    ) -> tuple[List[Entity], List[Relationship]]:
        """Crear entidades y relaciones de tópicos con deduplicación global.

        Solo crea entidades Topico y relaciones EXTRAIDO_DE desde chunks.
        Las relaciones TIENE_TOPICO proyecto->topico se crean después por agregación.
        """
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        existing_ids = existing_topic_ids or set()
        topics_by_name: dict[str, str] = {}

        for mention in llm_result.topics:
            topic_normalized = mention.topic.lower().strip()
            topic_id = topic_normalized.replace(" ", "_").replace(",", "").replace("/", "_")

            # El tópico ya existe globalmente
            if topic_id in existing_ids:
                relationships.append(
                    EXTRAIDO_DE(
                        mention.chunk_id, topic_id, properties={"evidence_text": mention.evidence}
                    )
                )
                continue

            # El tópico ya fue creado en esta misma llamada
            if topic_normalized in topics_by_name:
                existing_topic_id = topics_by_name[topic_normalized]
                relationships.append(
                    EXTRAIDO_DE(
                        mention.chunk_id,
                        existing_topic_id,
                        properties={"evidence_text": mention.evidence},
                    )
                )
                continue
            entities.append(Topico(id=topic_id, value=mention.topic))
            field_name = self.subfields_map.get(topic_normalized)
            if field_name:
                field_normalized = field_name.lower().strip()
                field_id = field_normalized.replace(" ", "_").replace(",", "").replace("/", "_")
                entities.append(Dominio(field_id, field_name))
                relationships.append(PERTENECE_A_DOMINIO(topic_id, field_id))
            relationships.append(
                EXTRAIDO_DE(
                    mention.chunk_id, topic_id, properties={"evidence_text": mention.evidence}
                )
            )
            relationships.append(
                EXTRAIDO_DE(
                    mention.chunk_id, topic_id, properties={"evidence_text": mention.evidence}
                )
            )

            topics_by_name[topic_normalized] = topic_id
            existing_ids.add(topic_id)

        return entities, relationships


def create_entities_and_relationships_from_llm_extraction(
    llm_result: LLMExtractionResult,
    project_id: str,
    existing_researcher_ids: Optional[set[str]] = None,
) -> tuple[List[Entity], List[Relationship]]:
    """Crear entidades y relaciones con deduplicación."""
    entities: list[Entity] = []
    relationships: list[Relationship] = []
    existing_ids = existing_researcher_ids or set()
    researchers_by_name: dict[str, str] = {}

    for mention in llm_result.researchers:
        name_normalized = mention.name.lower().strip()
        researcher_id = name_normalized.replace(" ", "_").replace(".", "").replace(",", "")

        if researcher_id in existing_ids:
            relationships.append(EXTRAIDO_DE(mention.chunk_id, researcher_id))
            continue

        if name_normalized in researchers_by_name:
            relationships.append(
                EXTRAIDO_DE(mention.chunk_id, researchers_by_name[name_normalized])
            )
            continue

        entities.append(
            Investigador(id=researcher_id, value={"name": mention.name, "source": "llm"})
        )
        relationships.append(PARTICIPO_EN(researcher_id, project_id))
        relationships.append(
            EXTRAIDO_DE(
                mention.chunk_id, researcher_id, properties={"evidence_text": mention.evidence}
            )
        )

        researchers_by_name[name_normalized] = researcher_id
        existing_ids.add(researcher_id)

    return entities, relationships
