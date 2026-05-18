"""Extracción de tópicos usando LLMs."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from institutional_graphrag.extraction.parse_topics import (
    LLMExtractionResult,
    TopicMention,
    parse_topic_response,
)
from institutional_graphrag.graph.schema import (
    EXTRAIDO_DE,
    PERTENECE_A_DOMINIO,
    Dominio,
    Entity,
    Relationship,
    Topico,
)
from institutional_graphrag.llm.llm_provider import get_llm_client

logger = logging.getLogger(__name__)
TOPICS_PATH = Path(__file__).parents[4] / "data" / "openalex_topics_es.json"

__all__ = [
    "LLMEntityExtractor",
    "LLMExtractionResult",
    "TopicMention",
]


class LLMEntityExtractor:
    """Extractor de tópicos usando LLMs."""

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
        """Crea un mapeo de fields y subfields para búsqueda rápida."""
        with open(file_path, "r", encoding="utf-8") as f:
            fields_dict = json.load(f)
        subfield_to_field = {}
        subfields_list = []
        for field, details in fields_dict.items():
            for subfield in details.get("subfields", []):
                subfield_normalized = subfield.lower().strip()
                subfield_to_field[subfield_normalized] = field
                subfields_list.append(subfield)
        return subfields_list, subfield_to_field

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
            return parse_topic_response(response, self.available_topics, chunk_id, chunk_text)

        except Exception as e:
            logger.error(f"Error en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                topics=[],
                errors=[{"type": "LLMExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

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

    def extract_topics_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: Optional[int] = None,
        include_headings: bool = True,
    ) -> LLMExtractionResult:
        """Extraer tópicos desde lista de chunks."""
        all_topics = []
        all_errors = []

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for i, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            raw_text = chunk.get("text", "")
            headings = chunk.get("metadata", {}).get("headings", [])

            if not raw_text.strip():
                continue

            clean_text = raw_text.strip()
            if include_headings and headings:
                last_heading = headings[-1].strip()
                if clean_text == last_heading:
                    final_text = "\n".join(headings)
                else:
                    final_text = "\n".join(headings) + "\n" + clean_text
            else:
                final_text = clean_text

            logger.debug(f"  Procesando chunk {i}/{len(chunks_to_process)}: {chunk_id}")
            result = self.extract_topics_from_chunk(final_text, chunk_id)
            all_topics.extend(result.topics)
            all_errors.extend(result.errors)
            if result.topics:
                logger.debug(f"    → Encontrados: {', '.join([t.topic for t in result.topics])}")

        return LLMExtractionResult(topics=all_topics, errors=all_errors)

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
            topic_value = "".join(
                c
                for c in unicodedata.normalize("NFD", mention.topic.lower())
                if unicodedata.category(c) != "Mn" or c == "\u0303"
            )
            topic_value = unicodedata.normalize("NFC", topic_value)

            entities.append(Topico(id=topic_id, value=topic_value))
            field_name = self.subfields_map.get(topic_normalized)
            if field_name:
                field_normalized = field_name.lower().strip()
                field_id = field_normalized.replace(" ", "_").replace(",", "").replace("/", "_")
                domain_value = "".join(
                    c
                    for c in unicodedata.normalize("NFD", field_name.lower())
                    if unicodedata.category(c) != "Mn" or c == "\u0303"
                )
                domain_value = unicodedata.normalize("NFC", domain_value)

                entities.append(Dominio(field_id, domain_value))
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
