"""Extracción de investigadores usando LLMs."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from institutional_graphrag.graph.schema import (
    EVIDENCIA_DE,
    PARTICIPO_EN,
    Entity,
    Investigador,
    Relationship,
)
from institutional_graphrag.rag.generate import get_llm_client

logger = logging.getLogger(__name__)


@dataclass
class ResearcherMention:
    """Mención de investigador en un chunk."""
    name: str
    evidence: str
    chunk_id: str


@dataclass
class LLMExtractionResult:
    """Resultado de extracción LLM."""
    researchers: List[ResearcherMention]
    errors: List[Dict[str, Any]]


class LLMEntityExtractor:
    """Extractor de investigadores usando LLMs."""

    def __init__(
        self,
        *,
        llm_provider: str = "ollama",
        llm_model: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_client = get_llm_client(provider=llm_provider, model=llm_model)

    def extract_researchers_from_chunk(
        self, chunk_text: str, chunk_id: str
    ) -> LLMExtractionResult:
        """Extraer investigadores de un chunk."""
        messages = [
            {"role": "system", "content": "Eres un asistente especializado en análisis de documentos académicos."},
            {"role": "user", "content": self._build_prompt(chunk_text)},
        ]

        try:
            response = self.llm_client.generate(
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return self._parse_response(response, chunk_id)

        except Exception as e:
            logger.error(f"Error en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                researchers=[],
                errors=[{"type": "LLMExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def _build_prompt(self, chunk_text: str) -> str:
        """Construir prompt para extracción."""
        return f"""Analiza este texto e identifica investigadores mencionados.

TEXTO:
{chunk_text}

Responde SOLO con JSON válido:
{{
  "researchers": [
    {{"name": "Nombre Completo", "evidence": "fragmento donde se menciona"}}
  ]
}}

Si no hay investigadores: {{"researchers": []}}"""

    def _parse_response(self, response: str, chunk_id: str) -> LLMExtractionResult:
        """Parsear respuesta del LLM."""
        errors = []
        researchers = []

        json_start = response.find("{")
        json_end = response.rfind("}") + 1

        if json_start == -1 or json_end == 0:
            errors.append({"type": "InvalidJSON", "chunk_id": chunk_id, "message": "No hay JSON en respuesta"})
            return LLMExtractionResult(researchers=[], errors=errors)

        try:
            data = json.loads(response[json_start:json_end])
            researchers_data = data.get("researchers", [])

            if not isinstance(researchers_data, list):
                errors.append({"type": "InvalidJSON", "chunk_id": chunk_id, "message": "'researchers' no es lista"})
                return LLMExtractionResult(researchers=[], errors=errors)

            for item in researchers_data:
                if not isinstance(item, dict):
                    continue

                name = item.get("name", "").strip()
                evidence = item.get("evidence", "").strip() or f"Mencionado en {chunk_id}"

                if name:
                    researchers.append(ResearcherMention(name=name, evidence=evidence, chunk_id=chunk_id))

        except json.JSONDecodeError as e:
            errors.append({"type": "JSONDecodeError", "chunk_id": chunk_id, "message": str(e)})

        return LLMExtractionResult(researchers=researchers, errors=errors)

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

            logger.info(f"Procesando chunk {i}/{len(chunks_to_process)}: {chunk_id}")

            result = self.extract_researchers_from_chunk(chunk_text, chunk_id)
            all_researchers.extend(result.researchers)
            all_errors.extend(result.errors)

        return LLMExtractionResult(researchers=all_researchers, errors=all_errors)


def create_entities_and_relationships_from_llm_extraction(
    llm_result: LLMExtractionResult,
    project_id: str,
    existing_researcher_ids: Optional[set[str]] = None,
) -> tuple[List[Entity], List[Relationship]]:
    """Crear entidades y relaciones con deduplicación."""
    entities = []
    relationships = []
    existing_ids = existing_researcher_ids or set()
    researchers_by_name = {}

    for mention in llm_result.researchers:
        name_normalized = mention.name.lower().strip()
        researcher_id = f"inv_{name_normalized.replace(' ', '_')}"

        if researcher_id in existing_ids:
            relationships.append(EVIDENCIA_DE(mention.chunk_id, researcher_id))
            continue

        if name_normalized in researchers_by_name:
            relationships.append(EVIDENCIA_DE(mention.chunk_id, researchers_by_name[name_normalized]))
            continue

        entities.append(Investigador(id=researcher_id, value=mention.name))
        relationships.append(PARTICIPO_EN(researcher_id, project_id))
        relationships.append(
            EVIDENCIA_DE(mention.chunk_id, researcher_id, properties={"evidence_text": mention.evidence})
        )

        researchers_by_name[name_normalized] = researcher_id
        existing_ids.add(researcher_id)

    return entities, relationships
