"""Parseo y validación de respuestas LLM para extracción de tópicos."""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class TopicMention:
    """Mención de tópico en un chunk."""

    topic: str
    evidence: str
    chunk_id: str


@dataclass
class LLMExtractionResult:
    """Resultado de extracción LLM."""

    topics: List[TopicMention]
    errors: List[Dict[str, Any]]


def _extract_json(response: str, chunk_id: str) -> tuple[Any, list]:
    errors: list = []
    json_match = re.search(r"<JSON>\s*(\{.*?\})\s*</JSON>", response, re.DOTALL)
    data = None
    if json_match:
        json_str = json_match.group(1)
        try:
            return json.loads(json_str), errors
        except json.JSONDecodeError as e:
            errors.append(
                {
                    "type": "JSONDecodeError",
                    "chunk_id": chunk_id,
                    "message": f"JSON dentro de tags inválido: {str(e)}",
                }
            )
            return data, errors

    json_start = response.find("{")
    if json_start == -1:
        errors.append(
            {
                "type": "InvalidJSON",
                "chunk_id": chunk_id,
                "message": "No hay JSON en respuesta (falta <JSON> tags)",
            }
        )
        return data, errors

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
    return data, errors


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\t\r\n]+", " ", text.lower()))


def _evidence_not_in_chunk_validation(
    evidence: str, chunk_text: str, lenght: int
) -> Optional[float]:
    if chunk_text and evidence and "Mencionado en" not in evidence and len(evidence) > lenght:
        chunk_normalized = _normalize_text(chunk_text)
        evidence_normalized = _normalize_text(evidence)
        evidence_words = [
            w for w in re.findall(r"\b\w+\b", evidence_normalized) if len(w) > 3 and not w.isdigit()
        ]
        if len(evidence_words) >= 3:
            words_in_chunk = sum(1 for w in evidence_words if w in chunk_normalized)
            match_ratio = words_in_chunk / len(evidence_words)
            if match_ratio < 0.7:
                return match_ratio
    return None


def _list_topic_validation(topic: str, available_topics: list[str]) -> bool:
    topic_lower = topic.lower()
    available_lower = [t.lower() for t in available_topics]
    return topic_lower not in available_lower


def _valid_json_structure(data: Any, info: str, chunk_id: str) -> tuple[list, list]:
    info_data = data.get(info, [])
    errors: list = []
    if not isinstance(info_data, list):
        errors.append(
            {"type": "InvalidJSON", "chunk_id": chunk_id, "message": f"'{info}' no es lista"}
        )
    return info_data, errors


def _validate_json_sub_structure(item: Any, value: str, chunk_id: str) -> tuple[str, str, list]:
    errors: list = []
    raw_value = ""
    evidence = ""
    if not isinstance(item, dict):
        errors.append(
            {
                "type": "InvalidJSON",
                "chunk_id": chunk_id,
                "message": "Elemento de 'topics' no es un objeto",
            }
        )
        return raw_value, evidence, errors

    raw = item.get(value, "")
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    raw_value = str(raw).strip() if raw else ""

    raw_evidence = item.get("evidence", "")
    if isinstance(raw_evidence, list):
        raw_evidence = raw_evidence[0] if raw_evidence else ""
    evidence = str(raw_evidence).strip() if raw_evidence else ""

    if not evidence:
        errors.append(
            {
                "type": "MissingEvidence",
                "chunk_id": chunk_id,
                "message": f"'{value}'='{raw_value}' sin evidence — descartado",
            }
        )

    return raw_value, evidence, errors


def parse_topic_response(
    response: str, available_topics: list[str], chunk_id: str, chunk_text: str = ""
) -> LLMExtractionResult:
    """Parsea la respuesta del LLM para tópicos."""
    errors: list = []
    topics: list = []

    data, errors = _extract_json(response, chunk_id)
    if errors:
        return LLMExtractionResult(topics=[], errors=errors)

    try:
        topics_data, errors = _valid_json_structure(data, "topics", chunk_id)
        if errors:
            return LLMExtractionResult(topics=[], errors=errors)

        for item in topics_data:
            topic, evidence, errors_aux = _validate_json_sub_structure(item, "topic", chunk_id)
            if errors_aux:
                errors.extend(errors_aux)
                continue

            if topic is None:
                errors.append(
                    {
                        "type": "MissingTopic",
                        "chunk_id": chunk_id,
                        "message": "Falta el campo 'topic' o está vacío",
                    }
                )
                continue

            if _list_topic_validation(topic, available_topics):
                errors.append(
                    {
                        "type": "InvalidTopic",
                        "chunk_id": chunk_id,
                        "message": (
                            f"Tópico '{topic}' no está en la lista permitida (inventado por LLM)"
                        ),
                    }
                )
                continue

            match_ratio = _evidence_not_in_chunk_validation(evidence, chunk_text, 15)
            if match_ratio is not None:
                errors.append(
                    {
                        "type": "EvidenceNotInChunk",
                        "chunk_id": chunk_id,
                        "message": (
                            f"La evidencia '{evidence[:80]}...' no está en el chunk "
                            f"(solo {match_ratio:.0%} de palabras coinciden)"
                        ),
                    }
                )
                continue
            topics.append(TopicMention(topic=topic, evidence=evidence, chunk_id=chunk_id))

    except json.JSONDecodeError as e:
        errors.append({"type": "JSONDecodeError", "chunk_id": chunk_id, "message": str(e)})

    return LLMExtractionResult(topics=topics, errors=errors)
