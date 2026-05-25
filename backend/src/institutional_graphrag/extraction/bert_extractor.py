"""Extracción de tópicos OpenAlex usando el modelo BERT fine-tuneado."""

from __future__ import annotations

import logging
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from institutional_graphrag.extraction.parse_topics import LLMExtractionResult, TopicMention
from institutional_graphrag.graph.schema import (
    EXTRAIDO_DE,
    PERTENECE_A_SUBCAMPO,
    Subcampo,
    Entity,
    Relationship,
    Topico,
)

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "BERT_TOPIC_MODEL",
    "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract",
)
TOPICS_PATH = Path(__file__).parents[4] / "data" / "openalex_topics.json"
TOPICS_ES_PATH = Path(__file__).parents[4] / "data" / "openalex_topics_es.json"

DEFAULT_THRESHOLD = 0.04
DEFAULT_TOP_K = 2


class BertTopicExtractor:
    """Clasifica tópicos OpenAlex usando BERT. Reemplaza al LLM para extracción de tópicos."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.top_k = top_k
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._id2label: Dict[int, str] = {}
        self._device = device
        self._topic_to_subfield: Dict[str, str] = {}
        self._topic_to_field: Dict[str, str] = {}
        self._load_topic_hierarchy()

    def _load_topic_hierarchy(self) -> None:
        """Carga la jerarquía tópico → subfield → field desde el JSON."""
        import json

        if not TOPICS_PATH.exists():
            return

        with open(TOPICS_PATH, encoding="utf-8") as f:
            data = json.load(f)

        for field, details in data.items():
            subfields = details.get("subfields", {})
            if not isinstance(subfields, dict):
                continue
            for subfield_name, topic_names in subfields.items():
                if not isinstance(topic_names, list):
                    continue
                for topic_name in topic_names:
                    if not topic_name:
                        continue
                    self._topic_to_subfield[topic_name] = subfield_name
                    self._topic_to_field[topic_name] = field

    @staticmethod
    def load_all_topics_and_subcampos() -> tuple[list, list]:
        """Carga todas las entidades Subcampo y Topico desde los tópicos de OpenAlex al inicio."""
        import json

        entities: list = []
        relationships: list = []

        if not TOPICS_ES_PATH.exists():
            logger.warning(f"Topics file not found: {TOPICS_ES_PATH}")
            return entities, relationships

        with open(TOPICS_ES_PATH, encoding="utf-8") as f:
            data = json.load(f)

        def _normalize_id(name: str) -> str:
            normalized = "".join(
                c
                for c in unicodedata.normalize("NFD", name.lower())
                if unicodedata.category(c) != "Mn" or c == "̃"
            )
            return (
                unicodedata.normalize("NFC", normalized)
                .replace(" ", "_")
                .replace(",", "")
                .replace("/", "_")
            )

        for _, field_data in data.items():
            subfields = field_data.get("subfields", {})
            if not isinstance(subfields, dict):
                continue
            for subfield_name, topic_names in subfields.items():
                if not subfield_name:
                    continue

                subcampo_id = _normalize_id(subfield_name)
                subcampo_value = unicodedata.normalize(
                    "NFC",
                    "".join(
                        c
                        for c in unicodedata.normalize("NFD", subfield_name.lower())
                        if unicodedata.category(c) != "Mn" or c == "̃"
                    ),
                )
                entities.append(Subcampo(id=subcampo_id, value=subcampo_value))

                if not isinstance(topic_names, list):
                    continue
                for topic_name in topic_names:
                    if not topic_name:
                        continue
                    topic_id = _normalize_id(topic_name)
                    topic_value = unicodedata.normalize(
                        "NFC",
                        "".join(
                            c
                            for c in unicodedata.normalize("NFD", topic_name.lower())
                            if unicodedata.category(c) != "Mn" or c == "̃"
                        ),
                    )
                    entities.append(Topico(id=topic_id, value=topic_value))
                    relationships.append(PERTENECE_A_SUBCAMPO(topic_id, subcampo_id))

        return entities, relationships

    def _load_model(self) -> None:
        """Carga el modelo BERT y tokenizer (lazy, solo la primera vez que se usa)."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Cargando modelo BERT: {MODEL_NAME}")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncate=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, output_hidden_states=False
            )
            model.eval()

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(self._device)

            self._tokenizer = tokenizer
            self._model = model
            self._id2label = model.config.id2label
            logger.info(f"Modelo cargado en {self._device}. Labels: {len(self._id2label)}")

        except ImportError as e:
            raise ImportError(
                "Se requiere 'transformers' y 'torch'. "
                "Instalá con: pip install transformers torch"
            ) from e

    def _format_input(self, heading: str, text: str) -> str:
        """Formatea el input al formato esperado por el modelo."""
        title = heading.strip() if heading else "NONE"
        abstract = text.strip()[:2500] if text else ""
        if abstract:
            return f"<TITLE> {title}\n<ABSTRACT> {abstract}"
        return f"<TITLE> {title}"

    def _predict_chunk(self, heading: str, text: str) -> List[Dict[str, Any]]:
        """Corre el modelo sobre un chunk y retorna lista de {topic, score}."""
        import torch

        self._load_model()
        assert self._tokenizer is not None and self._model is not None

        input_text = self._format_input(heading, text)
        inputs = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.sigmoid(outputs.logits)[0].cpu().tolist()

        results = [
            {"topic": self._id2label[i], "score": score}
            for i, score in enumerate(probs)
            if score >= self.threshold
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[: self.top_k]

    def extract_topics_from_chunk(
        self, chunk_text: str, chunk_id: str, heading: str = ""
    ) -> LLMExtractionResult:
        """Extrae tópicos de un chunk usando BERT."""
        if not chunk_text.strip():
            return LLMExtractionResult(topics=[], errors=[])

        try:
            predictions = self._predict_chunk(heading, chunk_text)
            topics = [
                TopicMention(
                    topic=pred["topic"],
                    evidence=f"score={pred['score']:.4f}",
                    chunk_id=chunk_id,
                )
                for pred in predictions
            ]
            return LLMExtractionResult(topics=topics, errors=[])

        except Exception as e:
            logger.error(f"Error BERT en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                topics=[],
                errors=[{"type": "BertExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def extract_topics_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: Optional[int] = None,
        include_headings: bool = True,
    ) -> LLMExtractionResult:
        """Clasifica tópicos sobre una lista de chunks, agrega por frecuencia entre chunks."""
        topic_chunk_count: Dict[str, int] = {}
        topic_first_chunk: Dict[str, str] = {}
        topic_max_score: Dict[str, float] = {}
        all_errors = []

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for i, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            raw_text = chunk.get("text", "")
            headings = chunk.get("metadata", {}).get("headings", [])
            heading = headings[-1].strip() if headings and include_headings else ""

            if not raw_text.strip():
                continue

            logger.debug(f"  BERT chunk {i}/{len(chunks_to_process)}: {chunk_id}")
            result = self.extract_topics_from_chunk(raw_text, chunk_id, heading)
            all_errors.extend(result.errors)

            for mention in result.topics:
                score = float(mention.evidence.replace("score=", ""))
                topic_chunk_count[mention.topic] = topic_chunk_count.get(mention.topic, 0) + 1
                if mention.topic not in topic_first_chunk:
                    topic_first_chunk[mention.topic] = chunk_id
                if score > topic_max_score.get(mention.topic, 0.0):
                    topic_max_score[mention.topic] = score

        # Ordenar por frecuencia (chunks en que apareció), desempatar por score máximo
        final_topics = [
            TopicMention(
                topic=topic,
                evidence=f"count={count} max_score={topic_max_score[topic]:.4f}",
                chunk_id=topic_first_chunk[topic],
            )
            for topic, count in sorted(
                topic_chunk_count.items(), key=lambda x: (x[1], topic_max_score[x[0]]), reverse=True
            )
        ]

        return LLMExtractionResult(topics=final_topics, errors=all_errors)

    def create_topics_from_bert_extraction(
        self,
        bert_result: LLMExtractionResult,
        existing_topic_ids: Optional[set] = None,
    ) -> tuple[List[Entity], List[Relationship]]:
        """Crea entidades Topico y Subcampo a partir del resultado BERT."""
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        existing_ids = existing_topic_ids or set()
        seen_topics: Dict[str, str] = {}
        seen_domains: set = set()

        for mention in bert_result.topics:
            topic_normalized = mention.topic.lower().strip()
            topic_id = (
                unicodedata.normalize(
                    "NFC",
                    "".join(
                        c
                        for c in unicodedata.normalize("NFD", topic_normalized)
                        if unicodedata.category(c) != "Mn" or c == "̃"
                    ),
                )
                .replace(" ", "_")
                .replace(",", "")
                .replace("/", "_")
            )

            if topic_id not in existing_ids and topic_id not in seen_topics:
                entities.append(Topico(id=topic_id, value=topic_normalized))
                seen_topics[topic_normalized] = topic_id

            actual_id = seen_topics.get(topic_normalized, topic_id)
            relationships.append(
                EXTRAIDO_DE(
                    mention.chunk_id, actual_id, properties={"evidence_text": mention.evidence}
                )
            )

            subfield = self._topic_to_subfield.get(mention.topic)
            if subfield:
                subcampo_id = subfield.lower().replace(" ", "_").replace(",", "").replace("/", "_")
                if subcampo_id not in seen_domains and subcampo_id not in existing_ids:
                    entities.append(Subcampo(id=subcampo_id, value=subfield.lower()))
                    seen_domains.add(subcampo_id)
                relationships.append(PERTENECE_A_SUBCAMPO(actual_id, subcampo_id))

        return entities, relationships
