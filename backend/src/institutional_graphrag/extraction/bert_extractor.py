"""Extracción de tópicos OpenAlex usando el modelo BERT fine-tuneado."""

from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from institutional_graphrag.extraction.parse_response import LLMExtractionResult, TopicMention
from institutional_graphrag.graph.schema import EXTRAIDO_DE, PERTENECE_A_DOMINIO, Dominio, Entity, Relationship, Topico

logger = logging.getLogger(__name__)

MODEL_NAME = "OpenAlex/bert-base-multilingual-cased-finetuned-openalex-topic-classification-title-abstract"
TOPICS_PATH = Path(__file__).parents[4] / "data" / "openalex_topics.json"
TOPICS_ES_PATH = Path(__file__).parents[4] / "data" / "openalex_topics_es.json"

DEFAULT_THRESHOLD = 0.04
DEFAULT_TOP_K = 5


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
        self._model = None
        self._tokenizer = None
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
            for subfield_entry in details.get("subfields", []):
                if isinstance(subfield_entry, dict):
                    subfield_name = subfield_entry.get("name", "")
                    for topic in subfield_entry.get("topics", []):
                        topic_name = topic.get("display_name", "")
                        self._topic_to_subfield[topic_name] = subfield_name
                        self._topic_to_field[topic_name] = field

    def _load_model(self) -> None:
        """Carga el modelo BERT y tokenizer (lazy, solo la primera vez que se usa)."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            logger.info(f"Cargando modelo BERT: {MODEL_NAME}")
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncate=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_NAME, output_hidden_states=False
            )
            self._model.eval()

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)

            self._id2label = self._model.config.id2label
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

    def extract_topics_from_chunk(self, chunk_text: str, chunk_id: str, heading: str = "") -> LLMExtractionResult:
        """Extrae tópicos de un chunk usando BERT."""
        if not chunk_text.strip():
            return LLMExtractionResult(researchers=[], topics=[], errors=[])

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
            return LLMExtractionResult(researchers=[], topics=topics, errors=[])

        except Exception as e:
            logger.error(f"Error BERT en chunk {chunk_id}: {e}")
            return LLMExtractionResult(
                researchers=[],
                topics=[],
                errors=[{"type": "BertExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def extract_topics_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: Optional[int] = None,
        include_headings: bool = True,
    ) -> LLMExtractionResult:
        """Clasifica tópicos sobre una lista de chunks, agrega scores por tópico."""
        # Scores acumulados por tópico (máximo entre chunks)
        topic_scores: Dict[str, float] = {}
        topic_chunk_ids: Dict[str, List[str]] = {}
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
                current = topic_scores.get(mention.topic, 0.0)
                if score > current:
                    topic_scores[mention.topic] = score
                    topic_chunk_ids[mention.topic] = [chunk_id]
                elif score == current:
                    topic_chunk_ids.setdefault(mention.topic, []).append(chunk_id)

        # Construir TopicMention con el chunk donde el score fue máximo
        final_topics = [
            TopicMention(
                topic=topic,
                evidence=f"score={score:.4f}",
                chunk_id=topic_chunk_ids[topic][0],
            )
            for topic, score in sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        ]

        return LLMExtractionResult(researchers=[], topics=final_topics, errors=all_errors)

    def create_topics_from_bert_extraction(
        self,
        bert_result: LLMExtractionResult,
        existing_topic_ids: Optional[set] = None,
    ) -> tuple[List[Entity], List[Relationship]]:
        """Crea entidades Topico y Dominio a partir del resultado BERT."""
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

            # Crear Dominio si el tópico pertenece a un subfield conocido
            subfield = self._topic_to_subfield.get(mention.topic)
            if subfield:
                domain_id = (
                    subfield.lower()
                    .replace(" ", "_")
                    .replace(",", "")
                    .replace("/", "_")
                )
                if domain_id not in seen_domains and domain_id not in existing_ids:
                    entities.append(Dominio(id=domain_id, value=subfield.lower()))
                    seen_domains.add(domain_id)
                relationships.append(PERTENECE_A_DOMINIO(actual_id, domain_id))

        return entities, relationships
