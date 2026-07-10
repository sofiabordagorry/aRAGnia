"""Extracción de tópicos OpenAlex usando el modelo BERT fine-tuneado."""

from __future__ import annotations

import logging
import os
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from institutional_graphrag.graph.schema import (
    EXTRAIDO_DE,
    PERTENECE_A_SUBCAMPO,
    TIENE_TOPICO,
    Relationship,
    Subcampo,
    Topico,
)

logger = logging.getLogger(__name__)

MODEL_NAME = os.getenv(
    "BERT_TOPIC_MODEL",
    "albertmartinez/openalex-topic-classification-title-abstract",
)
TOPICS_PATH = Path(__file__).parents[4] / "data" / "openalex_topics.json"
TOPICS_ES_PATH = Path(__file__).parents[4] / "data" / "openalex_topics_es.json"

DEFAULT_THRESHOLD = 0.08
DEFAULT_CONFIDENCE_LOGIT_THRESHOLD = 13.5
DEFAULT_COVERAGE_LOGIT_THRESHOLD = 0.02
DEFAULT_MIN_TOPICS = 1


@dataclass
class TopicMention:
    """Mención de tópico en un chunk."""

    topic: str
    evidence: str
    logit: float
    chunk_id: str


@dataclass
class BertExtractionResult:
    """Resultado de extracción LLM."""

    topics: List[TopicMention]
    errors: List[Dict[str, Any]]


class BertTopicExtractor:
    """Clasifica tópicos OpenAlex usando BERT. Reemplaza al LLM para extracción de tópicos."""

    def __init__(
        self,
        threshold: float | None,
        confidence_logit_threshold: float | None,
        coverage_logit_threshold: float | None,
        device: Optional[str] = None,
        min_topics: int | None = None,
    ):
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        self.confidence_logit_threshold = (
            confidence_logit_threshold
            if confidence_logit_threshold is not None
            else DEFAULT_CONFIDENCE_LOGIT_THRESHOLD
        )
        self.coverage_logit_threshold = (
            coverage_logit_threshold
            if coverage_logit_threshold is not None
            else DEFAULT_COVERAGE_LOGIT_THRESHOLD
        )
        self.min_topics = min_topics if min_topics is not None else DEFAULT_MIN_TOPICS
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._id2label: Dict[int, str] = {}
        self._device = device
        self._topic_to_subfield: Dict[str, str] = {}
        self._topic_to_field: Dict[str, str] = {}
        self._en_to_es_topic: Dict[str, str] = {}
        self._en_to_es_subfield: Dict[str, str] = {}
        self._load_topic_hierarchy()

    def _load_topic_hierarchy(self) -> None:
        """Carga la jerarquía tópico → subfield → field y los mapeos EN→ES."""
        import json

        if not TOPICS_PATH.exists():
            return

        with open(TOPICS_PATH, encoding="utf-8") as f:
            en_data = json.load(f)

        for field, details in en_data.items():
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

        if not TOPICS_ES_PATH.exists():
            return

        with open(TOPICS_ES_PATH, encoding="utf-8") as f:
            es_data = json.load(f)

        for (_, en_details), (_, es_details) in zip(en_data.items(), es_data.items()):
            en_subfields = en_details.get("subfields", {})
            es_subfields = es_details.get("subfields", {})
            for (en_sub, en_topics), (es_sub, es_topics) in zip(
                en_subfields.items(), es_subfields.items()
            ):
                self._en_to_es_subfield[en_sub] = es_sub
                if not isinstance(en_topics, list) or not isinstance(es_topics, list):
                    continue
                for en_topic, es_topic in zip(en_topics, es_topics):
                    if en_topic and es_topic:
                        self._en_to_es_topic[en_topic] = es_topic

    @staticmethod
    def _strip_label_prefix(label: str) -> str:
        """Los modelos BERT devuelven labels con prefijo de topic ID ('T10001: ' o 'NNNN: ') que no está en el JSON."""
        if ": " in label:
            prefix, name = label.split(": ", 1)
            if prefix.isdigit() or (prefix.startswith("T") and prefix[1:].isdigit()):
                return name
        return label

    @staticmethod
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

        for _, field_data in data.items():
            subfields = field_data.get("subfields", {})
            if not isinstance(subfields, dict):
                continue
            for subfield_name, topic_names in subfields.items():
                if not subfield_name:
                    continue

                subcampo_id = BertTopicExtractor._normalize_id(subfield_name)
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
                    topic_id = BertTopicExtractor._normalize_id(topic_name)
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

    def _format_input(self, title: str, text: str) -> str:
        """Formatea el input al formato esperado por el modelo."""
        safe_title = title.strip() if title else "NONE"
        abstract = text.strip()[:2500] if text else ""
        if abstract:
            return f"<TITLE> {safe_title}\n<ABSTRACT> {abstract}"
        return f"<TITLE> {safe_title}"

    def _predict_chunk(self, title: str, text: str) -> List[Dict[str, Any]]:
        """Corre el modelo sobre un chunk y retorna lista de {topic, score, logit}."""
        import torch

        self._load_model()
        assert self._tokenizer is not None and self._model is not None
        input_text = self._format_input(title, text)

        tokens = self._tokenizer(input_text, truncation=False)
        n_tokens = len(tokens["input_ids"])
        if n_tokens > 512:
            logger.warning(f"Chunk excede 512 tokens ({n_tokens}), se truncará")

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
            logits = outputs.logits[0].cpu().tolist()
            # El modelo es single_label_classification: softmax da una probabilidad
            # calibrada (suma 1 entre los topics) y hace interpretable el threshold.
            probs = torch.softmax(outputs.logits, dim=-1)[0].cpu().tolist()

        results = [
            {
                "topic": self._strip_label_prefix(self._id2label[i]),
                "score": score,
                "logit": logits[i],
            }
            for i, score in enumerate(probs)
            if score >= self.threshold
        ]
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def extract_topics_from_chunk(
        self, chunk_text: str, chunk_id: str, project_title: str = ""
    ) -> BertExtractionResult:
        """Extrae tópicos de un chunk usando BERT."""
        if not chunk_text.strip():
            return BertExtractionResult(topics=[], errors=[])

        try:
            predictions = self._predict_chunk(project_title, chunk_text)
            topics = [
                TopicMention(
                    topic=pred["topic"],
                    evidence=f"score={pred['score']:.4f}",
                    logit=pred["logit"],
                    chunk_id=chunk_id,
                )
                for pred in predictions
            ]
            return BertExtractionResult(topics=topics, errors=[])

        except Exception as e:
            logger.error(f"Error BERT en chunk {chunk_id}: {e}")
            return BertExtractionResult(
                topics=[],
                errors=[{"type": "BertExtractionError", "chunk_id": chunk_id, "message": str(e)}],
            )

    def extract_topics_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        max_chunks: Optional[int] = None,
        project_title: str = "",
    ) -> BertExtractionResult:
        """Clasifica tópicos sobre una lista de chunks, agrega por frecuencia entre chunks."""
        topic_chunk_count: Dict[str, int] = {}
        topic_first_chunk: Dict[str, str] = {}
        topic_max_score: Dict[str, float] = {}
        topic_sum_logit: Dict[str, float] = {}

        all_errors = []

        chunks_to_process = chunks[:max_chunks] if max_chunks else chunks

        for i, chunk in enumerate(chunks_to_process, 1):
            chunk_id = chunk.get("chunk_id", f"chunk_{i}")
            raw_text = chunk.get("text", "")

            if not raw_text.strip():
                continue

            logger.debug(f"  BERT chunk {i}/{len(chunks_to_process)}: {chunk_id}")
            result = self.extract_topics_from_chunk(raw_text, chunk_id, project_title)
            all_errors.extend(result.errors)

            for mention in result.topics:
                score = float(mention.evidence.replace("score=", ""))
                topic_chunk_count[mention.topic] = topic_chunk_count.get(mention.topic, 0) + 1
                topic_sum_logit[mention.topic] = (
                    topic_sum_logit.get(mention.topic, 0.0) + mention.logit
                )
                if mention.topic not in topic_first_chunk:
                    topic_first_chunk[mention.topic] = chunk_id
                if score > topic_max_score.get(mention.topic, 0.0):
                    topic_max_score[mention.topic] = score

        # Ordenar por frecuencia (chunks en que apareció), desempatar por score máximo
        final_topics = [
            TopicMention(
                topic=topic,
                evidence=f"count={count} max_score={topic_max_score[topic]:.4f}",
                logit=topic_sum_logit[topic],
                chunk_id=topic_first_chunk[topic],
            )
            for topic, count in sorted(
                topic_chunk_count.items(), key=lambda x: (x[1], topic_max_score[x[0]]), reverse=True
            )
        ]

        return BertExtractionResult(topics=final_topics, errors=all_errors)

    def aggregate_topics_for_project(
        self,
        project_id: str,
        project_bert_results: list,
        total_chunks: int,
    ) -> List[Relationship]:
        """Suma logits por tópico a nivel proyecto y crea TIENE_TOPICO para los que superan el
        threshold; si quedan menos de min_topics, completa con los de mayor cobertura."""
        relationships: List[Relationship] = []
        if not project_bert_results:
            return relationships

        topic_logit_sum: Dict[str, float] = defaultdict(float)
        topic_count_sum: Dict[str, int] = defaultdict(int)
        topic_chunks_info: Dict[str, List[tuple[str, str]]] = defaultdict(list)
        for mention in project_bert_results:
            topic_logit_sum[mention.topic] += mention.logit
            count = int(mention.evidence.split("count=")[1].split(" ")[0])
            topic_count_sum[mention.topic] += count
            topic_chunks_info[mention.topic].append((mention.chunk_id, mention.evidence))

        def coverage(topic_en: str) -> float:
            return topic_count_sum[topic_en] / total_chunks if total_chunks else 0.0

        def confidence(topic_en: str) -> float:
            return (
                topic_logit_sum[topic_en] / topic_count_sum[topic_en]
                if topic_count_sum[topic_en]
                else 0.0
            )

        passing = [
            topic_en
            for topic_en in topic_logit_sum
            if coverage(topic_en) >= self.coverage_logit_threshold
            and confidence(topic_en) >= self.confidence_logit_threshold
        ]
        selected = list(passing)
        if len(selected) < self.min_topics:
            fallback = sorted(
                (t for t in topic_logit_sum if t not in passing),
                key=lambda t: (coverage(t), confidence(t)),
                reverse=True,
            )
            selected.extend(fallback[: self.min_topics - len(selected)])

        for topic_en in selected:
            es_topic = self._en_to_es_topic.get(topic_en, topic_en)
            topic_id = self._normalize_id(es_topic)
            relationships.append(
                TIENE_TOPICO(
                    project_id,
                    topic_id,
                    properties={
                        "coverage_logit": float(coverage(topic_en)),
                        "confidence_logit": float(confidence(topic_en)),
                        "mention_count": int(topic_count_sum[topic_en]),
                        "fallback": topic_en not in passing,
                    },
                )
            )
            for chunk_id, evidence in topic_chunks_info[topic_en]:
                relationships.append(
                    EXTRAIDO_DE(chunk_id, topic_id, properties={"texto_evidencia": evidence})
                )
        return relationships
