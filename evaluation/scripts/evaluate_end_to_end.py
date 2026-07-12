#!/usr/bin/env python3
"""Pipeline end-to-end local para Institutional GraphRAG con evaluación QA.

Flujo predeterminado:
    data/corpus
        -> Docling
        -> chunking
        -> extracción de entidades y relaciones
        -> limpieza y carga en Neo4j
        -> ejecución de preguntas mediante GraphRAGRetriever.run_query/query
        -> evaluación con LLM-as-a-judge
        -> JSON de detalle/resumen, gráficas PNG y reporte HTML

No compara modelos ni variantes de prompts. La generación de respuestas usa el
modelo y el prompt configurados por defecto dentro de GraphRAGRetriever.

El dataset QA es obligatorio:

    python evaluation/scripts/evaluate_end_to_end.py \
        evaluation/ground_truth/datasetQA_GT_evaluation.json
Juez Anthropic por defecto:

    python evaluation/scripts/evaluate_end_to_end.py \
        evaluation/ground_truth/datasetQA_GT_evaluation.json

Usar otro modelo de Anthropic como juez:

    python evaluation/scripts/evaluate_end_to_end.py \
        evaluation/ground_truth/datasetQA_GT_evaluation.json \
        --judge-model NOMBRE_DEL_MODELO

Ejecutar sin evaluación LLM-as-a-judge:

    python evaluation/scripts/evaluate_end_to_end.py \
        evaluation/ground_truth/datasetQA_GT_evaluation.json \
        --no-judge

Usar Docling existente, pero regenerar chunks, extracción y Neo4j:

    python evaluation/scripts/evaluate_end_to_end.py \
        evaluation/ground_truth/datasetQA_GT_evaluation.json \
        --skip-docling

Ejecutar únicamente preguntas, respuestas, evaluación y reportes sobre el grafo
que ya está cargado:

    python evaluation/scripts/evaluate_end_to_end.py \
        evaluation/ground_truth/datasetQA_GT_evaluation.json \
        --qa-only

Cuando se ejecutan Docling o chunking, sus carpetas de salida se limpian siempre
antes de comenzar. Cuando se carga Neo4j, el grafo se limpia siempre antes de la
ingesta.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import generate_gt_answers as gen  # noqa: E402

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv
from docling_core.types.doc import DoclingDocument

from institutional_graphrag.config import EMBED_MODEL_ID
from institutional_graphrag.extraction.ie import EntityExtractor, ExtractionResult
from institutional_graphrag.graph.builder import GraphBuilder
from institutional_graphrag.graph.graph_loader import load_graph_json
from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker
from institutional_graphrag.ingest.docling_parser import (
    DEFAULT_DOCLING_DIR,
    DocumentAlreadyProcessed,
    parse_corpus,
    parse_single_document,
)
from institutional_graphrag.ingest.table_extractors import convert_tables_to_chunks

try:
    from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever
except ImportError:
    from graph_retriever import GraphRAGRetriever  # type: ignore


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("end_to_end_local_corpus_qa_eval")


REPO_DIR = Path(__file__).resolve().parents[2]
BACKEND_DIR = REPO_DIR / "backend"
DATA_DIR = REPO_DIR / "data"
EVAL_DIR = REPO_DIR / "evaluation"

CORPUS_DIR = DATA_DIR / "corpus"
DOCLING_DIR = DEFAULT_DOCLING_DIR
CHUNKS_DIR = DATA_DIR / "chunks"
EXTRACTED_FILENAME = "entity_documents.json"
RESULTS_DIR = EVAL_DIR / "results" / "graph_query_eval"

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_JUDGE = "claude-opus-4-8"

JUDGE_DIMS = ["factual_correctness", "completeness", "faithfulness"]
DIM_LABELS = {
    "factual_correctness": "Correctitud",
    "completeness": "Completitud",
    "faithfulness": "Fidelidad",
}

JUDGE_SYSTEM = (
    "Eres un evaluador experto de respuestas de un sistema GraphRAG. Compara la "
    "RESPUESTA CANDIDATA contra la RESPUESTA DE REFERENCIA usando como fuente de "
    "verdad los RESULTADOS DEL GRAFO. Puntuá de 1 a 5 (1=muy malo, 5=excelente):\n"
    "- factual_correctness: ¿coincide en hechos con la referencia?\n"
    "- completeness: ¿incluye todos los elementos esperados sin omitir resultados?\n"
    "- faithfulness: ¿no inventa información fuera de los resultados del grafo?\n"
    "Respondé SOLO con un objeto JSON válido con las claves: "
    "factual_correctness, completeness, faithfulness (enteros 1-5) y justification "
    "(string breve). Sin texto adicional ni markdown."
)


# -----------------------------------------------------------------------------
# Corpus -> Docling -> chunks -> extracción -> Neo4j
# -----------------------------------------------------------------------------


def load_env() -> None:
    env_path = BACKEND_DIR / ".env"
    load_dotenv(env_path)
    log.info("Variables cargadas desde: %s", env_path)


def clear_directory(directory: Path) -> None:
    """Elimina todo el contenido de una carpeta y vuelve a crearla vacía."""
    if directory.exists():
        for path in directory.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    directory.mkdir(parents=True, exist_ok=True)


def save_docling_dict(doc_dict: dict[str, Any], output_dir: Path) -> Path:
    filename = doc_dict.get("name", "sin_nombre")
    output_path = output_dir / f"{filename}.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(doc_dict, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def run_docling() -> int:
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"No existe el corpus local: {CORPUS_DIR}")

    clear_directory(DOCLING_DIR)
    log.info("Se limpió la carpeta de salida de Docling: %s", DOCLING_DIR)

    saved = 0
    log.info("Docling: %s -> %s", CORPUS_DIR, DOCLING_DIR)

    if CORPUS_DIR.is_file():
        try:
            saved_path = save_docling_dict(parse_single_document(CORPUS_DIR), DOCLING_DIR)
            log.info("Docling OK: %s", saved_path.name)
            saved += 1
        except DocumentAlreadyProcessed as exc:
            log.warning("Docling omitido: %s", exc)
    else:
        for doc_dict in parse_corpus(CORPUS_DIR):
            saved_path = save_docling_dict(doc_dict, DOCLING_DIR)
            log.info("Docling OK: %s", saved_path.name)
            saved += 1

    log.info("Docling finalizado: %s documentos", saved)
    return saved


def run_chunking() -> tuple[int, int, int]:
    if not DOCLING_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta Docling: {DOCLING_DIR}")

    json_files = sorted(DOCLING_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No hay JSONs de Docling en {DOCLING_DIR}")

    clear_directory(CHUNKS_DIR)
    log.info("Se limpió la carpeta de chunks: %s", CHUNKS_DIR)

    tokenizer = EMBED_MODEL_ID
    shared_chunker = get_native_chunker(tokenizer=tokenizer)
    processed = 0
    errors = 0
    total_chunks = 0

    log.info("Chunking: %s archivos", len(json_files))
    for json_file in json_files:
        output_file = CHUNKS_DIR / f"{json_file.stem}_chunks.json"
        try:
            doc_dict = json.loads(json_file.read_text(encoding="utf-8"))
            doc = DoclingDocument.model_validate(doc_dict)
            chunks = chunk_document(doc=doc, chunker=shared_chunker)
            output_file.write_text(
                json.dumps(
                    {
                        "source": doc.name,
                        "total_chunks": len(chunks),
                        "tokenizer": tokenizer,
                        "chunks": chunks,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            processed += 1
            total_chunks += len(chunks)
            log.info("Chunks OK: %s (%s)", output_file.name, len(chunks))
        except Exception as exc:  # noqa: BLE001
            errors += 1
            log.exception("Error procesando %s: %s", json_file.name, exc)

    convert_tables_to_chunks()
    log.info(
        "Chunking finalizado: procesados=%s errores=%s chunks=%s",
        processed,
        errors,
        total_chunks,
    )
    return processed, errors, total_chunks


def normalize_result(result: ExtractionResult) -> dict[str, Any]:
    return {
        "entities": sorted(
            [entity.to_dict() for entity in result.entities],
            key=lambda value: (value["label"], value["id"]),
        ),
        "relationships": sorted(
            [relationship.to_dict() for relationship in result.relationships],
            key=lambda value: (
                value["type"],
                value["source_id"],
                value["target_id"],
            ),
        ),
        "errors": sorted(
            result.errors,
            key=lambda value: (value.get("type", ""), value.get("message", "")),
        ),
    }


def run_extraction() -> Path:
    extractor = EntityExtractor()
    result = extractor.run()
    extractor.save_in_file(EXTRACTED_FILENAME)

    output_path = extractor.input_dir / EXTRACTED_FILENAME
    loaded = extractor.load_from_json(EXTRACTED_FILENAME)
    if loaded is None:
        raise RuntimeError("No se pudo volver a cargar el JSON de extracción")

    same_counts = (
        len(loaded.entities) == len(result.entities)
        and len(loaded.relationships) == len(result.relationships)
    )
    same_content = normalize_result(result) == normalize_result(loaded)

    log.info(
        "Extracción guardada en %s | entidades=%s relaciones=%s errores=%s",
        output_path,
        len(result.entities),
        len(result.relationships),
        len(result.errors),
    )
    log.info(
        "Validación del JSON: conteos_ok=%s contenido_ok=%s",
        same_counts,
        same_content,
    )
    return output_path


def neo4j_config() -> tuple[str, str, str]:
    host = os.getenv("HOST", "localhost")
    port = os.getenv("NEO4J_BOLT_PORT", "7687")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")

    if not user or not password:
        raise RuntimeError("Faltan NEO4J_USER o NEO4J_PASSWORD en backend/.env")

    return f"bolt://{host}:{port}", user, password


def load_into_neo4j(json_path: Path) -> None:
    if not json_path.exists():
        raise FileNotFoundError(f"No existe el JSON del grafo: {json_path}")

    uri, user, password = neo4j_config()
    entities, relationships = load_graph_json(json_path, uri, user, password)

    graph = GraphBuilder(uri, user, password)
    try:
        log.warning("Se limpiará Neo4j antes de cargar el grafo")
        graph.clear_graph()
        graph.ingest(entities=entities, relationships=relationships)
    finally:
        graph.close()

    log.info(
        "Neo4j cargado: entidades=%s relaciones=%s",
        len(entities),
        len(relationships),
    )

def load_qa_items(dataset_path: Path) -> List[Dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe el dataset QA: {dataset_path}")

    return gen.load_items(dataset_path)

def sentinel_matches(candidate: str, expected: str) -> bool:
    return _norm(candidate) == _norm(expected)

def norm_score(score_1_5: float) -> float:
    return (score_1_5 - 1.0) / 4.0

def mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0

def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), pct))

def result_field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)

def is_sentinel(item: Dict[str, Any]) -> bool:
    """Item cuya respuesta esperada es un mensaje centinela (fuera de alcance / sin info)."""
    subg = _norm(item.get("retrieved_subgraph", ""))
    ans = _norm(item.get("answer", ""))
    sentinels = {_norm(gen.OUT_OF_SCOPE), _norm(gen.NO_INFO), ""}
    return subg in sentinels or ans in sentinels

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def judge_answer(
    api_key: str,
    model: str,
    question: str,
    subgraph: str,
    gt_answer: str,
    candidate: str,
    retries: int = 3,
) -> Dict[str, Any]:
    """Llama a Anthropic y devuelve los scores 1-5 + justificación."""
    user = (
        f"PREGUNTA:\n{question}\n\n"
        f"RESULTADOS DEL GRAFO (fuente de verdad):\n{subgraph}\n\n"
        f"RESPUESTA DE REFERENCIA:\n{gt_answer}\n\n"
        f"RESPUESTA CANDIDATA:\n{candidate}\n\n"
        "Devolvé el JSON con los scores."
    )
    payload = {
        "model": model,
        "max_tokens": 512,
        "system": JUDGE_SYSTEM,
        "messages": [{"role": "user", "content": user}],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            response = requests.post(
                ANTHROPIC_URL,
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
            judge_text = response.json()["content"][0]["text"]
            data = _extract_json(judge_text)
            return {
                "factual_correctness": int(data["factual_correctness"]),
                "completeness": int(data["completeness"]),
                "faithfulness": int(data["faithfulness"]),
                "justification": str(data.get("justification", "")),
            }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(2 * (attempt + 1))

    raise RuntimeError(f"El juez falló tras {retries} intentos: {last_err}")

def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"El juez no devolvió JSON: {text[:200]!r}")
    return dict(json.loads(match.group(0)))


def run_qa_evaluation(
    dataset_path: Path,
    api_key: Optional[str],
    judge_model: str,
    max_questions: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    items = load_qa_items(dataset_path)
    if max_questions is not None:
        items = items[:max_questions]

    uri, user, password = neo4j_config()

    retriever = GraphRAGRetriever(
        neo4j_uri=uri,
        neo4j_user=user,
        neo4j_password=password,
    )

    records: List[Dict[str, Any]] = []
    try:
        for index, item in enumerate(items, start=1):
            qid = item.get("id", index)
            question = (item.get("pregunta") or item.get("question") or "").strip()
            if not question:
                log.warning("Pregunta %s vacía; se omite", qid)
                continue

            gt_answer = str(item.get("answer", "") or "")
            gt_subgraph = str(item.get("retrieved_subgraph", "") or "")
            sentinel = is_sentinel(item)

            log.info("[%s/%s] Pregunta id=%s", index, len(items), qid)
            started = time.perf_counter()
            candidate = ""
            generated_cypher = ""
            n_chunks = 0
            query_error = ""

            try:
                result = retriever.query(question)
                candidate = str(result_field(result, "answer", "") or "").strip()
                generated_cypher = str(
                    result_field(result, "cypher_query", "") or ""
                )
                chunks = result_field(result, "chunks", []) or []
                n_chunks = len(chunks)
            except Exception as exc:  # noqa: BLE001
                query_error = f"{type(exc).__name__}: {exc}"
                log.exception("Falló query para la pregunta %s", qid)

            latency = time.perf_counter() - started

            record: Dict[str, Any] = {
                "id": qid,
                "category": item.get("categoria", item.get("category", "")),
                "question": question,
                "gt_answer": gt_answer,
                "gt_cypher_query": item.get("cypher_query", ""),
                "gt_retrieved_subgraph": gt_subgraph,
                "candidate": candidate,
                "generated_cypher_query": generated_cypher,
                "latency_s": round(latency, 4),
                "n_chunks": n_chunks,
                "is_sentinel": sentinel,
                "sentinel_correct": None,
                "scores": None,
                "quality_overall": None,
                "justification": "",
                "query_error": query_error,
                "judge_error": "",
            }

            if query_error:
                records.append(record)
                continue

            if sentinel:
                record["sentinel_correct"] = sentinel_matches(candidate, gt_answer)
                log.info(
                    "  centinela=%s latencia=%.2fs",
                    "OK" if record["sentinel_correct"] else "FAIL",
                    latency,
                )
                records.append(record)
                continue
            
            elif api_key:
                try:
                    scores = judge_answer(
                        api_key,
                        judge_model,
                        question,
                        gt_subgraph,
                        gt_answer,
                        candidate,
                    )
                    record["scores"] = {dimension: scores[dimension] for dimension in JUDGE_DIMS}
                    record["justification"] = scores["justification"]
                    record["quality_overall"] = norm_score(
                        mean([record["scores"][dimension] for dimension in JUDGE_DIMS])
                    )
                    log.info(
                        "  calidad=%.3f latencia=%.2fs",
                        record["quality_overall"],
                        latency,
                    )
                except Exception as exc:  # noqa: BLE001
                    record["judge_error"] = f"{type(exc).__name__}: {exc}"
                    log.exception("Falló el juez para la pregunta %s", qid)
            else:
                log.info("  latencia=%.2fs (sin juez)", latency)

            records.append(record)
    finally:
        close_method = getattr(retriever, "close", None)
        if callable(close_method):
            close_method()
    judge_backend = "anthropic" if api_key else "none"
    return records, aggregate_results(records, judge_backend, judge_model)


def aggregate_results(
    records: List[Dict[str, Any]],
    judge_backend: str,
    judge_model: str,
) -> Dict[str, Any]:
    judged = [record for record in records if record.get("scores") is not None]
    sentinels = [record for record in records if record.get("is_sentinel")]
    query_errors = [record for record in records if record.get("query_error")]
    judge_errors = [record for record in records if record.get("judge_error")]
    latencies = [
        float(record["latency_s"])
        for record in records
        if record.get("latency_s") is not None
    ]

    dim_means = {
        dimension: mean(
            [
                norm_score(float(record["scores"][dimension]))
                for record in judged
            ]
        )
        for dimension in JUDGE_DIMS
    }

    quality_overall = mean(
        [float(record["quality_overall"]) for record in judged]
    )
    sentinel_accuracy = mean(
        [1.0 if record.get("sentinel_correct") else 0.0 for record in sentinels]
    )

    by_cat: Dict[str, float] = {}
    cats = sorted({r["category"] for r in records})
    for cat in cats:
        cat_judged = [r for r in judged if r["category"] == cat]
        cat_sent = [r for r in sentinels if r["category"] == cat]
        if cat_judged:
            by_cat[cat] = mean(
                [norm_score(mean([r["scores"][d] for d in JUDGE_DIMS])) for r in cat_judged]
            )
        elif cat_sent:
            by_cat[cat] = mean([1.0 if r["sentinel_correct"] else 0.0 for r in cat_sent])
        else:
            by_cat[cat] = 0.0

    return {
        "judge": {
            "backend": judge_backend,
            "model": judge_model,
        },
        "n_total": len(records),
        "n_judged": len(judged),
        "n_sentinel": len(sentinels),
        "n_query_errors": len(query_errors),
        "n_judge_errors": len(judge_errors),
        "dim_means": dim_means,
        "quality_overall": quality_overall,
        "sentinel_accuracy": sentinel_accuracy,
        "by_category": by_cat,
        "latency": {
            "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
            "median": round(statistics.median(latencies), 2) if latencies else 0.0,
            "p95": round(percentile(latencies, 95), 2) if latencies else 0.0,
            "n": len(latencies),
        },
        "query_error_ids": [record.get("id") for record in query_errors],
        "judge_error_ids": [record.get("id") for record in judge_errors],
    }


# -----------------------------------------------------------------------------
# Salidas: JSON, gráficas y HTML
# -----------------------------------------------------------------------------


def generate_charts(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    images_dir: Path,
) -> Dict[str, Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    judged = [
        record for record in records if record.get("quality_overall") is not None
    ]
    if judged:
        labels = [str(record["id"]) for record in judged]
        values = [float(record["quality_overall"]) for record in judged]
        height = max(4.0, 0.24 * len(judged) + 1.5)

        fig, ax = plt.subplots(figsize=(8, height))
        y = np.arange(len(judged))[::-1]
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score normalizado (0-1)")
        ax.set_ylabel("ID de pregunta")
        ax.set_title("Calidad global por pregunta")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = images_dir / "quality_by_question.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["quality_by_question"] = path

        dimension_values = [
            float(summary["dim_means"][dimension]) for dimension in JUDGE_DIMS
        ]
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar([DIM_LABELS[dimension] for dimension in JUDGE_DIMS], dimension_values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score normalizado (0-1)")
        ax.set_title("Promedio por dimensión")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = images_dir / "dimensions.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["dimensions"] = path

        counts = {score: 0 for score in range(1, 6)}
        for record in judged:
            for dimension in JUDGE_DIMS:
                counts[int(record["scores"][dimension])] += 1

        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.bar([str(score) for score in range(1, 6)], [counts[score] for score in range(1, 6)])
        ax.set_xlabel("Score del juez")
        ax.set_ylabel("Cantidad")
        ax.set_title("Distribución de scores del juez")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = images_dir / "score_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["score_distribution"] = path

    latency_records = [
        record for record in records if record.get("latency_s") is not None
    ]
    if latency_records:
        labels = [str(record["id"]) for record in latency_records]
        values = [float(record["latency_s"]) for record in latency_records]
        height = max(4.0, 0.24 * len(latency_records) + 1.5)

        fig, ax = plt.subplots(figsize=(8, height))
        y = np.arange(len(latency_records))[::-1]
        ax.barh(y, values)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Segundos")
        ax.set_ylabel("ID de pregunta")
        ax.set_title("Latencia de consulta y generación de respuesta")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = images_dir / "latency_by_question.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["latency_by_question"] = path

    categories = list(summary.get("by_category", {}).keys())
    if categories:
        values = [float(summary["by_category"][category]) for category in categories]
        fig, ax = plt.subplots(figsize=(max(7, len(categories) * 1.1), 4.5))
        ax.bar(categories, values)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score normalizado (0-1)")
        ax.set_title("Score por categoría")
        ax.tick_params(axis="x", rotation=25)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        path = images_dir / "categories.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["categories"] = path

    return paths


def metric_cell(value: float) -> str:
    if value >= 0.7:
        color = "#27ae60"
    elif value >= 0.4:
        color = "#e67e22"
    else:
        color = "#e74c3c"
    return (
        f'<td style="background:{color}18;color:{color};font-weight:600;">'
        f"{value:.3f}</td>"
    )


def generate_html_report(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
    chart_paths: Dict[str, Path],
    output_path: Path,
) -> None:
    def relative_image(key: str) -> str:
        return str(chart_paths[key].relative_to(output_path.parent)).replace("\\", "/")

    dimension_headers = "".join(
        f"<th>{DIM_LABELS[dimension]}</th>" for dimension in JUDGE_DIMS
    )
    dimension_cells = "".join(
        metric_cell(float(summary["dim_means"].get(dimension, 0.0)))
        for dimension in JUDGE_DIMS
    )

    chart_cards: List[str] = []
    chart_titles = {
        "quality_by_question": "Calidad global por pregunta",
        "dimensions": "Promedio por dimensión",
        "score_distribution": "Distribución de scores del juez",
        "categories": "Score por categoría",
        "latency_by_question": "Latencia end-to-end por pregunta",
    }
    for key, title in chart_titles.items():
        if key in chart_paths:
            chart_cards.append(
                '<div class="card">'
                f"<h2>{title}</h2>"
                f'<img src="{relative_image(key)}" alt="{title}">'
                "</div>"
            )

    rows: List[str] = []
    for record in records:
        scores = record.get("scores") or {}
        score_cells = "".join(
            f"<td>{scores.get(dimension, '—')}</td>" for dimension in JUDGE_DIMS
        )
        quality = record.get("quality_overall")
        quality_text = f"{quality:.3f}" if isinstance(quality, (float, int)) else "—"
        candidate = str(record.get("candidate", ""))
        error = record.get("query_error") or record.get("judge_error") or ""

        rows.append(
            "<tr>"
            f"<td>{html.escape(str(record.get('id', '')))}</td>"
            f"<td>{html.escape(str(record.get('category', '')))}</td>"
            f"<td class='question'>{html.escape(str(record.get('question', '')))}</td>"
            f"<td>{quality_text}</td>"
            f"{score_cells}"
            f"<td>{float(record.get('latency_s', 0.0)):.2f}s</td>"
            f"<td>{'sí' if record.get('is_sentinel') else 'no'}</td>"
            f"<td>{html.escape(str(record.get('sentinel_correct'))) if record.get('is_sentinel') else '—'}</td>"
            f"<td>{record.get('n_chunks', 0)}</td>"
            f"<td class='candidate'>{html.escape(candidate[:600])}{'…' if len(candidate) > 600 else ''}</td>"
            f"<td class='cypher'><pre>{html.escape(str(record.get('generated_cypher_query', '')))}</pre></td>"
            f"<td class='error'>{html.escape(str(error))}</td>"
            "</tr>"
        )

    judge = summary["judge"]
    html_document = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Evaluación end-to-end GraphRAG</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin:0; padding:2rem 3rem; background:#f0f2f5; color:#2c3e50; }}
    h1 {{ margin-bottom:.25rem; }}
    h2 {{ font-size:1.15rem; border-left:4px solid #3498db; padding-left:10px; }}
    .subtitle {{ color:#666; margin-top:0; }}
    .card {{ background:#fff; border-radius:10px; padding:1.2rem; box-shadow:0 2px 8px rgba(0,0,0,.08); margin:1rem 0; overflow-x:auto; }}
    table {{ border-collapse:collapse; width:100%; font-size:.8rem; }}
    th {{ background:#2c3e50; color:white; padding:7px 9px; text-align:center; white-space:nowrap; }}
    td {{ border-bottom:1px solid #eee; padding:7px 9px; vertical-align:top; }}
    td.question {{ min-width:260px; }}
    td.candidate {{ min-width:360px; }}
    td.cypher {{ min-width:300px; }}
    td.error {{ color:#c0392b; min-width:180px; }}
    pre {{ white-space:pre-wrap; margin:0; font-size:.75rem; }}
    img {{ max-width:950px; width:100%; height:auto; border-radius:6px; }}
  </style>
</head>
<body>
  <h1>Evaluación end-to-end GraphRAG</h1>
  <p class="subtitle">
    Preguntas: {summary['n_total']} ·
    Juzgadas: {summary['n_judged']} ·
    Centinelas: {summary['n_sentinel']} ·
    Errores de consulta: {summary['n_query_errors']} ·
    Errores del juez: {summary['n_judge_errors']} ·
    Juez: <code>{html.escape(judge['backend'])}/{html.escape(judge['model'])}</code>
  </p>

  <div class="card">
    <h2>Resumen global</h2>
    <table>
      <thead>
        <tr>{dimension_headers}<th>Calidad global</th><th>Centinelas</th><th>Latencia media</th><th>Mediana</th><th>p95</th></tr>
      </thead>
      <tbody>
        <tr>
          {dimension_cells}
          {metric_cell(float(summary['quality_overall']))}
          {metric_cell(float(summary['sentinel_accuracy']))}
          <td>{summary['latency']['mean']:.2f}s</td>
          <td>{summary['latency']['median']:.2f}s</td>
          <td>{summary['latency']['p95']:.2f}s</td>
        </tr>
      </tbody>
    </table>
  </div>

  {''.join(chart_cards)}

  <div class="card">
    <h2>Detalle por pregunta</h2>
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Categoría</th><th>Pregunta</th><th>Calidad</th>
          {dimension_headers}
          <th>Latencia</th><th>Centinela</th><th>Centinela OK</th>
          <th>Chunks</th><th>Respuesta candidata</th><th>Cypher generada</th><th>Error</th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
  </div>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_document, encoding="utf-8")


def save_results(
    records: List[Dict[str, Any]],
    summary: Dict[str, Any],
) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details_path = RESULTS_DIR / "qa_query_details.json"
    summary_path = RESULTS_DIR / "qa_query_summary.json"
    html_path = RESULTS_DIR / "qa_query_report.html"

    details_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    charts = generate_charts(records, summary, RESULTS_DIR / "images")
    generate_html_report(records, summary, charts, html_path)

    log.info("Detalle JSON: %s", details_path)
    log.info("Resumen JSON: %s", summary_path)
    log.info("Reporte HTML: %s", html_path)


# -----------------------------------------------------------------------------
# CLI y ejecución
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta el pipeline local y evalúa las respuestas producidas "
            "por GraphRAGRetriever con un único juez."
        )
    )
    parser.add_argument(
        "qa_dataset",
        type=Path,
        help="Dataset JSON obligatorio con preguntas, retrieved_subgraph y answer.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_ANTHROPIC_JUDGE,
        help=(
            "Modelo de Anthropic para usar como juez. "
            f"Default: {DEFAULT_ANTHROPIC_JUDGE}."
        ),
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="No ejecuta el LLM-as-a-judge; solo consulta y mide latencias.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Limita las preguntas para una prueba rápida.",
    )
    parser.add_argument(
        "--skip-docling",
        action="store_true",
        help="No ejecuta Docling y usa los JSON existentes en data/docling.",
    )
    parser.add_argument(
        "--skip-chunking",
        action="store_true",
        help="No genera chunks y usa los existentes en data/chunks.",
    )
    parser.add_argument(
        "--qa-only",
        action="store_true",
        help=(
            "Ejecuta solamente preguntas, respuestas, evaluación y reportes. "
            "Omite Docling, chunking, extracción y carga en Neo4j."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    if args.skip_chunking and not args.skip_docling and not args.qa_only:
        raise ValueError(
            "--skip-chunking requiere también --skip-docling, "
            "porque de lo contrario los chunks podrían no coincidir "
            "con los documentos de Docling."
        )
    try:
        if args.qa_only:
            log.info(
                "Modo --qa-only: se omiten Docling, chunking, extracción y carga en Neo4j"
            )
        else:
            if args.skip_docling:
                log.info("Docling omitido por --skip-docling")
            else:
                log.info("ETAPA 1/6 - Docling")
                run_docling()

            if args.skip_chunking:
                log.info("Chunking omitido por --skip-chunking")
            else:
                log.info("ETAPA 2/6 - Chunking")
                processed, errors, _ = run_chunking()
                if errors:
                    raise RuntimeError(
                        f"Falló el chunking de {errors} documentos. "
                        "Se cancela la extracción para no generar un grafo incompleto."
                    )
                if processed == 0:
                    raise RuntimeError("No se procesó correctamente ningún documento.")
            log.info("ETAPA 3/6 - Extracción de entidades y relaciones")
            graph_json = run_extraction()

            log.info("ETAPA 4/6 - Limpieza y carga en Neo4j")
            load_into_neo4j(graph_json)

        log.info("ETAPA 5/6 - Consultas y evaluación QA")
        
        api_key: Optional[str] = None
        if not args.no_judge:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print(
                    "ADVERTENCIA: falta ANTHROPIC_API_KEY; "
                    "corriendo en modo --no-judge.",
                    flush=True,
                )

        records, summary = run_qa_evaluation(
            dataset_path=args.qa_dataset,
            api_key=api_key,
            judge_model=args.judge_model,
            max_questions=args.max_questions,
        )

        log.info("ETAPA 6/6 - JSON, gráficas y HTML")
        save_results(records, summary)

        print("\n" + "=" * 78)
        print("RESULTADO END-TO-END")
        print("=" * 78)
        print(f"Preguntas:           {summary['n_total']}")
        print(f"Calidad global:      {summary['quality_overall']:.3f}")
        print(f"Centinelas:          {summary['sentinel_accuracy']:.3f}")
        print(f"Latencia media:      {summary['latency']['mean']:.2f}s")
        print(f"Latencia p95:        {summary['latency']['p95']:.2f}s")
        print(f"Errores de consulta: {summary['n_query_errors']}")
        print(f"Errores del juez:    {summary['n_judge_errors']}")
        print("=" * 78)
        print(f"Resumen JSON: {RESULTS_DIR / 'qa_query_summary.json'}")
        print(f"Detalle JSON: {RESULTS_DIR / 'qa_query_details.json'}")
        print(f"Reporte HTML: {RESULTS_DIR / 'qa_query_report.html'}")
        return 0

    except Exception as exc:  # noqa: BLE001
        log.exception("PIPELINE FALLÓ: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())