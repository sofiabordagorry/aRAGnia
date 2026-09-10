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

Regenerar únicamente gráficas y HTML desde los resultados existentes:

    python evaluation/scripts/evaluate_end_to_end.py \
        --charts-only

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

from aragnia.config import EMBED_MODEL_ID
from aragnia.extraction.ie import EntityExtractor, ExtractionResult
from aragnia.graph.builder import GraphBuilder
from aragnia.graph.graph_loader import load_graph_json
from aragnia.ingest.chunker import chunk_document, get_native_chunker
from aragnia.ingest.docling_parser import (
    DEFAULT_DOCLING_DIR,
    DocumentAlreadyProcessed,
    parse_corpus,
    parse_single_document,
)
from aragnia.ingest.table_extractors import convert_tables_to_chunks
from aragnia.llm.llm_provider import (
    HuggingFaceClient,
    get_llm_client,
)
try:
    from aragnia.retrieval.graph_retriever import GraphRAGRetriever
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

DEFAULT_DATASET = EVAL_DIR / "ground_truth" / "extraction" / "ground_truth_kg_evaluation.json"

CORPUS_DIR = DATA_DIR / "corpus"
DOCLING_DIR = DEFAULT_DOCLING_DIR
CHUNKS_DIR = DATA_DIR / "chunks"
EXTRACTED_FILENAME = "entity_documents.json"
RESULTS_DIR = EVAL_DIR / "results" / "end_to_end"
DEFAULT_DETAILS_PATH = RESULTS_DIR / "end_to_end_details.json"
DEFAULT_SUMMARY_PATH = RESULTS_DIR / "end_to_end_summary.json"


ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_JUDGE = "claude-opus-4-8"

JUDGE_DIMS = ["factual_correctness", "completeness", "faithfulness"]
DIM_LABELS = {
    "factual_correctness": "Correctitud",
    "completeness": "Completitud",
    "faithfulness": "Fidelidad",
}

PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

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
    subg = _norm(item.get("cypher_result", ""))
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


def unload_huggingface_model(
    model_id: str,
    retriever: GraphRAGRetriever,
) -> None:
    """Elimina todas las referencias al modelo y libera su memoria CUDA."""
    import gc
    import torch

    # GraphRAGRetriever puede conservar referencias al mismo cliente.
    for attribute in (
        "cypher_llm_client",
        "answer_llm_client",
        "generation_llm_client",
        "llm_client",
    ):
        client = getattr(retriever, attribute, None)

        if getattr(client, "model_id", None) == model_id:
            setattr(retriever, attribute, None)

    client = HuggingFaceClient._instances.pop(model_id, None)

    if client is not None:
        client.pipe = None
        client.model = None
        client.tokenizer = None
        del client

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

        free, total = torch.cuda.mem_get_info()

        log.info(
            "Modelo descargado: %s | GPU libre: %.2f/%.2f GiB",
            model_id,
            free / 1024**3,
            total / 1024**3,
        )

def run_qa_evaluation(
    dataset_path: Path,
    api_key: Optional[str],
    judge_model: str,
    max_questions: Optional[int] = None,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    items = load_qa_items(dataset_path)
    if max_questions is not None:
        items = items[:max_questions]
    retrieval_model = os.getenv("HF_RETRIEVAL_MODEL")
    generation_model = os.getenv("HF_GENERATION_MODEL")
    
    if not retrieval_model:
        raise RuntimeError("Falta HF_RETRIEVAL_MODEL")

    if not generation_model:
        raise RuntimeError("Falta HF_GENERATION_MODEL")
    
    os.environ["HF_GENERATION_MODEL"] = retrieval_model
    uri, user, password = neo4j_config()

    retrieval_records: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    retriever = GraphRAGRetriever(
        neo4j_uri=uri,
        neo4j_user=user,
        neo4j_password=password,
    )

    try:
        for index, item in enumerate(items, start=1):
            qid = item.get("id", index)
            question = (item.get("pregunta") or item.get("question") or "").strip()
            if not question:
                log.warning("Pregunta %s vacía; se omite", qid)
                continue

            gt_answer = str(item.get("answer", "") or "")
            gt_subgraph = str(item.get("cypher_result", "") or "")
            sentinel = is_sentinel(item)

            log.info("[%s/%s] Pregunta id=%s", index, len(items), qid)
            
            records_result: List[Any] = []
            query_error = ""
            cypher_query: str = ""
            invalid_result = None

            started = time.perf_counter()

            try:
                invalid_result, records_result, cypher_query = retriever.generate_cypher_query_result(
                    user_query=question
                )
            except Exception as exc:  # noqa: BLE001
                query_error = f"{type(exc).__name__}: {exc}"
                log.exception("Falló query para la pregunta %s", qid)

            latency = time.perf_counter() - started
            retrieval_records.append(
                {
                    "id": qid,
                    "category": item.get("categoria", item.get("category", "")),
                    "question": question,
                    "gt_answer": gt_answer,
                    "gt_cypher_query": item.get("cypher_query", ""),
                    "gt_cypher_result": gt_subgraph,
                    "records_result": records_result,
                    "generated_cypher_query": cypher_query,
                    "latency_s": round(latency, 4),
                    "is_sentinel": sentinel,
                    "query_error": query_error,
                    "invalid_result": invalid_result,
                }
            )

        unload_huggingface_model(
            model_id=retrieval_model,
            retriever=retriever,
        )
        os.environ["HF_GENERATION_MODEL"] = generation_model
        retriever.answer_llm_client = get_llm_client(model=generation_model)
        retriever.cypher_llm_client = None
        for index, retrieval_record in enumerate(retrieval_records,start=1,):
            qid = retrieval_record.get("id")
            category = retrieval_record.get("category", "")
            question = retrieval_record.get("question", "")
            gt_answer = retrieval_record.get("gt_answer", "")
            gt_cypher_query = retrieval_record.get("gt_cypher_query", "")
            gt_cypher_result = retrieval_record.get("gt_cypher_result", "")
            records_result = retrieval_record.get("records_result", [])
            cypher_query = retrieval_record.get("generated_cypher_query", "")
            latency_s = retrieval_record.get("latency_s", 0.0)
            sentinel = retrieval_record.get("is_sentinel", False)
            query_error = retrieval_record.get("query_error", "")
            invalid_result = retrieval_record.get("invalid_result")

            candidate = ""
            generated_cypher = cypher_query
            n_chunks = 0
            generation_latency = 0.0
            if not query_error:
                started = time.perf_counter()
                try:
                    if not records_result:
                        result = invalid_result
                    else:
                        result = retriever.generate_result(
                            records=records_result, user_query=question, cypher_query=cypher_query
                        )

                    candidate = str(result_field(result, "answer", "") or "").strip()
                    generated_cypher = str(
                        result_field(result, "cypher_query", cypher_query) or cypher_query
                    )
                    chunks = result_field(result, "chunks", []) or []
                    n_chunks = len(chunks)
                except Exception as exc:  # noqa: BLE001
                    query_error = f"{type(exc).__name__}: {exc}"
                    log.exception("Falló query para la pregunta %s", qid)
                generation_latency = time.perf_counter() - started

            latency = float(latency_s) + generation_latency
            record: Dict[str, Any] = {
                "id": qid,
                "category": category,
                "question": question,
                "gt_answer": gt_answer,
                "gt_cypher_query": gt_cypher_query,
                "gt_cypher_result": gt_cypher_result,
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
                        gt_cypher_result,
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
        try:
            unload_huggingface_model(
                model_id=generation_model,
                retriever=retriever,
            )
        except Exception:  # noqa: BLE001
            log.exception(
                "No se pudo descargar el modelo de generación"
            )
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
        cat_scores = [
            norm_score(mean([r["scores"][d] for d in JUDGE_DIMS])) for r in cat_judged
        ] + [1.0 if r["sentinel_correct"] else 0.0 for r in cat_sent]
        if cat_scores:
            by_cat[cat] = mean(cat_scores)

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

def load_existing_results(
    details_path: Path = DEFAULT_DETAILS_PATH,
    summary_path: Path = DEFAULT_SUMMARY_PATH,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Carga el summary y los records de una evaluación ya ejecutada."""

    if not summary_path.exists():
        raise FileNotFoundError(
            f"No existe el JSON de resumen: {summary_path}"
        )

    if not details_path.exists():
        raise FileNotFoundError(
            f"No existe el JSON de detalle: {details_path}"
        )

    summary_data = json.loads(
        summary_path.read_text(encoding="utf-8")
    )
    details_data = json.loads(
        details_path.read_text(encoding="utf-8")
    )

    if not isinstance(summary_data, dict):
        raise ValueError(
            "El JSON de resumen debe contener un objeto."
        )

    if not isinstance(details_data, list):
        raise ValueError(
            "El JSON de detalle debe contener una lista de registros."
        )

    if not all(isinstance(record, dict) for record in details_data):
        raise ValueError(
            "Todos los elementos del JSON de detalle deben ser objetos."
        )

    summary: Dict[str, Any] = dict(summary_data)
    records: List[Dict[str, Any]] = list(details_data)

    return summary, records

# -----------------------------------------------------------------------------
# Salidas: JSON, gráficas y HTML
# -----------------------------------------------------------------------------



def build_generation_style_results(
    summary: Dict[str, Any],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Adapta summary + records al formato de evaluate_generation solo en memoria.

    El end-to-end usa dos modelos distintos:
    - HF_RETRIEVAL_MODEL para generar/ejecutar la consulta Cypher.
    - HF_GENERATION_MODEL para redactar la respuesta final.

    No genera ni guarda un archivo JSON combinado.
    """
    retrieval_model = str(
        os.getenv("HF_RETRIEVAL_MODEL")
        or summary.get("retrieval_model")
        or "retrieval-no-configurado"
    )
    generation_model = str(
        os.getenv("HF_GENERATION_MODEL")
        or summary.get("generation_model")
        or summary.get("model")
        or "generation-no-configurado"
    )

    backend = str(
        summary.get("backend")
        or os.getenv("E2E_BACKEND")
        or "huggingface"
    )
    prompt = str(
        summary.get("prompt")
        or os.getenv("E2E_PROMPT")
        or ""
    )

    # Las etiquetas se dividen en dos líneas para que sean legibles tanto en
    # las gráficas de Matplotlib como en las tablas y el mini análisis del HTML.
    display = str(
        os.getenv("E2E_DISPLAY")
        or (
            f"Retrieval: {retrieval_model}\n"
            f"Generación: {generation_model}"
        )
    )
    label = str(
        os.getenv("E2E_LABEL")
        or (
            f"Retrieval: {retrieval_model}\n"
            f"Generación: {generation_model} / {prompt}"
        )
    )

    return [
        {
            "display": display,
            "backend": backend,
            "model": f"{retrieval_model} -> {generation_model}",
            "retrieval_model": retrieval_model,
            "generation_model": generation_model,
            "prompt": prompt,
            "label": label,
            "n_judged": int(summary.get("n_judged", 0)),
            "n_sentinel": int(summary.get("n_sentinel", 0)),
            "dim_means": dict(summary.get("dim_means", {})),
            "quality_overall": float(summary.get("quality_overall", 0.0)),
            "sentinel_accuracy": float(summary.get("sentinel_accuracy", 0.0)),
            "by_category": dict(summary.get("by_category", {})),
            "latency": dict(summary.get("latency", {})),
            "records": records,
        }
    ]


def _zoom_floor(values: List[float], pad: float = 0.03) -> float:
    """Piso 'redondo' para un eje cuando los valores se agolpan cerca del techo (1.0).

    Devuelve el múltiplo de 0.05 inmediatamente por debajo de (min - pad), acotado a
    [0, 0.95]. Sirve para que las diferencias en la zona 0.89–1.0 sean visibles en vez
    de quedar aplastadas contra un eje que arranca en 0.
    """
    if not values:
        return 0.0
    lo = float(np.floor((min(values) - pad) * 20) / 20.0)
    return min(max(lo, 0.0), 0.95)


def _heatmap(
    path: Path, matrix: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, fig_h: float
) -> Path:
    """Heatmap combos × columnas con anotación numérica y colormap con zoom cerca del techo.

    Reemplaza a las barras agrupadas de 25 series (ilegibles): cada fila es una
    combinación (mejor arriba) y cada columna una dimensión/categoría.
    """
    finite = matrix[np.isfinite(matrix)]
    vmin = _zoom_floor(list(finite)) if finite.size else 0.0
    fig, ax = plt.subplots(figsize=(max(4.5, 1.25 * len(col_labels) + 2.5), fig_h))
    im = ax.imshow(matrix, aspect="auto", cmap="YlGn", vmin=vmin, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "–", va="center", ha="center", fontsize=7, color="#999")
                continue
            frac = (v - vmin) / (1.0 - vmin) if vmin < 1.0 else 1.0
            ax.text(j, i, f"{v:.3f}", va="center", ha="center", fontsize=7,
                    color="white" if frac > 0.6 else "#222")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="Score (0–1)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def generate_charts(results: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Path]:
    """Gráficas en formato imprimible (para el informe).

    Todas las combinaciones se muestran en barras horizontales / heatmaps con las
    etiquetas legibles (una por fila), ordenadas mejor-primero, y con ejes/colormap
    recortados cerca del techo para que las diferencias no queden aplastadas.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    results = sorted(results, key=lambda r: r["quality_overall"], reverse=True)
    labels = [r["label"] for r in results]
    n = len(results)
    y = np.arange(n)[::-1]  # fila 0 (mejor) arriba
    fig_h = max(4.0, 0.34 * n + 1.6)

    # 1) Calidad global: barras horizontales con eje recortado; centinelas como marcador
    #    (son ~1.0 en todas las combinaciones, así que un rombo alcanza para verlo).
    quality = [r["quality_overall"] for r in results]
    sent = [r["sentinel_accuracy"] for r in results]
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.barh(y, quality, 0.62, color=PALETTE[0], label="Calidad global", zorder=3)
    ax.scatter(sent, y, marker="D", s=20, color=PALETTE[1], label="Centinelas (exact-match)", zorder=4)
    for yi, q in zip(y, quality):
        ax.text(q - 0.001, yi, f"{q:.3f}", va="center", ha="right", fontsize=7, color="white", zorder=5)
    ax.set_yticks(y)
    ax.tick_params(axis="y", left=False, labelleft=False)
    ax.set_xlim(_zoom_floor(quality + sent), 1.005)
    ax.set_xlabel("Score (0–1)")
    ax.set_title("Calidad global y manejo de centinelas")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = images_dir / "chart_overall.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["overall"] = p

    # 2) Score por dimensión → barras.
    dimensions = JUDGE_DIMS
    dim_labels = [DIM_LABELS[d] for d in dimensions]

    values = [results[0]["dim_means"][d] for d in dimensions]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    bars = ax.bar(
        dim_labels,
        values,
        width=0.6,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Score por dimensión")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    p = images_dir / "chart_dimensions.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    paths["dimensions"] = p

    # 3) Score por categoría → barras, similar a score por dimensión.
    all_cats = sorted({c for r in results for c in r["by_category"]})

    values = [results[0]["by_category"].get(c, 0.0) for c in all_cats]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bars = ax.bar(
        all_cats,
        values,
        width=0.6,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.015,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Score por categoría")

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.xticks(rotation=20, ha="right")

    fig.tight_layout()

    p = images_dir / "chart_categories.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)

    paths["categories"] = p

    # 4) Latencia (media + p95): barras horizontales simples.
    by_lat = sorted(results, key=lambda r: r["latency"]["mean"])

    # Si hay una sola combinación, usamos sus valores
    mean_val = by_lat[0]["latency"]["mean"]
    p95_val = by_lat[0]["latency"]["p95"]

    y_labels = ["Media", "p95"]
    values = [mean_val, p95_val]
    colors = [PALETTE[3], PALETTE[1]]

    fig, ax = plt.subplots(figsize=(8.0, 4.5))

    ax.barh(y_labels, values, color=colors)

    ax.set_xlabel("Latencia (segundos)")
    ax.set_ylabel("Estadístico de latencia")
    ax.set_title("Latencia de generación por combinación")

    max_latency = max(values)
    ax.set_xlim(0, max_latency * 1.08)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    p = images_dir / "chart_latency.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["latency"] = p

    # 5) Distribución de scores del juez (1-5): barras horizontales apiladas (% del total).
    #    Revela la FORMA, no solo la media: dos combinaciones con media parecida pueden
    #    diferir en cuántos fallos graves (score 1) tienen. Poolea las 3 dimensiones.
    score_colors = {5: "#27ae60", 4: "#7fc97f", 3: "#f39c12", 2: "#e67e22", 1: "#e74c3c"}
    dist_pct: Dict[int, List[float]] = {s: [] for s in (5, 4, 3, 2, 1)}
    any_scores = False
    for r in results:
        counts = {s: 0 for s in (1, 2, 3, 4, 5)}
        total = 0
        for rec in r.get("records", []):
            sc = rec.get("scores")
            if not sc:
                continue
            for d in JUDGE_DIMS:
                counts[int(sc[d])] += 1
                total += 1
        any_scores = any_scores or total > 0
        for s in (5, 4, 3, 2, 1):
            dist_pct[s].append(100.0 * counts[s] / total if total else 0.0)

    if any_scores:
        fig, ax = plt.subplots(figsize=(8.0, fig_h))
        left = np.zeros(n)
        for s in (5, 4, 3, 2, 1):
            vals = np.array(dist_pct[s])
            ax.barh(y, vals, 0.62, left=left, label=str(s), color=score_colors[s])
            left += vals
        ax.set_yticks(y)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% de scores del juez")
        ax.set_title("Distribución de scores del juez (1–5)")
        ax.legend(title="Score", fontsize=8, ncol=5, loc="upper center",
                  bbox_to_anchor=(0.5, -0.06 - 2.0 / fig_h), framealpha=0.9)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = images_dir / "chart_score_distribution.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["score_dist"] = p

    return paths

def _val_color(val: float) -> str:
    if val >= 0.7:
        return "#27ae60"
    if val >= 0.4:
        return "#e67e22"
    return "#e74c3c"


def _metric_cell(val: float) -> str:
    color = _val_color(val)
    return f'<td style="background:{color}18;color:{color};font-weight:600;">{val:.3f}</td>'


def generate_html_report(
    results: List[Dict[str, Any]],
    output_path: Path,
    chart_paths: Dict[str, Path],
    judge_model: str,
    judged: bool,
) -> None:
    all_cats = sorted({c for r in results for c in r["by_category"]})

    def html_text(value: Any) -> str:
        """Escapa HTML y conserva los saltos de línea como <br>."""
        return html.escape(str(value)).replace("\n", "<br>")

    def html_label(result: Dict[str, Any]) -> str:
        return html_text(result.get("label", ""))

    def overall_rows() -> str:
        rows = []
        for r in results:
            row = f"<tr><td class='llm-name'>{html_label(r)}</td>"
            for d in JUDGE_DIMS:
                row += _metric_cell(r["dim_means"][d])
            row += _metric_cell(r["quality_overall"])
            row += _metric_cell(r["sentinel_accuracy"])
            lat = r["latency"]
            row += f"<td>{lat['mean']:.1f}s</td><td>{lat['median']:.1f}s</td><td>{lat['p95']:.1f}s</td>"
            row += "</tr>"
            rows.append(row)
        return "\n".join(rows)

    def category_rows() -> str:
        rows = []
        for r in results:
            row = f"<tr><td class='llm-name'>{html_label(r)}</td>"
            for c in all_cats:
                row += _metric_cell(r["by_category"].get(c, 0.0))
            row += "</tr>"
            rows.append(row)
        return "\n".join(rows)

    cat_headers = "".join(f"<th>{c}</th>" for c in all_cats)

    analysis_items = []
    if judged:
        best_q = max(results, key=lambda r: r["quality_overall"])
        analysis_items.append(
            f"<strong>{html_label(best_q)}</strong> logra la mejor calidad global "
            f"({best_q['quality_overall']:.3f})."
        )

        by_model: Dict[str, List[float]] = {}
        for r in results:
            by_model.setdefault(r["display"], []).append(r["quality_overall"])
        best_model = max(by_model.items(), key=lambda kv: max(kv[1]))
        analysis_items.append(
            f"El modelo <strong>{html_text(best_model[0])}</strong> es el de mayor calidad pico "
            f"({max(best_model[1]):.3f})."
        )
        by_prompt: Dict[str, List[float]] = {}
        for r in results:
            by_prompt.setdefault(r["prompt"], []).append(r["quality_overall"])
        best_prompt = max(by_prompt.items(), key=lambda kv: mean(kv[1]))
        analysis_items.append(
            f"El prompt <strong>{html_text(best_prompt[0])}</strong> es el mejor en promedio "
            f"({mean(best_prompt[1]):.3f})."
        )
        best_sent = max(results, key=lambda r: r["sentinel_accuracy"])
        analysis_items.append(
            f"<strong>{html_label(best_sent)}</strong> maneja mejor los centinelas "
            f"(accuracy {best_sent['sentinel_accuracy']:.3f})."
        )
    fastest = min(results, key=lambda r: r["latency"]["mean"] or float("inf"))
    analysis_items.append(
        f"<strong>{html_label(fastest)}</strong> es el más rápido "
        f"(latencia media {fastest['latency']['mean']:.1f}s, p95 {fastest['latency']['p95']:.1f}s)."
    )
    analysis_html = "\n".join(f"<li>{item}</li>" for item in analysis_items)

    img_overall = chart_paths["overall"].relative_to(output_path.parent)
    img_dims = chart_paths["dimensions"].relative_to(output_path.parent)
    img_cats = chart_paths["categories"].relative_to(output_path.parent)
    img_lat = chart_paths["latency"].relative_to(output_path.parent)

    dist_block = ""
    if "score_dist" in chart_paths:
        img_dist = chart_paths["score_dist"].relative_to(output_path.parent)
        dist_block = (
            '<div class="card">\n'
            "    <h2>Distribución de scores del juez (1-5)</h2>\n"
            f'    <div class="chart-wrap"><img src="{img_dist}" alt="Distribución de scores"></div>\n'
            "  </div>"
        )

    dim_headers = "".join(f"<th>{DIM_LABELS[d]}</th>" for d in JUDGE_DIMS)

    html_document = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Evaluación de Generación – GraphRAG</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0; padding: 2rem 3rem; background: #f0f2f5; color: #2c3e50; }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.2rem; color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 2rem; }}
    p.subtitle {{ color: #666; margin-top: 0; }}
    .card {{ background: white; border-radius: 10px; padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 1rem 0; }}
    .overflow-x {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
    th {{ background: #2c3e50; color: white; padding: 7px 10px; text-align: center; white-space: nowrap; font-size: 0.78rem; }}
    th:first-child {{ text-align: left; }}
    td {{ padding: 7px 10px; text-align: center; border-bottom: 1px solid #f0f0f0; }}
    td.llm-name {{ text-align: left; font-weight: 600; white-space: normal;
      line-height: 1.45; min-width: 360px; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafbfc; }}
    .chart-wrap {{ max-width: 900px; }}
    .chart-wrap img {{ width: 100%; height: auto; border-radius: 6px; }}
    ul.analysis {{ line-height: 1.9; }}
  </style>
</head>
<body>
  <h1>Evaluación de Generación – Comparación de LLMs y Prompts</h1>
  <p class="subtitle">{len(results)} combinaciones modelo×prompt evaluadas sobre el GT de validación. Juez: <code>{judge_model}</code>. Scores normalizados 0-1.</p>

  <div class="card">
    <h2>Mini Análisis</h2>
    <ul class="analysis">
      {analysis_html}
    </ul>
  </div>

  <div class="card">
    <h2>Resumen Global</h2>
    <div class="overflow-x">
      <table>
        <thead>
          <tr>
            <th rowspan="2">Modelo / Prompt</th>
            <th colspan="{len(JUDGE_DIMS)}" style="background:#1a252f;">Calidad (juez)</th>
            <th rowspan="2">Calidad global</th>
            <th rowspan="2">Centinelas</th>
            <th colspan="3" style="background:#1a252f;">Latencia</th>
          </tr>
          <tr>{dim_headers}<th>media</th><th>mediana</th><th>p95</th></tr>
        </thead>
        <tbody>
          {overall_rows()}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Score por Categoría</h2>
    <div class="overflow-x">
      <table>
        <thead><tr><th>Modelo / Prompt</th>{cat_headers}</tr></thead>
        <tbody>{category_rows()}</tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Calidad global y centinelas</h2>
    <div class="chart-wrap"><img src="{img_overall}" alt="Calidad global"></div>
  </div>
  <div class="card">
    <h2>Score por dimensión</h2>
    <div class="chart-wrap"><img src="{img_dims}" alt="Score por dimensión"></div>
  </div>
  {dist_block}
  <div class="card">
    <h2>Score por categoría</h2>
    <div class="chart-wrap"><img src="{img_cats}" alt="Score por categoría"></div>
  </div>
  <div class="card">
    <h2>Latencia de generación</h2>
    <div class="chart-wrap"><img src="{img_lat}" alt="Latencia"></div>
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

    details_path = RESULTS_DIR / "end_to_end_details.json"
    summary_path = RESULTS_DIR / "end_to_end_summary.json"

    details_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


    log.info("Detalle JSON: %s", details_path)
    log.info("Resumen JSON: %s", summary_path)


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
        "--charts-only",
        action="store_true",
        help=(
            "Crea las gráficas y reportes a partir de los JSON existentes en "
            f"{DEFAULT_DETAILS_PATH} y {DEFAULT_SUMMARY_PATH}, sin volver a ejecutar el pipeline."
        ),
    )

    parser.add_argument(
        "qa_dataset",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Dataset JSON con preguntas, cypher_result y answer. "
            "Es obligatorio salvo cuando se usa --charts-only."
        ),
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
            "Omite Docling, chunking y extracción."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env()
    if not args.charts_only:
        if args.skip_chunking and not args.skip_docling and not args.qa_only:
            raise ValueError(
                "--skip-chunking requiere también --skip-docling, "
                "porque de lo contrario los chunks podrían no coincidir "
                "con los documentos de Docling."
            )
        if args.qa_dataset is None:
            raise ValueError(
                "Debe indicarse qa_dataset salvo cuando se usa --charts-only."
            )
        try:
            if args.qa_only:
                load_into_neo4j(DEFAULT_DATASET)

                log.info(
                    "Modo --qa-only: se omiten Docling, chunking y extracción"
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
        except Exception as exc:  # noqa: BLE001
            log.exception("PIPELINE FALLÓ: %s", exc)
            return 1
    else:
        summary, records = load_existing_results()

    results = build_generation_style_results(summary, records)
    html_path = RESULTS_DIR / "end_to_end_report.html"
    charts = generate_charts(results, RESULTS_DIR / "images")

    judge = summary.get("judge", {})
    generate_html_report(
        results=results,
        output_path=html_path,
        chart_paths=charts,
        judge_model=str(judge.get("model", "")),
        judged=bool(summary.get("n_judged", 0)),
    )
    
    log.info("Reporte HTML: %s", html_path)
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
    print(f"Reporte HTML: {RESULTS_DIR / 'end_to_end_report.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())