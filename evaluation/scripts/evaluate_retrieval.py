#!/usr/bin/env python3
"""evaluate_retrieval.py

Evalúa la RECUPERACIÓN (Cypher + Contexto) sobre el ground truth de validación.

Para cada modelo evaluado:
  1. Actúa como el motor de generación Cypher.
  2. Ejecuta la query en Neo4j y extrae el subgrafo (`retrieved_subgraph`).
  3. Compara el subgrafo recuperado contra el subgrafo de referencia (GT) usando
     un LLM-as-a-judge (API de Anthropic), evaluando Recall y Precision.
  4. Mide latencias y la tasa de acierto en casos centinela.

Salidas (en evaluation/results/retrieval/):
  - retrieval_comparison_summary.json
  - retrieval_details.json
  - images/*.png
  - retrieval_report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
import numpy as np
import requests
from dotenv import load_dotenv

SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS_DIR.parent
BACKEND_DIR = EVAL_DIR.parent / "backend"
sys.path.append(str(SCRIPTS_DIR))
sys.path.append(str(BACKEND_DIR))

logger = logging.getLogger(__name__)

# Importamos tu retriever y clientes LLM
from institutional_graphrag.retrieval.graph_retriever import GraphRAGRetriever
from institutional_graphrag.llm.llm_provider import HuggingFaceClient, OllamaClient

DEFAULT_DATASET = EVAL_DIR / "ground_truth" / "datasetQA_GT.json"
RESULTS_DIR = EVAL_DIR / "results" / "retrieval"

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

SENTINEL_NOT_IN_SCHEMA = "La consulta solicitada está fuera del alcance del esquema actual del grafo."
SENTINEL_NO_INFO = "No se encontró ningún elemento que cumpla con los criterios de la consulta."

MODELS: List[Dict[str, str]] = [
    {"display": "Qwen 2.5 3B", "backend": "ollama", "model": "qwen2.5:3b-instruct"},
    # {"display": "Qwen 2.5 3B", "backend": "huggingface", "model": "Qwen/Qwen2.5-3B-Instruct"},
    # {"display": "Qwen 2.5 Coder 7B", "backend": "huggingface", "model": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    # {"display": "Text2Cypher Gemma 2 9B", "backend": "huggingface", "model": "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"},
    # {"display": "Llama 3.1 8B", "backend": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
]

def prompt_baseline(retriever: GraphRAGRetriever, user_query: str) -> str:
    return retriever.__class__._build_cypher_generation_prompt(retriever, user_query)

def prompt_concise(retriever: GraphRAGRetriever, user_query: str) -> str:
    schema = retriever._fetch_schema()
    fewshot = retriever._fetch_fewshot_examples(user_query)
    return f"""Generate a precise Neo4j Cypher query to answer the user's question.

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
<QUERY>
"""

PROMPT_VARIANTS: Dict[str, Callable[[GraphRAGRetriever, str], str]] = {
    "baseline": prompt_baseline,
    "concise": prompt_concise,
}

JUDGE_DIMS = ["recall", "precision"]
DIM_LABELS = {"recall": "Recall", "precision": "Precision"}

PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f1c40f", "#9b59b6", "#34495e"]

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def is_sentinel(gt_subgraph: str) -> bool:
    return _norm(gt_subgraph) in {_norm(SENTINEL_NOT_IN_SCHEMA), _norm(SENTINEL_NO_INFO), ""}

def sentinel_matches(candidate: str, expected: str) -> bool:
    return _norm(candidate) == _norm(expected)

def build_client(spec: Dict[str, str]) -> Any:
    backend = spec["backend"].lower()
    if backend == "huggingface":
        if spec["model"] not in HuggingFaceClient._instances:
            HuggingFaceClient._instances[spec["model"]] = HuggingFaceClient(spec["model"])
        return HuggingFaceClient._instances[spec["model"]]
    if backend == "ollama": return OllamaClient(model=spec["model"])
    raise ValueError(f"Backend desconocido: {spec['backend']!r}")

def free_client(client: Any) -> None:
    if not isinstance(client, HuggingFaceClient): return
    import gc, torch
    HuggingFaceClient._instances.pop(getattr(client, "model_id", None), None)
    for attr in ("pipe", "model"):
        if hasattr(client, attr): delattr(client, attr)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

JUDGE_SYSTEM = (
    "Eres un evaluador experto de recuperación de información en sistemas GraphRAG. "
    "Compara el CONTEXTO RECUPERADO contra el CONTEXTO DE REFERENCIA (Ground Truth). "
    "Puntuá de 1 a 5 (1=muy malo, 5=excelente):\n"
    "- recall: ¿El contexto recuperado contiene toda la información clave y entidades de la referencia?\n"
    "- precision: ¿El contexto recuperado es conciso, evitando ruido o registros irrelevantes?\n"
    "Respondé SOLO con un objeto JSON válido: {\"recall\": int, \"precision\": int, \"justification\": str}."
)

def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match: raise ValueError(f"El juez no devolvió JSON: {text[:100]}...")
    return dict(json.loads(match.group(0)))

def judge_retrieval_anthropic(api_key: str, model: str, question: str, gt_subgraph: str, candidate_subgraph: str, retries: int = 3) -> Dict[str, Any]:
    user = f"PREGUNTA:\n{question}\n\nCONTEXTO DE REFERENCIA:\n{gt_subgraph}\n\nCONTEXTO RECUPERADO:\n{candidate_subgraph}\n\nDevolvé el JSON con los scores."
    payload = {"model": model, "max_tokens": 512, "system": JUDGE_SYSTEM, "messages": [{"role": "user", "content": user}]}
    headers = {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION, "content-type": "application/json"}

    for attempt in range(retries):
        try:
            r = requests.post(ANTHROPIC_URL, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            data = _extract_json(r.json()["content"][0]["text"])
            return {"recall": int(data["recall"]), "precision": int(data["precision"]), "justification": str(data.get("justification", ""))}
        except Exception as exc:
            if attempt == retries - 1: raise RuntimeError(f"El juez falló: {exc}")
            time.sleep(2 * (attempt + 1))

def judge_retrieval_local(model: str, question: str, gt_subgraph: str, candidate_subgraph: str) -> Dict[str, Any]:
    """Usa un modelo de Ollama local como juez."""
    client = OllamaClient(model=model)
    user = f"PREGUNTA:\n{question}\n\nCONTEXTO DE REFERENCIA:\n{gt_subgraph}\n\nCONTEXTO RECUPERADO:\n{candidate_subgraph}\n\nDevolvé el JSON con los scores del 1 al 5."
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
    
    response = client.generate(messages=messages, temperature=0.0, max_tokens=512)
    data = _extract_json(response)
    return {
        "recall": int(data.get("recall", 1)), 
        "precision": int(data.get("precision", 1)), 
        "justification": str(data.get("justification", ""))
    }

def norm_score(score_1_5: float) -> float: return (score_1_5 - 1.0) / 4.0
def mean(values: List[float]) -> float: return round(sum(values) / len(values), 4) if values else 0.0

def evaluate_combo(
    spec: Dict[str, str], prompt_id: str, builder: Callable, items: List[Dict[str, Any]], 
    api_key: Optional[str], judge_model: str, local_judge: Optional[str], client: Any, retriever: GraphRAGRetriever
) -> Dict[str, Any]:
    label = f"{spec['display']} / {prompt_id}"
    print(f"\n=== Evaluando: {label} ===", flush=True)

    retriever.cypher_llm_client = client 
    retriever._build_cypher_generation_prompt = lambda q: builder(retriever, q)
    
    records_res: List[Dict[str, Any]] = []
    
    for item in items:
        question = (item.get("pregunta") or item.get("question", "")).strip()
        if not question: continue
        gt_subgraph = item.get("retrieved_subgraph", "")
        qid = item.get("id", "?")
        sentinel = is_sentinel(gt_subgraph)
        cypher_query = ""

        logger.info(f"[{qid}] Procesando pregunta: '{question[:60]}...'")
        t0 = time.perf_counter()
        
        try:
            logger.info(f"[{qid}] Invocando retriever.generate_cypher_query_result...")            
            result_obj, neo_records, cypher_query = retriever.generate_cypher_query_result(user_query=question)
            logger.info(f"[{qid}] QUERY RETORNADA AL SCRIPT: {cypher_query}")
            logger.info(f"[{qid}] Registros obtenidos de Neo4j: {len(neo_records) if neo_records else 0}")
            if result_obj.answer == SENTINEL_NOT_IN_SCHEMA:
                candidate_subgraph = SENTINEL_NOT_IN_SCHEMA
            else:
                chunks, evidence_entities, chunk_to_entities = (
                    retriever.extract_chunks_and_entities_from_results(neo_records)
                )
                if not chunks:
                    logger.warning("No se encontraron chunks en los resultados del grafo")
                    if neo_records:
                        logger.info("Sin chunks pero con resultados del grafo, procesando...")
                        candidate_subgraph = retriever._build_aggregation_context(neo_records) if neo_records else SENTINEL_NO_INFO
                    else:
                        candidate_subgraph = SENTINEL_NO_INFO
                else:
                    if evidence_entities:
                        candidate_subgraph = retriever.build_entity_context(evidence_entities, chunk_to_entities)
                    else:
                        candidate_subgraph = retriever._build_aggregation_context(neo_records)    
        except Exception as e:
            logger.warning(f"[{qid}] Excepción no controlada: {e}")
            candidate_subgraph = SENTINEL_NO_INFO
                
        latency = time.perf_counter() - t0
        logger.info(f"[{qid}] Generación y recuperación finalizada en {latency:.2f}s")

        rec = {
            "id": qid, "category": item.get("categoria", item.get("category", "")),
            "question": question, "gt_subgraph": gt_subgraph, "generated query": cypher_query, "candidate_subgraph": candidate_subgraph,
            "latency_s": latency, "is_sentinel": sentinel, "scores": None, "sentinel_correct": None, "justification": "",
        }

        if sentinel:
            rec["sentinel_correct"] = sentinel_matches(candidate_subgraph, gt_subgraph)
            print(f"  [{qid}] centinela -> {'OK' if rec['sentinel_correct'] else 'FAIL'} ({latency:.1f}s)", flush=True)
        elif local_judge:
            try:
                logger.info(f"[{qid}] Enviando candidato al juez local ({local_judge})...")
                scores = judge_retrieval_local(local_judge, question, gt_subgraph, candidate_subgraph)
                rec["scores"] = scores
                rec["justification"] = scores["justification"]
                avg = mean([norm_score(scores[d]) for d in JUDGE_DIMS])
                print(f"  [{qid}] {latency:.1f}s  calidad={avg:.2f} (Juez Local)", flush=True)
            except Exception as e:
                print(f"  [{qid}] Falló el juez local: {e}", flush=True)
        elif api_key:
            scores = judge_retrieval_anthropic(api_key, judge_model, question, gt_subgraph, candidate_subgraph)
            rec["scores"] = scores
            rec["justification"] = scores["justification"]
            avg = mean([norm_score(scores[d]) for d in JUDGE_DIMS])
            print(f"  [{qid}] {latency:.1f}s  calidad={avg:.2f}", flush=True)
        else:
            print(f"  [{qid}] {latency:.1f}s  (sin juez)", flush=True)

        records_res.append(rec)
    
    return aggregate_combo(spec, prompt_id, records_res)

def aggregate_combo(
    spec: Dict[str, str], prompt_id: str, records: List[Dict[str, Any]]
) -> Dict[str, Any]:
    judged = [r for r in records if r["scores"] is not None]
    sentinels = [r for r in records if r["is_sentinel"]]
    latencies = [r["latency_s"] for r in records if r["latency_s"] is not None]

    dim_means = {
        d: mean([norm_score(r["scores"][d]) for r in judged]) for d in JUDGE_DIMS
    }
    quality_overall = mean([norm_score(mean([r["scores"][d] for d in JUDGE_DIMS])) for r in judged])
    sentinel_acc = mean([1.0 if r["sentinel_correct"] else 0.0 for r in sentinels])

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

    lat_stats = {
        "mean": round(statistics.mean(latencies), 2) if latencies else 0.0,
        "median": round(statistics.median(latencies), 2) if latencies else 0.0,
        "p95": round(_percentile(latencies, 95), 2) if latencies else 0.0,
        "n": len(latencies),
    }

    return {
        "display": spec["display"],
        "backend": spec["backend"],
        "model": spec["model"],
        "prompt": prompt_id,
        "label": f"{spec['display']} / {prompt_id}",
        "n_judged": len(judged),
        "n_sentinel": len(sentinels),
        "dim_means": dim_means,
        "quality_overall": quality_overall,
        "sentinel_accuracy": sentinel_acc,
        "by_category": by_cat,
        "latency": lat_stats,
        "records": records,
    }


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values), pct))
    
# def _grouped_bar(ax: Any, labels: List[str], series: List[Dict[str, Any]], title: str, ylabel: str) -> None:
#     n_series = len(series)
#     bar_w = 0.8 / max(n_series, 1)
#     x = np.arange(len(labels))
#     for i, s in enumerate(series):
#         offset = (i - n_series / 2 + 0.5) * bar_w
#         ax.bar(x + offset, s["data"], bar_w, label=s["label"], color=PALETTE[i % len(PALETTE)])
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
#     ax.set_ylabel(ylabel)
#     ax.set_title(title)
#     ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0)
#     ax.spines[["top", "right"]].set_visible(False)


# def generate_charts(results: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Path]:
#     images_dir.mkdir(parents=True, exist_ok=True)
#     paths: Dict[str, Path] = {}
#     combo_labels = [r["label"] for r in results]

#     # 1) Calidad global 
#     fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.3), 4.5))
#     x = np.arange(len(results))
#     ax.bar(x - 0.2, [r["quality_overall"] for r in results], 0.4, label="Calidad global", color=PALETTE[0])
#     ax.bar(x + 0.2, [r["sentinel_accuracy"] for r in results], 0.4, label="Centinelas", color=PALETTE[2])
#     ax.set_xticks(x)
#     ax.set_xticklabels(combo_labels, rotation=25, ha="right", fontsize=8)
#     ax.set_ylim(0, 1.05)
#     ax.set_ylabel("Score (0-1)")
#     ax.set_title("Calidad global y manejo de centinelas")
#     ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0)
#     ax.spines[["top", "right"]].set_visible(False)
#     fig.tight_layout()
#     p = images_dir / "chart_overall.png"
#     fig.savefig(p, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     paths["overall"] = p

#     # 2) Score por dimensión.
#     series_dim = [
#         {"label": r["label"], "data": [r["dim_means"][d] for d in JUDGE_DIMS]} for r in results
#     ]
#     fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.4), 4.5))
#     _grouped_bar(ax, [DIM_LABELS[d] for d in JUDGE_DIMS], series_dim, "Score por dimensión", "Score (0-1)")
#     ax.set_ylim(0, 1.05)
#     fig.tight_layout()
#     p = images_dir / "chart_dimensions.png"
#     fig.savefig(p, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     paths["dimensions"] = p

#     # 3) Score por categoría.
#     all_cats = sorted({c for r in results for c in r["by_category"]})
#     series_cat = [
#         {"label": r["label"], "data": [r["by_category"].get(c, 0.0) for c in all_cats]} for r in results
#     ]
#     fig, ax = plt.subplots(figsize=(max(8, len(all_cats) * 1.4), 5))
#     _grouped_bar(ax, all_cats, series_cat, "Score por categoría", "Score (0-1)")
#     ax.set_ylim(0, 1.05)
#     fig.tight_layout()
#     p = images_dir / "chart_categories.png"
#     fig.savefig(p, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     paths["categories"] = p

#     # 4) Latencia (mean + p95) por combinación
#     fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.3), 4.5))
#     x = np.arange(len(results))
#     ax.bar(x - 0.2, [r["latency"]["mean"] for r in results], 0.4, label="Media", color=PALETTE[3])
#     ax.bar(x + 0.2, [r["latency"]["p95"] for r in results], 0.4, label="p95", color=PALETTE[1])
#     ax.set_xticks(x)
#     ax.set_xticklabels(combo_labels, rotation=25, ha="right", fontsize=8)
#     ax.set_ylabel("Segundos")
#     ax.set_title("Latencia de generación por combinación")
#     ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0)
#     ax.spines[["top", "right"]].set_visible(False)
#     fig.tight_layout()
#     p = images_dir / "chart_latency.png"
#     fig.savefig(p, dpi=150, bbox_inches="tight")
#     plt.close(fig)
#     paths["latency"] = p

#     # 5) Distribución de scores del juez (1-5) por combinación (barra apilada, % del total).
#     #    Revela la FORMA, no solo la media: dos modelos con media parecida pueden diferir
#     #    en cuántos fallos graves (score 1) tienen. Poolea las 3 dimensiones juzgadas.
#     score_colors = {5: "#27ae60", 4: "#7fc97f", 3: "#f39c12", 2: "#e67e22", 1: "#e74c3c"}
#     dist_pct: Dict[int, List[float]] = {s: [] for s in (5, 4, 3, 2, 1)}
#     any_scores = False
#     for r in results:
#         counts = {s: 0 for s in (1, 2, 3, 4, 5)}
#         total = 0
#         for rec in r["records"]:
#             sc = rec.get("scores")
#             if not sc:
#                 continue
#             for d in JUDGE_DIMS:
#                 counts[int(sc[d])] += 1
#                 total += 1
#         any_scores = any_scores or total > 0
#         for s in (5, 4, 3, 2, 1):
#             dist_pct[s].append(100.0 * counts[s] / total if total else 0.0)

#     if any_scores:
#         fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.3), 4.5))
#         x = np.arange(len(results))
#         bottom = np.zeros(len(results))
#         for s in (5, 4, 3, 2, 1):
#             vals = np.array(dist_pct[s])
#             ax.bar(x, vals, 0.6, bottom=bottom, label=str(s), color=score_colors[s])
#             bottom += vals
#         ax.set_xticks(x)
#         ax.set_xticklabels(combo_labels, rotation=25, ha="right", fontsize=8)
#         ax.set_ylim(0, 100)
#         ax.set_ylabel("% de scores")
#         ax.set_title("Distribución de scores del juez (1-5) por combinación")
#         ax.legend(title="Score", fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), framealpha=0.9, borderaxespad=0.0)
#         ax.spines[["top", "right"]].set_visible(False)
#         fig.tight_layout()
#         p = images_dir / "chart_score_distribution.png"
#         fig.savefig(p, dpi=150, bbox_inches="tight")
#         plt.close(fig)
#         paths["score_dist"] = p

#     return paths

# def _val_color(val: float) -> str:
#     if val >= 0.7:
#         return "#27ae60"
#     if val >= 0.4:
#         return "#e67e22"
#     return "#e74c3c"


# def _metric_cell(val: float) -> str:
#     color = _val_color(val)
#     return f'<td style="background:{color}18;color:{color};font-weight:600;">{val:.3f}</td>'


# def generate_html_report(
#     results: List[Dict[str, Any]],
#     output_path: Path,
#     chart_paths: Dict[str, Path],
#     judge_model: str,
#     judged: bool,
# ) -> None:
#     all_cats = sorted({c for r in results for c in r["by_category"]})

#     def overall_rows() -> str:
#         rows = []
#         for r in results:
#             row = f"<tr><td class='llm-name'>{r['label']}</td>"
#             for d in JUDGE_DIMS:
#                 row += _metric_cell(r["dim_means"][d])
#             row += _metric_cell(r["quality_overall"])
#             row += _metric_cell(r["sentinel_accuracy"])
#             lat = r["latency"]
#             row += f"<td>{lat['mean']:.1f}s</td><td>{lat['median']:.1f}s</td><td>{lat['p95']:.1f}s</td>"
#             row += "</tr>"
#             rows.append(row)
#         return "\n".join(rows)

#     def category_rows() -> str:
#         rows = []
#         for r in results:
#             row = f"<tr><td class='llm-name'>{r['label']}</td>"
#             for c in all_cats:
#                 row += _metric_cell(r["by_category"].get(c, 0.0))
#             row += "</tr>"
#             rows.append(row)
#         return "\n".join(rows)

#     cat_headers = "".join(f"<th>{c}</th>" for c in all_cats)

#     analysis_items = []
#     if judged:
#         best_q = max(results, key=lambda r: r["quality_overall"])
#         analysis_items.append(
#             f"<strong>{best_q['label']}</strong> logra la mejor calidad global "
#             f"({best_q['quality_overall']:.3f})."
#         )

#         by_model: Dict[str, List[float]] = {}
#         for r in results:
#             by_model.setdefault(r["display"], []).append(r["quality_overall"])
#         best_model = max(by_model.items(), key=lambda kv: max(kv[1]))
#         analysis_items.append(
#             f"El modelo <strong>{best_model[0]}</strong> es el de mayor calidad pico "
#             f"({max(best_model[1]):.3f})."
#         )
#         by_prompt: Dict[str, List[float]] = {}
#         for r in results:
#             by_prompt.setdefault(r["prompt"], []).append(r["quality_overall"])
#         best_prompt = max(by_prompt.items(), key=lambda kv: mean(kv[1]))
#         analysis_items.append(
#             f"El prompt <strong>{best_prompt[0]}</strong> es el mejor en promedio "
#             f"({mean(best_prompt[1]):.3f})."
#         )
#         best_sent = max(results, key=lambda r: r["sentinel_accuracy"])
#         analysis_items.append(
#             f"<strong>{best_sent['label']}</strong> maneja mejor los centinelas "
#             f"(accuracy {best_sent['sentinel_accuracy']:.3f})."
#         )
#     fastest = min(results, key=lambda r: r["latency"]["mean"] or float("inf"))
#     analysis_items.append(
#         f"<strong>{fastest['label']}</strong> es el más rápido "
#         f"(latencia media {fastest['latency']['mean']:.1f}s, p95 {fastest['latency']['p95']:.1f}s)."
#     )
#     analysis_html = "\n".join(f"<li>{item}</li>" for item in analysis_items)

#     img_overall = chart_paths["overall"].relative_to(output_path.parent)
#     img_dims = chart_paths["dimensions"].relative_to(output_path.parent)
#     img_cats = chart_paths["categories"].relative_to(output_path.parent)
#     img_lat = chart_paths["latency"].relative_to(output_path.parent)

#     dist_block = ""
#     if "score_dist" in chart_paths:
#         img_dist = chart_paths["score_dist"].relative_to(output_path.parent)
#         dist_block = (
#             '<div class="card">\n'
#             "    <h2>Distribución de scores del juez (1-5)</h2>\n"
#             f'    <div class="chart-wrap"><img src="{img_dist}" alt="Distribución de scores"></div>\n'
#             "  </div>"
#         )

#     dim_headers = "".join(f"<th>{DIM_LABELS[d]}</th>" for d in JUDGE_DIMS)

#     html = f"""<!DOCTYPE html>
# <html lang="es">
# <head>
#   <meta charset="UTF-8">
#   <title>Evaluación de Generación – GraphRAG</title>
#   <style>
#     *, *::before, *::after {{ box-sizing: border-box; }}
#     body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
#       margin: 0; padding: 2rem 3rem; background: #f0f2f5; color: #2c3e50; }}
#     h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
#     h2 {{ font-size: 1.2rem; color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 2rem; }}
#     p.subtitle {{ color: #666; margin-top: 0; }}
#     .card {{ background: white; border-radius: 10px; padding: 1.5rem;
#       box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 1rem 0; }}
#     .overflow-x {{ overflow-x: auto; }}
#     table {{ border-collapse: collapse; width: 100%; font-size: 0.85rem; }}
#     th {{ background: #2c3e50; color: white; padding: 7px 10px; text-align: center; white-space: nowrap; font-size: 0.78rem; }}
#     th:first-child {{ text-align: left; }}
#     td {{ padding: 7px 10px; text-align: center; border-bottom: 1px solid #f0f0f0; }}
#     td.llm-name {{ text-align: left; font-weight: 600; white-space: nowrap; }}
#     tr:last-child td {{ border-bottom: none; }}
#     tr:hover td {{ background: #fafbfc; }}
#     .chart-wrap {{ max-width: 900px; }}
#     .chart-wrap img {{ width: 100%; height: auto; border-radius: 6px; }}
#     ul.analysis {{ line-height: 1.9; }}
#   </style>
# </head>
# <body>
#   <h1>Evaluación de Generación – Comparación de LLMs y Prompts</h1>
#   <p class="subtitle">{len(results)} combinaciones modelo×prompt evaluadas sobre el GT de validación. Juez: <code>{judge_model}</code>. Scores normalizados 0-1.</p>

#   <div class="card">
#     <h2>Mini Análisis</h2>
#     <ul class="analysis">
#       {analysis_html}
#     </ul>
#   </div>

#   <div class="card">
#     <h2>Resumen Global</h2>
#     <div class="overflow-x">
#       <table>
#         <thead>
#           <tr>
#             <th rowspan="2">Modelo / Prompt</th>
#             <th colspan="{len(JUDGE_DIMS)}" style="background:#1a252f;">Calidad (juez)</th>
#             <th rowspan="2">Calidad global</th>
#             <th rowspan="2">Centinelas</th>
#             <th colspan="3" style="background:#1a252f;">Latencia</th>
#           </tr>
#           <tr>{dim_headers}<th>media</th><th>mediana</th><th>p95</th></tr>
#         </thead>
#         <tbody>
#           {overall_rows()}
#         </tbody>
#       </table>
#     </div>
#   </div>

#   <div class="card">
#     <h2>Score por Categoría</h2>
#     <div class="overflow-x">
#       <table>
#         <thead><tr><th>Modelo / Prompt</th>{cat_headers}</tr></thead>
#         <tbody>{category_rows()}</tbody>
#       </table>
#     </div>
#   </div>

#   <div class="card">
#     <h2>Calidad global y centinelas</h2>
#     <div class="chart-wrap"><img src="{img_overall}" alt="Calidad global"></div>
#   </div>
#   <div class="card">
#     <h2>Score por dimensión</h2>
#     <div class="chart-wrap"><img src="{img_dims}" alt="Score por dimensión"></div>
#   </div>
#   {dist_block}
#   <div class="card">
#     <h2>Score por categoría</h2>
#     <div class="chart-wrap"><img src="{img_cats}" alt="Score por categoría"></div>
#   </div>
#   <div class="card">
#     <h2>Latencia de generación</h2>
#     <div class="chart-wrap"><img src="{img_lat}" alt="Latencia"></div>
#   </div>
# </body>
# </html>"""

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     output_path.write_text(html, encoding="utf-8")

def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Modelo de Anthropic.")
    parser.add_argument("--local-judge", type=str, default=None, help="Nombre del modelo Ollama para usar como juez local (ej. llama3.1).")
    parser.add_argument("--max-questions", type=int, default=None, help="Limita a N preguntas (smoke test).")
    parser.add_argument("--no-judge", action="store_true")
    args = parser.parse_args()

    load_dotenv(BACKEND_DIR / ".env")

    original_backend = os.getenv("LLM_BACKEND")
    os.environ["LLM_BACKEND"] = "ollama"
    
    retriever = GraphRAGRetriever(
        neo4j_uri=f"bolt://{os.getenv('HOST', 'localhost')}:{os.getenv('NEO4J_BOLT_PORT', '7687')}",
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password")
    )

    if original_backend:
        os.environ["LLM_BACKEND"] = original_backend
    else:
        del os.environ["LLM_BACKEND"]

    items = json.loads(args.dataset.read_text(encoding="utf-8"))
    
    if args.max_questions is not None:
        items = items[:args.max_questions]
        print(f"[INFO] Smoke test activado: limitando a {args.max_questions} preguntas.", flush=True)

    api_key = os.getenv("ANTHROPIC_API_KEY") if not args.no_judge and not args.local_judge else None

    if args.local_judge:
        print(f"[INFO] Usando juez local: {args.local_judge}", flush=True)

    results = []
    for spec in MODELS:
        try:
            client = None
            client = build_client(spec)
            for prompt_id, builder in PROMPT_VARIANTS.items():
                try:
                    results.append(evaluate_combo(spec, prompt_id, builder, items, api_key, args.judge_model, args.local_judge, client, retriever))
                except Exception as exc:
                    print(f"[ERROR] Falló combinación {spec['display']} / {prompt_id}: {exc}")
        except Exception as exc:
            print(f"[ERROR] Modelo {spec['display']} falló al cargar: {exc}")
        finally:
            if client is not None:
                free_client(client)
            
    retriever.close()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval_details.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    (RESULTS_DIR / "retrieval_comparison_summary.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "records"} for r in results], ensure_ascii=False, indent=2))

    # print("\n[INFO] Generando gráficas y reporte HTML...", flush=True)
    # chart_paths = generate_charts(results, RESULTS_DIR / "images")
    # html_path = RESULTS_DIR / "retrieval_report.html"
    # generate_html_report(results, html_path, chart_paths, args.judge_model, judged=bool(api_key or args.local_judge))
    # print(f"[INFO] Reporte HTML listo en: {html_path}", flush=True)

    # print("\n" + "=" * 78)
    # print(f"{'Modelo / Prompt':<34} {'Calidad':>8} {'Centin.':>8} {'Lat.med':>9}")
    # print("=" * 78)
    # for r in sorted(results, key=lambda x: x["quality_overall"], reverse=True):
    #     print(f"{r['label']:<34} {r['quality_overall']:>8.3f} {r['sentinel_accuracy']:>8.3f} {r['latency']['mean']:>8.1f}s")
    print("=" * 78)

if __name__ == "__main__":
    main()