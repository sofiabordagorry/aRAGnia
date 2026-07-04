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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
PROMPTS_JSON_PATH = EVAL_DIR / "prompts_variants.json"
DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

SENTINEL_NOT_IN_SCHEMA = "La consulta solicitada está fuera del alcance del esquema actual del grafo."
SENTINEL_NO_INFO = "No se encontró ningún elemento que cumpla con los criterios de la consulta."

MODELS: List[Dict[str, str]] = [
    # {"display": "Qwen 2.5 3B", "backend": "ollama", "model": "qwen2.5:3b-instruct"},
    {"display": "Qwen 2.5 14B", "backend": "huggingface", "model": "Qwen/Qwen2.5-14B-Instruct"},
    {"display": "Qwen 2.5 Coder 7B", "backend": "huggingface", "model": "Qwen/Qwen2.5-Coder-7B-Instruct"},
    # {"display": "Text2Cypher Gemma 2 9B", "backend": "huggingface", "model": "neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1"},
    {"display": "Llama 3.1 8B", "backend": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
]

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
        "recall": max(1, min(5, int(data.get("recall", 1)))), 
        "precision": max(1, min(5, int(data.get("precision", 1)))),
        "justification": str(data.get("justification", ""))
    }

def norm_score(score_1_5: float) -> float: return (score_1_5 - 1.0) / 4.0
def mean(values: List[float]) -> float: return round(sum(values) / len(values), 4) if values else 0.0

def fill_placeholders(retriever: GraphRAGRetriever, template: str, query: str) -> str:
    """Rellena los marcadores de posición del prompt dinámico con datos reales de Neo4j."""
    schema_text = retriever._fetch_schema()
    fewshot_text = retriever._fetch_fewshot_examples(query)
    
    return template.replace(
        "{schema}", schema_text
    ).replace(
        "{fewshot}", fewshot_text
    ).replace(
        "{user_query}", query
    )

def evaluate_combo(
    spec: Dict[str, str], prompt_id: str, prompt_template: Callable, items: List[Dict[str, Any]], 
    api_key: Optional[str], judge_model: str, local_judge: Optional[str], client: Any, retriever: GraphRAGRetriever
) -> Dict[str, Any]:
    label = f"{spec['display']} / {prompt_id}"
    print(f"\n=== Evaluando: {label} ===", flush=True)

    retriever.cypher_llm_client = client
    # retriever.answer_llm_client = client
    retriever._build_cypher_generation_prompt = lambda q: fill_placeholders(retriever, prompt_template, q)
    candidate_subgraph = "Placeholder"

    original_classify = retriever._classify_query_intent
    def tracking_classify(q):
        intent = original_classify(q)
        retriever._last_intent = intent
        return intent
    retriever._classify_query_intent = tracking_classify
    
    records_res: List[Dict[str, Any]] = []
    
    for item in items:
        question = (item.get("pregunta") or item.get("question", "")).strip()
        if not question: continue
        gt_subgraph = item.get("retrieved_subgraph", "")
        qid = item.get("id", "?")
        sentinel = is_sentinel(gt_subgraph)
        cypher_query = ""

        logger.info(f"Procesando pregunta: [{qid}]")
        t0 = time.perf_counter()
        
        try:         
            result_obj, neo_records, cypher_query = retriever.generate_cypher_query_result(user_query=question)
            if result_obj.answer == SENTINEL_NOT_IN_SCHEMA:
                candidate_subgraph = SENTINEL_NOT_IN_SCHEMA
            elif neo_records == []:
                logger.warning(result_obj.answer)
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

        if getattr(retriever, "_last_intent", None) == "CHAT":
            scores = {"recall": 1, "precision": 1, "justification": "Penalizado automáticamente: El LLM de clasificación (Answer Model) evaluó erróneamente la consulta como CHAT en lugar de SEARCH."}
            rec["scores"] = scores
            rec["justification"] = scores["justification"]
            avg = mean([norm_score(scores[d]) for d in JUDGE_DIMS])
            print(f"  [{qid}] {latency:.1f}s  calidad={avg:.2f} (FAIL DIRECTO: Intención CHAT)", flush=True)

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
    ax.set_yticklabels(labels, fontsize=8)
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

    # 2) Score por dimensión → heatmap (combos × 3 dimensiones).
    dim_matrix = np.array([[r["dim_means"][d] for d in JUDGE_DIMS] for r in results])
    paths["dimensions"] = _heatmap(
        images_dir / "chart_dimensions.png", dim_matrix, labels,
        [DIM_LABELS[d] for d in JUDGE_DIMS], "Score por dimensión", fig_h,
    )

    # 3) Score por categoría → heatmap (combos × categorías).
    all_cats = sorted({c for r in results for c in r["by_category"]})
    cat_matrix = np.array([[r["by_category"].get(c, np.nan) for c in all_cats] for r in results])
    paths["categories"] = _heatmap(
        images_dir / "chart_categories.png", cat_matrix, labels, all_cats, "Score por categoría", fig_h,
    )

    # 4) Latencia (media + p95): barras horizontales agrupadas, ordenadas por media.
    by_lat = sorted(results, key=lambda r: r["latency"]["mean"])
    lat_labels = [r["label"] for r in by_lat]
    yl = np.arange(len(by_lat))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, fig_h))
    ax.barh(yl + 0.2, [r["latency"]["mean"] for r in by_lat], 0.4, color=PALETTE[3], label="Media")
    ax.barh(yl - 0.2, [r["latency"]["p95"] for r in by_lat], 0.4, color=PALETTE[1], label="p95")
    ax.set_yticks(yl)
    ax.set_yticklabels(lat_labels, fontsize=8)
    ax.set_xlabel("Segundos")
    ax.set_title("Latencia de generación por combinación (menor es mejor)")
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
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
                score_val = int(sc[d])
                score_val = max(1, min(5, score_val))
                counts[score_val] += 1
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
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% de scores del juez")
        ax.set_title("Distribución de scores del juez (1–5) por combinación")
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

    def overall_rows() -> str:
        rows = []
        for r in results:
            row = f"<tr><td class='llm-name'>{r['label']}</td>"
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
            row = f"<tr><td class='llm-name'>{r['label']}</td>"
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
            f"<strong>{best_q['label']}</strong> logra la mejor calidad global "
            f"({best_q['quality_overall']:.3f})."
        )

        by_model: Dict[str, List[float]] = {}
        for r in results:
            by_model.setdefault(r["display"], []).append(r["quality_overall"])
        best_model = max(by_model.items(), key=lambda kv: max(kv[1]))
        analysis_items.append(
            f"El modelo <strong>{best_model[0]}</strong> es el de mayor calidad pico "
            f"({max(best_model[1]):.3f})."
        )
        by_prompt: Dict[str, List[float]] = {}
        for r in results:
            by_prompt.setdefault(r["prompt"], []).append(r["quality_overall"])
        best_prompt = max(by_prompt.items(), key=lambda kv: mean(kv[1]))
        analysis_items.append(
            f"El prompt <strong>{best_prompt[0]}</strong> es el mejor en promedio "
            f"({mean(best_prompt[1]):.3f})."
        )
        best_sent = max(results, key=lambda r: r["sentinel_accuracy"])
        analysis_items.append(
            f"<strong>{best_sent['label']}</strong> maneja mejor los centinelas "
            f"(accuracy {best_sent['sentinel_accuracy']:.3f})."
        )
    fastest = min(results, key=lambda r: r["latency"]["mean"] or float("inf"))
    analysis_items.append(
        f"<strong>{fastest['label']}</strong> es el más rápido "
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

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Evaluación de Recuperación – GraphRAG</title>
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
    td.llm-name {{ text-align: left; font-weight: 600; white-space: nowrap; }}
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
    output_path.write_text(html, encoding="utf-8")

def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(filename)s:%(lineno)d - %(levelname)s - %(message)s', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Modelo de Anthropic.")
    parser.add_argument("--local-judge", type=str, default=None, help="Nombre del modelo Ollama para usar como juez local (ej. llama3.1).")
    parser.add_argument("--max-questions", type=int, default=None, help="Limita a N preguntas (smoke test).")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--prompts", type=Path, default=PROMPTS_JSON_PATH, help="Ruta al JSON de variantes de prompts.")
    parser.add_argument(
        "--replot",
        action="store_true",
        help="No corre generación ni juez: regenera solo las gráficas y el HTML "
        "desde retrieval_details.json ya existente.",
    )
    parser.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help="JSON opcional con la lista de modelos (sobrescribe MODELS).",
    )
    args = parser.parse_args()

    load_dotenv(BACKEND_DIR / ".env")

    if args.replot:
        details_path = RESULTS_DIR / "retrieval_details.json"
        if not details_path.exists():
            raise FileNotFoundError(f"No existe {details_path}; corré la evaluación primero.")
        results = json.loads(details_path.read_text(encoding="utf-8"))
        results.sort(key=lambda r: r["quality_overall"], reverse=True)
        chart_paths = generate_charts(results, RESULTS_DIR / "images")
        judged = any(rec.get("scores") for r in results for rec in r.get("records", []))
        generate_html_report(
            results, RESULTS_DIR / "retrieval_report.html", chart_paths, args.judge_model, judged=judged
        )
        print(f"Regeneradas {len(chart_paths)} gráficas + HTML desde {details_path.name} (sin re-evaluar).")
        return

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

    if not args.prompts.exists():
        raise FileNotFoundError(f"No se encontró el archivo de prompts en {args.prompts}")
    prompt_variants: Dict[str, str] = json.loads(args.prompts.read_text(encoding="utf-8"))
    print(f"[INFO] Se cargaron {len(prompt_variants)} variantes de prompts desde el JSON.", flush=True)
    
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
            for prompt_id, prompt_template in prompt_variants.items():
                try:
                    results.append(evaluate_combo(spec, prompt_id, prompt_template, items, api_key, args.judge_model, args.local_judge, client, retriever))
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

    print("\n[INFO] Generando gráficas y reporte HTML...", flush=True)
    chart_paths = generate_charts(results, RESULTS_DIR / "images")
    html_path = RESULTS_DIR / "retrieval_report.html"
    generate_html_report(results, html_path, chart_paths, args.judge_model, judged=bool(api_key or args.local_judge))
    print(f"[INFO] Reporte HTML listo en: {html_path}", flush=True)

    print("\n" + "=" * 78)
    print(f"{'Modelo / Prompt':<34} {'Calidad':>8} {'Centin.':>8} {'Lat.med':>9}")
    print("=" * 78)
    for r in sorted(results, key=lambda x: x["quality_overall"], reverse=True):
        print(f"{r['label']:<34} {r['quality_overall']:>8.3f} {r['sentinel_accuracy']:>8.3f} {r['latency']['mean']:>8.1f}s")
    print("=" * 78)

if __name__ == "__main__":
    main()