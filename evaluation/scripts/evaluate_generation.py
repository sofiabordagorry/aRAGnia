#!/usr/bin/env python3
"""evaluate_generation.py

Evalúa la GENERACIÓN de respuestas en lenguaje natural sobre el ground truth de
validación (`datasetQA_GT.json`). Responde dos preguntas del issue #239:

  - ¿Qué LLM / prompt genera mejores respuestas a partir del subgrafo recuperado?
  - ¿Cuánto tarda cada modelo (tiempos de espera)?

Para cada combinación de modelo × variante de prompt:
  1. Genera la respuesta a partir de `pregunta` + `retrieved_subgraph`
     (HuggingFace u Ollama, según el backend de cada modelo), midiendo la latencia.
  2. Juzga la respuesta contra el `answer` de referencia del GT usando un
     LLM-as-a-judge vía la API de Anthropic (factual / completitud / fidelidad).
  3. Los items centinela (fuera de alcance / sin info) se evalúan de forma
     determinística (exact-match), sin invocar al juez.

Salidas (en evaluation/results/generation/):
  - generation_comparison_summary.json  → métricas agregadas por modelo×prompt
  - generation_details.json             → detalle por pregunta (para inspección)
  - images/*.png                        → gráficas de calidad y latencia
  - generation_report.html              → reporte con tablas, colores y gráficas

Uso:
  export ANTHROPIC_API_KEY=...   # (o ponerlo en backend/.env)
  python evaluation/scripts/evaluate_generation.py
  python evaluation/scripts/evaluate_generation.py --max-questions 3   # smoke test
  python evaluation/scripts/evaluate_generation.py --no-judge          # solo latencias
"""

from __future__ import annotations

import argparse
import json
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
sys.path.append(str(SCRIPTS_DIR))

import generate_gt_answers as gen  # noqa: E402
from institutional_graphrag.llm.llm_provider import (  # noqa: E402
    HuggingFaceClient,
    OllamaClient,
)

DEFAULT_DATASET = EVAL_DIR / "ground_truth" / "datasetQA_GT.json"
RESULTS_DIR = EVAL_DIR / "results" / "generation"

DEFAULT_JUDGE_MODEL = "claude-opus-4-8"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"

MODELS: List[Dict[str, str]] = [
    {"display": "Qwen3 4B (2507)", "backend": "huggingface", "model": "Qwen/Qwen3-4B-Instruct-2507"},
    {"display": "Qwen2.5 7B", "backend": "huggingface", "model": "Qwen/Qwen2.5-7B-Instruct"},
    {"display": "Llama 3.1 8B", "backend": "huggingface", "model": "meta-llama/Llama-3.1-8B-Instruct"},
    {"display": "Falcon3 10B", "backend": "huggingface", "model": "tiiuae/Falcon3-10B-Instruct"},
    {"display": "Falcon3 10B", "backend": "huggingface", "model": "tiiuae/Falcon3-10B-Instruct"},
    {"display": "Gemma 3 12B", "backend": "huggingface", "model": "google/gemma-3-12b-it"},
    {"display": "Mistral Small 3.1 24B", "backend": "huggingface", "model": "mistralai/Mistral-Small-3.1-24B-Instruct-2503"},
]

# Variantes de prompt: id -> builder(pregunta, subgrafo) -> messages.
# "baseline" es el que esta en el sistema
def _prompt_concise(pregunta: str, subgrafo: str) -> List[Dict[str, str]]:
    """Variante más corta y estricta, sin la rama de conteo."""
    system = (
        "Eres un asistente académico. Respondé en ESPAÑOL usando SOLO los resultados "
        "del grafo. Listá TODOS los resultados, sin inventar ni interpretar. Sé conciso."
    )
    user = f"PREGUNTA: {pregunta}\n\nRESULTADOS DEL GRAFO:\n{subgrafo}\n\nRESPUESTA:"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


PROMPT_VARIANTS: Dict[str, Callable[[str, str], List[Dict[str, str]]]] = {
    "baseline": gen.build_messages,
    "concise": _prompt_concise,
}

# Dimensiones juzgadas por el LLM (escala 1-5).
JUDGE_DIMS = ["factual_correctness", "completeness", "faithfulness"]
DIM_LABELS = {
    "factual_correctness": "Correctitud",
    "completeness": "Completitud",
    "faithfulness": "Fidelidad",
}

PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def is_sentinel(item: Dict[str, Any]) -> bool:
    """Item cuya respuesta esperada es un mensaje centinela (fuera de alcance / sin info)."""
    subg = _norm(item.get("retrieved_subgraph", ""))
    ans = _norm(item.get("answer", ""))
    sentinels = {_norm(gen.OUT_OF_SCOPE), _norm(gen.NO_INFO), ""}
    return subg in sentinels or ans in sentinels


def sentinel_matches(candidate: str, expected: str) -> bool:
    return _norm(candidate) == _norm(expected)

def build_client(spec: Dict[str, str]) -> Any:
    backend = spec["backend"].lower()
    if backend == "huggingface":
        if spec["model"] not in HuggingFaceClient._instances:
            HuggingFaceClient._instances[spec["model"]] = HuggingFaceClient(spec["model"])
        return HuggingFaceClient._instances[spec["model"]]
    if backend == "ollama":
        return OllamaClient(model=spec["model"])
    raise ValueError(f"Backend desconocido: {spec['backend']!r}")


def generate_with_latency(
    client: Any,
    builder: Callable[[str, str], List[Dict[str, str]]],
    pregunta: str,
    subgrafo: str,
) -> tuple[str, Optional[float]]:
    """Genera la respuesta y devuelve (respuesta, latencia_segundos | None si centinela)."""
    subgrafo = (subgrafo or "").strip()
    if not subgrafo or subgrafo == gen.NO_INFO:
        return gen.NO_INFO, None
    if subgrafo == gen.OUT_OF_SCOPE:
        return gen.OUT_OF_SCOPE, None

    messages = builder(pregunta, subgrafo)
    t0 = time.perf_counter()
    answer = client.generate(messages=messages, temperature=0.1, max_tokens=2048)
    latency = time.perf_counter() - t0
    return answer.strip(), latency

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


def _extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"El juez no devolvió JSON: {text[:200]!r}")
    return dict(json.loads(match.group(0)))


def judge_answer(
    api_key: str,
    model: str,
    pregunta: str,
    subgrafo: str,
    gt_answer: str,
    candidate: str,
    retries: int = 3,
) -> Dict[str, Any]:
    """Llama a la API de Anthropic y devuelve los scores 1-5 + justificación."""
    user = (
        f"PREGUNTA:\n{pregunta}\n\n"
        f"RESULTADOS DEL GRAFO (fuente de verdad):\n{subgrafo}\n\n"
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
            r = requests.post(ANTHROPIC_URL, json=payload, headers=headers, timeout=120)
            r.raise_for_status()
            text = r.json()["content"][0]["text"]
            data = _extract_json(text)
            return {
                "factual_correctness": int(data["factual_correctness"]),
                "completeness": int(data["completeness"]),
                "faithfulness": int(data["faithfulness"]),
                "justification": str(data.get("justification", "")),
            }
        except Exception as exc:
            last_err = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"El juez falló tras {retries} intentos: {last_err}")

def norm_score(score_1_5: float) -> float:
    """Normaliza una puntuación 1-5 a 0-1."""
    return (score_1_5 - 1.0) / 4.0


def mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate_combo(
    spec: Dict[str, str],
    prompt_id: str,
    builder: Callable[[str, str], List[Dict[str, str]]],
    items: List[Dict[str, Any]],
    api_key: Optional[str],
    judge_model: str,
) -> Dict[str, Any]:
    combo_label = f"{spec['display']} / {prompt_id}"
    print(f"\n=== {combo_label} ===", flush=True)
    client = build_client(spec)

    records: List[Dict[str, Any]] = []
    for item in items:
        pregunta = (item.get("pregunta") or "").strip()
        if not pregunta:
            continue
        subgrafo = item.get("retrieved_subgraph", "")
        gt_answer = item.get("answer", "")
        qid = item.get("id", "?")
        sentinel = is_sentinel(item)

        candidate, latency = generate_with_latency(client, builder, pregunta, subgrafo)

        rec: Dict[str, Any] = {
            "id": qid,
            "categoria": item.get("categoria", ""),
            "pregunta": pregunta,
            "gt_answer": gt_answer,
            "candidate": candidate,
            "latency_s": latency,
            "is_sentinel": sentinel,
            "scores": None,
            "sentinel_correct": None,
            "justification": "",
        }

        if sentinel:
            rec["sentinel_correct"] = sentinel_matches(candidate, gt_answer)
            print(f"  [{qid}] centinela -> {'OK' if rec['sentinel_correct'] else 'FAIL'}", flush=True)
        elif api_key:
            scores = judge_answer(api_key, judge_model, pregunta, subgrafo, gt_answer, candidate)
            rec["scores"] = {d: scores[d] for d in JUDGE_DIMS}
            rec["justification"] = scores["justification"]
            avg = mean([norm_score(scores[d]) for d in JUDGE_DIMS])
            print(f"  [{qid}] {latency:.1f}s  calidad={avg:.2f}", flush=True)
        else:
            print(f"  [{qid}] {latency:.1f}s  (sin juez)", flush=True)

        records.append(rec)

    return aggregate_combo(spec, prompt_id, records)


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
    cats = sorted({r["categoria"] for r in records})
    for cat in cats:
        cat_judged = [r for r in judged if r["categoria"] == cat]
        cat_sent = [r for r in sentinels if r["categoria"] == cat]
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
    
def _grouped_bar(ax: Any, labels: List[str], series: List[Dict[str, Any]], title: str, ylabel: str) -> None:
    n_series = len(series)
    bar_w = 0.8 / max(n_series, 1)
    x = np.arange(len(labels))
    for i, s in enumerate(series):
        offset = (i - n_series / 2 + 0.5) * bar_w
        ax.bar(x + offset, s["data"], bar_w, label=s["label"], color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def generate_charts(results: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    combo_labels = [r["label"] for r in results]

    # 1) Calidad global 
    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.3), 4.5))
    x = np.arange(len(results))
    ax.bar(x - 0.2, [r["quality_overall"] for r in results], 0.4, label="Calidad global", color=PALETTE[0])
    ax.bar(x + 0.2, [r["sentinel_accuracy"] for r in results], 0.4, label="Centinelas", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Calidad global y manejo de centinelas")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = images_dir / "chart_overall.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["overall"] = p

    # 2) Score por dimensión.
    series_dim = [
        {"label": r["label"], "data": [r["dim_means"][d] for d in JUDGE_DIMS]} for r in results
    ]
    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.4), 4.5))
    _grouped_bar(ax, [DIM_LABELS[d] for d in JUDGE_DIMS], series_dim, "Score por dimensión", "Score (0-1)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = images_dir / "chart_dimensions.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["dimensions"] = p

    # 3) Score por categoría.
    all_cats = sorted({c for r in results for c in r["by_category"]})
    series_cat = [
        {"label": r["label"], "data": [r["by_category"].get(c, 0.0) for c in all_cats]} for r in results
    ]
    fig, ax = plt.subplots(figsize=(max(8, len(all_cats) * 1.4), 5))
    _grouped_bar(ax, all_cats, series_cat, "Score por categoría", "Score (0-1)")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = images_dir / "chart_categories.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["categories"] = p

    # 4) Latencia (mean + p95) por combinación
    fig, ax = plt.subplots(figsize=(max(7, len(results) * 1.3), 4.5))
    x = np.arange(len(results))
    ax.bar(x - 0.2, [r["latency"]["mean"] for r in results], 0.4, label="Media", color=PALETTE[3])
    ax.bar(x + 0.2, [r["latency"]["p95"] for r in results], 0.4, label="p95", color=PALETTE[1])
    ax.set_xticks(x)
    ax.set_xticklabels(combo_labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Segundos")
    ax.set_title("Latencia de generación por combinación")
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = images_dir / "chart_latency.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["latency"] = p

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

    dim_headers = "".join(f"<th>{DIM_LABELS[d]}</th>" for d in JUDGE_DIMS)

    html = f"""<!DOCTYPE html>
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

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET, help="JSON del GT de validación.")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Modelo de Anthropic para el juez.")
    parser.add_argument("--max-questions", type=int, default=None, help="Limita N preguntas (smoke test).")
    parser.add_argument("--no-judge", action="store_true", help="No juzga: solo genera y mide latencias.")
    parser.add_argument(
        "--models-file",
        type=Path,
        default=None,
        help="JSON opcional con la lista de modelos (sobrescribe MODELS).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(EVAL_DIR.parent / "backend" / ".env")

    if not args.dataset.exists():
        raise FileNotFoundError(f"No existe el dataset: {args.dataset}")

    items = gen.load_items(args.dataset)
    if args.max_questions is not None:
        items = items[: args.max_questions]

    models = json.loads(args.models_file.read_text(encoding="utf-8")) if args.models_file else MODELS

    api_key: Optional[str] = None
    if not args.no_judge:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            print("ADVERTENCIA: falta ANTHROPIC_API_KEY; corriendo en modo --no-judge.", flush=True)

    print(f"Dataset: {args.dataset}  ({len(items)} preguntas)")
    print(f"Combinaciones: {len(models)} modelos × {len(PROMPT_VARIANTS)} prompts")

    # Si una combinación falla (modelo gated sin token, OOM, error de API, etc.)
    # seguimos con las demás para no perder los resultados ya generados.
    results: List[Dict[str, Any]] = []
    failed: List[tuple[str, str]] = []
    for spec in models:
        for prompt_id, builder in PROMPT_VARIANTS.items():
            label = f"{spec['display']} / {prompt_id}"
            try:
                results.append(evaluate_combo(spec, prompt_id, builder, items, api_key, args.judge_model))
            except Exception as exc:  # noqa: BLE001 - aislar el fallo de una combinación
                print(f"[ERROR] {label} falló: {type(exc).__name__}: {exc}", flush=True)
                failed.append((label, f"{type(exc).__name__}: {exc}"))
                continue

    if not results:
        print("Ninguna combinación se evaluó con éxito.", flush=True)
        sys.exit(1)

    results.sort(key=lambda r: r["quality_overall"], reverse=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (RESULTS_DIR / "generation_details.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = [{k: v for k, v in r.items() if k != "records"} for r in results]
    (RESULTS_DIR / "generation_comparison_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    chart_paths = generate_charts(results, RESULTS_DIR / "images")
    html_path = RESULTS_DIR / "generation_report.html"
    generate_html_report(results, html_path, chart_paths, args.judge_model, judged=bool(api_key))

    print("\n" + "=" * 78)
    print(f"{'Modelo / Prompt':<34} {'Calidad':>8} {'Centin.':>8} {'Lat.med':>9} {'Lat.p95':>9}")
    print("=" * 78)
    for r in results:
        print(
            f"{r['label']:<34} {r['quality_overall']:>8.3f} {r['sentinel_accuracy']:>8.3f}"
            f" {r['latency']['mean']:>8.1f}s {r['latency']['p95']:>8.1f}s"
        )
    print("=" * 78)
    if failed:
        print(f"\nCombinaciones con error ({len(failed)}):")
        for label, err in failed:
            print(f"  - {label}: {err}")
    print(f"\nResumen JSON:  {RESULTS_DIR / 'generation_comparison_summary.json'}")
    print(f"Detalle JSON:  {RESULTS_DIR / 'generation_details.json'}")
    print(f"Reporte HTML:  {html_path}")


if __name__ == "__main__":
    main()
