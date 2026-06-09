#!/usr/bin/env python3
"""evaluate_topic_classification.py

Evalúa y ayuda a definir los parámetros de la clasificación de tópicos.

Los parámetros son:
  - chunking_threshold: score mínimo para que un tópico cuente en un chunk.
  - coverage_threshold: fracción mínima de chunks del proyecto en que aparece el tópico.
  - confidence_threshold: logit promedio mínimo del tópico en el proyecto.
  - chunk_size: tamaño del chunk (al cambiarlo se re-chunkea y se vuelve a correr BERT).

Cada parámetro se define como una lista de valores y el script prueba todas las
combinaciones, predice los tópicos por proyecto y los compara contra el GT
(ground_truth_kg.json) calculando Precision / Recall / F1.

Genera, en evaluation/results/topic_classification/, un resumen y un detalle en JSON, las
gráficas en images/ y un reporte HTML. Las corridas son acumulativas.

Configuración: editá las constantes de arriba (CHUNKING_THRESHOLDS, etc.) y corré el
script sin argumentos:

  python evaluation/scripts/evaluate_topic_classification.py

También se pueden pasar flags que sobrescriben esas listas, por ejemplo:

  python evaluation/scripts/evaluate_topic_classification.py --chunking-thresholds 0.04,0.06,0.08
  python evaluation/scripts/evaluate_topic_classification.py --chunk-sizes 256,384,512
"""

from __future__ import annotations

import argparse
import base64
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPTS_DIR.parent
REPO_ROOT = EVAL_DIR.parent

from institutional_graphrag.extraction.bert_extractor import (  # noqa: E402
    DEFAULT_CONFIDENCE_LOGIT_THRESHOLD,
    DEFAULT_COVERAGE_LOGIT_THRESHOLD,
    DEFAULT_THRESHOLD,
    BertTopicExtractor,
)

DEFAULT_GT_PATH = EVAL_DIR / "ground_truth" / "extraction" / "ground_truth_kg.json"
DEFAULT_CHUNKS_DIR = REPO_ROOT / "data" / "chunks"
DEFAULT_DOCLING_DIR = REPO_ROOT / "data" / "docling"
RESULTS_DIR = EVAL_DIR / "results" / "topic_classification"
CACHE_DIR = RESULTS_DIR / "cache"

# Score mínimo cacheado por chunk; los chunking_threshold a probar deben ser >= a esto.
RAW_SCORE_FLOOR = 0.001

PARAM_DIMENSIONS = (
    "chunking_threshold",
    "coverage_threshold",
    "confidence_threshold",
    "chunk_size",
)
PARAM_SHORT = {
    "chunking_threshold": "ck",
    "coverage_threshold": "cov",
    "confidence_threshold": "conf",
    "chunk_size": "size",
}

CHUNKING_THRESHOLDS: List[float] = [0.02, 0.04, 0.06, 0.08]  # umbral etapa de chunking (score por chunk)
COVERAGE_THRESHOLDS: List[float] = [DEFAULT_COVERAGE_LOGIT_THRESHOLD]      # umbral final: fracción de chunks (default 0.5)
CONFIDENCE_THRESHOLDS: List[float] = [DEFAULT_CONFIDENCE_LOGIT_THRESHOLD]  # umbral final: logit promedio (default 8)
CHUNK_SIZES: List[Optional[int]] = [None]   # tamaño de chunk; None = usar los chunks ya generados en data/chunks

# Opciones de corrida:
MAX_PROJECTS: Optional[int] = None  # limitar numero de proyectos (None = todos)
FRESH = False        # True = ignorar corridas previas (no acumular en el reporte)
USE_CACHE = True     # cachear predicciones BERT entre corridas (acelera los barridos)
DEVICE: Optional[str] = None  # "cuda" / "cpu" (None = autodetecta)

def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def normalize_topic_key(name: str) -> str:
    """Clave de comparación de tópicos: nombre en minúsculas, sin acentos (salvo ñ)."""
    return BertTopicExtractor._normalize_id(name)

def build_ground_truth(gt_data: dict) -> Tuple[Dict[str, List[str]], Dict[str, set]]:
    """Devuelve (docs_by_project, gt_topics_by_project).

    - docs_by_project: project_id -> lista de base_name de documentos (sin tablas).
    - gt_topics_by_project: project_id -> set de claves de tópico esperadas.
    """
    entities = gt_data.get("entities", [])
    relationships = gt_data.get("relationships", [])

    # id de Documento -> base_name (descartando tablas)
    doc_base_name: Dict[str, str] = {}
    for e in entities:
        if e.get("label") != "Documento":
            continue
        value = e.get("value", {}) or {}
        if value.get("type") == "tabla":
            continue
        base_name = value.get("base_name")
        if base_name and not str(base_name).endswith("_table"):
            doc_base_name[e["id"]] = base_name

    topic_key_by_id: Dict[str, str] = {}
    for e in entities:
        if e.get("label") == "Topico":
            val = e.get("value")
            if isinstance(val, str) and val:
                topic_key_by_id[e["id"]] = normalize_topic_key(val)

    docs_by_project: Dict[str, List[str]] = {}
    gt_topics_by_project: Dict[str, set] = {}

    for r in relationships:
        rtype = r.get("type")
        if rtype == "ES_DESCRITO_POR":
            base_name = doc_base_name.get(r.get("target_id"))
            if base_name:
                docs_by_project.setdefault(r["source_id"], [])
                if base_name not in docs_by_project[r["source_id"]]:
                    docs_by_project[r["source_id"]].append(base_name)
        elif rtype == "TIENE_TOPICO":
            target = r.get("target_id")

            key = topic_key_by_id.get(target)
            if key is None and target:
                key = normalize_topic_key(
                    str(target).removesuffix("_topic").replace("_", " ")
                )
            if key:
                gt_topics_by_project.setdefault(r["source_id"], set()).add(key)

    return docs_by_project, gt_topics_by_project

def _load_chunks(chunks_dir: Path, base_name: str) -> Optional[List[dict]]:
    path = chunks_dir / f"{base_name}_chunks.json"
    if not path.exists():
        return None
    payload = load_json(path)
    chunks = payload.get("chunks", [])
    return chunks if isinstance(chunks, list) else []


def _rechunk_project(
    docling_dir: Path,
    base_names: List[str],
    chunk_size: int,
) -> Dict[str, List[dict]]:
    """Re-chunkea los documentos del proyecto desde data/docling con un max_tokens dado."""
    from docling.chunking import HybridChunker
    from docling_core.types.doc import DoclingDocument

    from institutional_graphrag.config import EMBED_MODEL_ID
    from institutional_graphrag.ingest.chunker import chunk_document

    chunker = HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=chunk_size)
    out: Dict[str, List[dict]] = {}
    for base_name in base_names:
        doc_path = docling_dir / f"{base_name}.json"
        if not doc_path.exists():
            out[base_name] = []
            continue
        doc = DoclingDocument.model_validate(load_json(doc_path))
        out[base_name] = chunk_document(doc=doc, chunker=chunker)
    return out


def compute_raw_predictions(
    docs_by_project: Dict[str, List[str]],
    chunks_dir: Path,
    docling_dir: Path,
    chunk_size: Optional[int],
    project_title_by_id: Dict[str, str],
    device: Optional[str],
    max_projects: Optional[int],
) -> Dict[str, Any]:
    """Corre BERT una vez por proyecto y devuelve, por chunk, todos los tópicos con
    score >= RAW_SCORE_FLOOR. ``chunk_size=None`` usa los chunks ya existentes en disco;
    un valor entero re-chunkea desde docling con ese max_tokens.
    """
    extractor = BertTopicExtractor(
        threshold=RAW_SCORE_FLOOR,
        confidence_logit_threshold=None,
        coverage_logit_threshold=None,
        device=device,
    )

    projects_out: Dict[str, Any] = {}
    processed = 0
    for project_id, base_names in docs_by_project.items():
        if max_projects is not None and processed >= max_projects:
            break

        if chunk_size is None:
            chunks_by_doc = {bn: _load_chunks(chunks_dir, bn) for bn in base_names}
        else:
            chunks_by_doc = _rechunk_project(docling_dir, base_names, chunk_size)

        available = [bn for bn, c in chunks_by_doc.items() if c]
        if not available:
            print(f"  [skip] {project_id}: sin chunks disponibles")
            continue

        title = project_title_by_id.get(project_id, "")
        total_chunks = 0
        chunk_preds: List[List[list]] = []

        for base_name in base_names:
            chunks = chunks_by_doc.get(base_name)
            if not chunks:
                continue
            total_chunks += len(chunks)  # se cuenta igual que en el pipeline
            for chunk in chunks:
                text = chunk.get("text", "")
                if not text or not text.strip():
                    continue
                result = extractor.extract_topics_from_chunk(
                    text, chunk.get("chunk_id", ""), title
                )
                preds = [
                    [m.topic, float(m.evidence.replace("score=", "")), float(m.logit)]
                    for m in result.topics
                ]
                if preds:
                    chunk_preds.append(preds)

        projects_out[project_id] = {
            "total_chunks": total_chunks,
            "chunks": chunk_preds,
        }
        processed += 1
        print(f"  [bert] {project_id}: {total_chunks} chunks, {len(chunk_preds)} con tópicos")

    return {"chunk_size": chunk_size, "projects": projects_out}


def raw_cache_path(chunk_size: Optional[int]) -> Path:
    key = "default" if chunk_size is None else f"size{chunk_size}"
    return CACHE_DIR / f"raw_preds_{key}.json"


def get_raw_predictions(
    docs_by_project: Dict[str, List[str]],
    chunks_dir: Path,
    docling_dir: Path,
    chunk_size: Optional[int],
    project_title_by_id: Dict[str, str],
    device: Optional[str],
    max_projects: Optional[int],
    use_cache: bool,
) -> Dict[str, Any]:
    """Devuelve las predicciones crudas, usando cache en disco cuando esté disponible."""
    cache_path = raw_cache_path(chunk_size)
    if use_cache and cache_path.exists():
        print(f"  [cache] usando predicciones crudas de {cache_path.name}")
        return load_json(cache_path)

    label = "default (chunks en disco)" if chunk_size is None else f"chunk_size={chunk_size}"
    print(f"Corriendo BERT sobre los proyectos [{label}]...")
    raw = compute_raw_predictions(
        docs_by_project,
        chunks_dir,
        docling_dir,
        chunk_size,
        project_title_by_id,
        device,
        max_projects,
    )
    save_json(raw, cache_path)
    print(f"  [cache] predicciones crudas guardadas en {cache_path.name}")
    return raw

def aggregate_predicted_topics(
    raw: Dict[str, Any],
    chunking_threshold: float,
    coverage_threshold: float,
    confidence_threshold: float,
    en_to_es: Dict[str, str],
) -> Dict[str, set]:
    """Reproduce la lógica del pipeline (extract + aggregate_topics_for_project) a partir
    de las predicciones crudas, devolviendo project_id -> set de claves de tópico."""
    predicted: Dict[str, set] = {}

    for project_id, pdata in raw["projects"].items():
        total_chunks = pdata["total_chunks"]
        if total_chunks <= 0:
            predicted[project_id] = set()
            continue

        topic_count: Dict[str, int] = {}
        topic_sum_logit: Dict[str, float] = {}
        for chunk_preds in pdata["chunks"]:
            for topic_en, score, logit in chunk_preds:
                if score < chunking_threshold:
                    continue
                topic_count[topic_en] = topic_count.get(topic_en, 0) + 1
                topic_sum_logit[topic_en] = topic_sum_logit.get(topic_en, 0.0) + logit

        keys: set = set()
        for topic_en, count in topic_count.items():
            coverage = count / total_chunks
            confidence = topic_sum_logit[topic_en] / count
            if coverage >= coverage_threshold and confidence >= confidence_threshold:
                es_topic = en_to_es.get(topic_en, topic_en)
                keys.add(normalize_topic_key(es_topic))
        predicted[project_id] = keys

    return predicted


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def score_config(
    predicted: Dict[str, set],
    gt_topics_by_project: Dict[str, set],
) -> Dict[str, Any]:
    """Compara predicho vs GT. Macro-promedia por proyecto y micro-agrega global."""
    per_project: Dict[str, Any] = {}
    p_vals, r_vals, f1_vals = [], [], []
    tp_tot = fp_tot = fn_tot = 0

    # Solo proyectos con GT de tópicos y con predicción disponible.
    project_ids = [p for p in predicted if p in gt_topics_by_project]

    for project_id in project_ids:
        pred = predicted.get(project_id, set())
        gold = gt_topics_by_project.get(project_id, set())
        tp = len(pred & gold)
        fp = len(pred - gold)
        fn = len(gold - pred)
        m = prf(tp, fp, fn)
        m["predicted"] = sorted(pred)
        m["expected"] = sorted(gold)
        per_project[project_id] = m
        p_vals.append(m["precision"])
        r_vals.append(m["recall"])
        f1_vals.append(m["f1"])
        tp_tot += tp
        fp_tot += fp
        fn_tot += fn

    n = len(project_ids)
    macro = {
        "precision_macro": round(sum(p_vals) / n, 4) if n else 0.0,
        "recall_macro": round(sum(r_vals) / n, 4) if n else 0.0,
        "f1_macro": round(sum(f1_vals) / n, 4) if n else 0.0,
    }
    micro = prf(tp_tot, fp_tot, fn_tot)
    return {
        "n_projects": n,
        "macro": macro,
        "micro": micro,
        "per_project": per_project,
    }


def build_param_grid(
    chunking: List[float],
    coverage: List[float],
    confidence: List[float],
    sizes: List[Optional[int]],
) -> List[Dict[str, Any]]:
    """Producto cartesiano de las listas -> lista de combinaciones de parámetros."""
    combos: List[Dict[str, Any]] = []
    for ck, cov, conf, size in itertools.product(chunking, coverage, confidence, sizes):
        combos.append(
            {
                "chunking_threshold": ck,
                "coverage_threshold": cov,
                "confidence_threshold": conf,
                "chunk_size": size,
            }
        )
    return combos


def varying_dims(combos: List[Dict[str, Any]]) -> List[str]:
    """Dimensiones cuyo valor cambia entre combinaciones (las que se están comparando)."""
    return [d for d in PARAM_DIMENSIONS if len({c["params"][d] for c in combos}) > 1]


def sweep_description(varying: List[str]) -> str:
    if not varying:
        return "configuración única"
    return " × ".join(varying)


def _fmt_value(dim: str, value: Any) -> Any:
    if dim == "chunk_size":
        return "def" if value is None else value
    return value


def config_signature(params: Dict[str, Any]) -> str:
    return (
        f"ck{params['chunking_threshold']}_cov{params['coverage_threshold']}"
        f"_conf{params['confidence_threshold']}_size{params['chunk_size']}"
    )


def config_label(params: Dict[str, Any], varying: List[str]) -> str:
    """Etiqueta corta: muestra solo los parámetros que varían entre combinaciones.

    Si no varía ninguno (una sola combinación) muestra los cuatro valores."""
    dims = varying if varying else list(PARAM_DIMENSIONS)
    return " ".join(f"{PARAM_SHORT[d]}={_fmt_value(d, params[d])}" for d in dims)


PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#34495e"]


def generate_charts(configs: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Path]:
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    labels = [c["label"] for c in configs]
    x = np.arange(len(configs))

    # P / R / F1 macro por configuración
    fig, ax = plt.subplots(figsize=(max(6, len(configs) * 1.2), 4.5))
    for i, key in enumerate(("precision_macro", "recall_macro", "f1_macro")):
        ax.plot(
            x,
            [c["metrics"]["macro"][key] for c in configs],
            marker="o",
            color=PALETTE[i],
            label=key.replace("_macro", "").capitalize(),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score (macro)")
    ax.set_title("Precision / Recall / F1 macro por configuración")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = images_dir / "chart_metrics_vs_param.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["metrics"] = p

    # F1 por proyecto (una barra por configuración)
    all_projects = sorted(
        {pid for c in configs for pid in c["metrics"]["per_project"]}
    )
    if all_projects:
        n_series = len(configs)
        bar_w = 0.8 / max(n_series, 1)
        xp = np.arange(len(all_projects))
        fig, ax = plt.subplots(figsize=(max(7, len(all_projects) * 1.3), 4.5))
        for i, c in enumerate(configs):
            offset = (i - n_series / 2 + 0.5) * bar_w
            data = [
                c["metrics"]["per_project"].get(pid, {}).get("f1", 0.0)
                for pid in all_projects
            ]
            ax.bar(xp + offset, data, bar_w, label=c["label"], color=PALETTE[i % len(PALETTE)])
        ax.set_xticks(xp)
        ax.set_xticklabels(all_projects, rotation=25, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("F1")
        ax.set_title("F1 por proyecto y configuración")
        ax.legend(fontsize=7, ncol=2)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        p = images_dir / "chart_f1_by_project.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["by_project"] = p

    return paths


REPORT_CSS = """
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0; padding: 2rem 3rem; background: #f0f2f5; color: #2c3e50; }
    h1 { font-size: 1.7rem; margin-bottom: 0.25rem; }
    h2 { font-size: 1.15rem; color: #34495e; border-left: 4px solid #3498db;
      padding-left: 10px; margin-top: 1.5rem; }
    p.subtitle { color: #666; margin-top: 0; }
    .card { background: white; border-radius: 10px; padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 1rem 0; }
    .overflow-x { overflow-x: auto; }
    table { border-collapse: collapse; width: 100%; font-size: 0.82rem; }
    th { background: #2c3e50; color: white; padding: 7px 10px; text-align: center;
      white-space: nowrap; font-size: 0.76rem; }
    th:first-child { text-align: left; }
    td { padding: 7px 10px; text-align: center; border-bottom: 1px solid #f0f0f0; }
    td.cfg { text-align: left; font-weight: 600; white-space: nowrap; }
    tr:hover td { background: #fafbfc; }
    .chart-wrap { max-width: 900px; }
    .chart-wrap img { width: 100%; height: auto; border-radius: 6px; }
    ul.analysis { line-height: 1.8; }
"""


def _chart_src(path: Path, embed: bool, rel_to: Optional[Path]) -> str:
    """Devuelve el src de una imagen: data-URI base64 (embed) o ruta relativa (archivo)."""
    if embed:
        b64 = base64.b64encode(path.read_bytes()).decode()
        return f"data:image/png;base64,{b64}"
    return str(path.relative_to(rel_to)) if rel_to else str(path)


def render_report_cards(
    summary: Dict[str, Any],
    chart_paths: Dict[str, Path],
    embed_images: bool = False,
    rel_to: Optional[Path] = None,
) -> str:
    """Genera el cuerpo del reporte (h1 + subtítulo + cards), reutilizable tanto por el
    archivo HTML como por el servidor interactivo."""
    configs = summary["configs"]

    def val_color(v: float) -> str:
        if v >= 0.7:
            return "#27ae60"
        if v >= 0.4:
            return "#e67e22"
        return "#e74c3c"

    def mcell(v: float) -> str:
        c = val_color(v)
        return f'<td style="background:{c}18;color:{c};font-weight:600;">{v:.3f}</td>'

    def ccell(v: int) -> str:
        return f"<td>{v}</td>"

    rows = []
    for c in configs:
        p = c["params"]
        macro = c["metrics"]["macro"]
        micro = c["metrics"]["micro"]
        row = (
            f"<tr><td class='cfg'>{c['label']}</td>"
            f"<td>{p['chunking_threshold']}</td>"
            f"<td>{p['coverage_threshold']}</td>"
            f"<td>{p['confidence_threshold']}</td>"
            f"<td>{p['chunk_size'] if p['chunk_size'] is not None else 'default'}</td>"
            f"<td>{c['metrics']['n_projects']}</td>"
            + mcell(macro["precision_macro"])
            + mcell(macro["recall_macro"])
            + mcell(macro["f1_macro"])
            + mcell(micro["precision"])
            + mcell(micro["recall"])
            + mcell(micro["f1"])
            + ccell(micro["tp"])
            + ccell(micro["fp"])
            + ccell(micro["fn"])
            + "</tr>"
        )
        rows.append(row)
    table_rows = "\n".join(rows)

    # Tabla por proyecto (F1 de cada configuración).
    all_projects = sorted({pid for c in configs for pid in c["metrics"]["per_project"]})
    proj_header = "".join(f"<th>{c['label']}</th>" for c in configs)
    proj_rows = []
    for pid in all_projects:
        cells = "".join(
            mcell(c["metrics"]["per_project"].get(pid, {}).get("f1", 0.0)) for c in configs
        )
        proj_rows.append(f"<tr><td class='cfg'>{pid}</td>{cells}</tr>")
    proj_table = "\n".join(proj_rows)

    best = max(configs, key=lambda c: c["metrics"]["macro"]["f1_macro"])
    fixed = summary.get("fixed_params", {})
    fixed_str = (
        ", ".join(f"{PARAM_SHORT[d]}={_fmt_value(d, v)}" for d, v in fixed.items())
        if fixed
        else "(ninguno)"
    )
    analysis = [
        f"La mejor configuración por F1 macro es <strong>{best['label']}</strong> "
        f"(F1={best['metrics']['macro']['f1_macro']:.3f}, "
        f"P={best['metrics']['macro']['precision_macro']:.3f}, "
        f"R={best['metrics']['macro']['recall_macro']:.3f}).",
        f"Parámetros que varían: <strong>{summary['sweep']}</strong> · "
        f"{len(configs)} combinación(es).",
        f"Parámetros fijos: <strong>{fixed_str}</strong>.",
        "El GT de tópicos está <strong>pendiente de actualización con los valores de los "
        "profes</strong>: las métricas son provisionales y sirven para comparar parámetros "
        "entre sí, no como score absoluto.",
    ]
    analysis_html = "\n".join(f"<li>{a}</li>" for a in analysis)

    img_metrics_block = (
        f'<div class="card"><h2>P / R / F1 macro por configuración</h2>'
        f'<div class="chart-wrap"><img src="{_chart_src(chart_paths["metrics"], embed_images, rel_to)}" alt="metrics"></div></div>'
        if "metrics" in chart_paths
        else ""
    )
    img_proj_block = (
        f'<div class="card"><h2>F1 por proyecto</h2>'
        f'<div class="chart-wrap"><img src="{_chart_src(chart_paths["by_project"], embed_images, rel_to)}" alt="f1 by project"></div></div>'
        if "by_project" in chart_paths
        else ""
    )

    return f"""
  <h1>Parámetros de clasificación de tópicos</h1>
  <p class="subtitle">Variando: <strong>{summary['sweep']}</strong> ·
    {len(configs)} combinación(es) · GT: {Path(summary['gt_file']).name} ·
    generado {summary['generated_at']}</p>

  <div class="card">
    <h2>Análisis</h2>
    <ul class="analysis">
      {analysis_html}
    </ul>
  </div>

  <div class="card">
    <h2>Resumen por configuración</h2>
    <div class="overflow-x">
      <table>
        <thead>
          <tr>
            <th rowspan="2">Config</th>
            <th colspan="4" style="background:#1a252f;">Parámetros</th>
            <th rowspan="2">Proy.</th>
            <th colspan="3" style="background:#1a252f;">Macro</th>
            <th colspan="3" style="background:#1a252f;">Micro</th>
            <th colspan="3" style="background:#1a252f;">Conteo (micro)</th>
          </tr>
          <tr>
            <th>chunk_thr</th><th>cov</th><th>conf</th><th>size</th>
            <th>P</th><th>R</th><th>F1</th>
            <th>P</th><th>R</th><th>F1</th>
            <th>TP</th><th>FP</th><th>FN</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>F1 por proyecto y configuración</h2>
    <div class="overflow-x">
      <table>
        <thead><tr><th>Proyecto</th>{proj_header}</tr></thead>
        <tbody>
          {proj_table}
        </tbody>
      </table>
    </div>
  </div>

  {img_metrics_block}
  {img_proj_block}
"""


def generate_html_report(
    summary: Dict[str, Any],
    output_path: Path,
    chart_paths: Dict[str, Path],
) -> None:
    """Escribe el reporte HTML standalone (imágenes referenciadas por ruta relativa)."""
    cards = render_report_cards(
        summary, chart_paths, embed_images=False, rel_to=output_path.parent
    )
    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Parámetros de clasificación de tópicos</title>
  <style>{REPORT_CSS}</style>
</head>
<body>
{cards}
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def build_project_titles(gt_data: dict) -> Dict[str, str]:
    titles: Dict[str, str] = {}
    for e in gt_data.get("entities", []):
        if e.get("label") not in ("Proyecto", "Grupo"):
            continue
        val = e.get("value")
        if isinstance(val, dict):
            titles[e["id"]] = val.get("title", "") or ""
        elif isinstance(val, str):
            titles[e["id"]] = val
    return titles


def parse_float_list(raw: str, name: str) -> List[float]:
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise ValueError(f"La lista de {name} no puede estar vacía.")
    try:
        return [float(v) for v in parts]
    except ValueError:
        raise ValueError(f"Valores inválidos para {name}: {raw!r}")


def parse_size_list(raw: str) -> List[Optional[int]]:
    """Lista de tamaños de chunk; acepta 'none'/'default' para usar los chunks existentes."""
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise ValueError("La lista de chunk_size no puede estar vacía.")
    out: List[Optional[int]] = []
    for v in parts:
        if v.lower() in ("none", "default", "def"):
            out.append(None)
        else:
            try:
                out.append(int(v))
            except ValueError:
                raise ValueError(f"chunk_size inválido: {v!r}")
    return out


def validate_grid(combos: List[Dict[str, Any]]) -> None:
    """Valida la grilla (p.ej. el piso de cache para chunking_threshold)."""
    too_low = sorted({c["chunking_threshold"] for c in combos if c["chunking_threshold"] < RAW_SCORE_FLOOR})
    if too_low:
        raise ValueError(
            f"Valores de chunking_threshold por debajo del piso de cache "
            f"({RAW_SCORE_FLOOR}): {too_low}. Subí los valores o bajá RAW_SCORE_FLOOR."
        )


def load_gt_context(gt_file: Path, device: Optional[str]) -> Dict[str, Any]:
    """Carga GT (proyectos, docs, tópicos), títulos y el mapeo EN->ES."""
    gt_data = load_json(gt_file)
    docs_by_project, gt_topics_by_project = build_ground_truth(gt_data)
    project_titles = build_project_titles(gt_data)
    mapping_extractor = BertTopicExtractor(None, None, None, device=device)
    return {
        "gt_file": gt_file,
        "docs_by_project": docs_by_project,
        "gt_topics_by_project": gt_topics_by_project,
        "project_titles": project_titles,
        "en_to_es": mapping_extractor._en_to_es_topic,
    }


def evaluate_combos(
    combos: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    chunks_dir: Path,
    docling_dir: Path,
    device: Optional[str],
    max_projects: Optional[int],
    use_cache: bool,
) -> List[Dict[str, Any]]:
    """Evalúa cada combinación de parámetros y devuelve las configuraciones con métricas."""
    new_configs: List[Dict[str, Any]] = []
    raw_by_size: Dict[Optional[int], Dict[str, Any]] = {}  # cache en memoria por chunk_size

    varying = varying_dims([{"params": c} for c in combos])

    for params in combos:
        size = params["chunk_size"]
        if size not in raw_by_size:
            raw_by_size[size] = get_raw_predictions(
                ctx["docs_by_project"],
                chunks_dir,
                docling_dir,
                size,
                ctx["project_titles"],
                device,
                max_projects,
                use_cache=use_cache,
            )
        raw = raw_by_size[size]

        predicted = aggregate_predicted_topics(
            raw,
            params["chunking_threshold"],
            params["coverage_threshold"],
            params["confidence_threshold"],
            ctx["en_to_es"],
        )
        metrics = score_config(predicted, ctx["gt_topics_by_project"])
        label = config_label(params, varying)
        new_configs.append(
            {
                "signature": config_signature(params),
                "label": label,
                "params": params,
                "metrics": metrics,
            }
        )
        print(
            f"  -> {label}: "
            f"F1 macro={metrics['macro']['f1_macro']:.3f} "
            f"P={metrics['macro']['precision_macro']:.3f} "
            f"R={metrics['macro']['recall_macro']:.3f} "
            f"(n={metrics['n_projects']})"
        )
    return new_configs


def merge_and_write(
    gt_file: Path,
    new_configs: List[Dict[str, Any]],
    fresh: bool,
) -> Dict[str, Any]:
    """Mergea con corridas previas (acumulativo), persiste JSON + gráficas + HTML y
    devuelve la estructura completa lista para renderizar.

    Re-etiqueta y ordena todas las configuraciones (previas + nuevas) según qué
    parámetros varían en el conjunto combinado, para que el reporte sea coherente."""
    details_path = RESULTS_DIR / "topic_params_details.json"
    configs_by_sig: Dict[str, Dict[str, Any]] = {}
    if not fresh and details_path.exists():
        prev = load_json(details_path)
        for c in prev.get("configs", []):
            configs_by_sig[c["signature"]] = c
    for c in new_configs:
        configs_by_sig[c["signature"]] = c

    configs = list(configs_by_sig.values())

    varying = varying_dims(configs)
    for c in configs:
        c["label"] = config_label(c["params"], varying)

    sort_dims = varying if varying else list(PARAM_DIMENSIONS)
    configs.sort(
        key=lambda c: tuple((c["params"][d] is None, c["params"][d]) for d in sort_dims)
    )

    fixed = {d: configs[0]["params"][d] for d in PARAM_DIMENSIONS if d not in varying}

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    full_summary = {
        "sweep": sweep_description(varying),
        "gt_file": str(gt_file),
        "generated_at": generated_at,
        "fixed_params": fixed,
        "configs": configs,
    }

    stripped_summary = dict(full_summary)
    stripped_summary["configs"] = [
        {k: v for k, v in c.items() if k != "metrics"}
        | {
            "n_projects": c["metrics"]["n_projects"],
            "macro": c["metrics"]["macro"],
            "micro": c["metrics"]["micro"],
        }
        for c in configs
    ]
    save_json(stripped_summary, RESULTS_DIR / "topic_params_summary.json")
    save_json(
        {"sweep": full_summary["sweep"], "generated_at": generated_at, "configs": configs},
        details_path,
    )

    chart_paths = generate_charts(configs, RESULTS_DIR / "images")
    generate_html_report(full_summary, RESULTS_DIR / "topic_params_report.html", chart_paths)

    return {"summary": full_summary, "configs": configs, "chart_paths": chart_paths}


def main() -> None:
    # Los flags (opcionales) sobrescriben las constantes de arriba.
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunking-thresholds", default=None,
                    help="Lista (coma) de umbrales de chunking. Default: constante CHUNKING_THRESHOLDS.")
    ap.add_argument("--coverage-thresholds", default=None,
                    help="Lista (coma) de coverage final. Default: constante COVERAGE_THRESHOLDS.")
    ap.add_argument("--confidence-thresholds", default=None,
                    help="Lista (coma) de confidence final. Default: constante CONFIDENCE_THRESHOLDS.")
    ap.add_argument("--chunk-sizes", default=None,
                    help="Lista (coma) de tamaños de chunk; 'none' = chunks existentes. "
                         "Default: constante CHUNK_SIZES.")
    ap.add_argument("--gt-file", type=Path, default=DEFAULT_GT_PATH)
    ap.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    ap.add_argument("--docling-dir", type=Path, default=DEFAULT_DOCLING_DIR)
    ap.add_argument("--max-projects", type=int, default=MAX_PROJECTS)
    ap.add_argument("--device", default=DEVICE, help="cuda / cpu (auto si no se especifica).")
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignorar el cache de predicciones crudas (re-corre BERT).")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignorar el reporte previo y empezar de cero.")
    args = ap.parse_args()

    # Cada dimensión: del flag (string a parsear) o de la constante.
    chunking = parse_float_list(args.chunking_thresholds, "chunking_threshold") if args.chunking_thresholds else list(CHUNKING_THRESHOLDS)
    coverage = parse_float_list(args.coverage_thresholds, "coverage_threshold") if args.coverage_thresholds else list(COVERAGE_THRESHOLDS)
    confidence = parse_float_list(args.confidence_thresholds, "confidence_threshold") if args.confidence_thresholds else list(CONFIDENCE_THRESHOLDS)
    sizes = parse_size_list(args.chunk_sizes) if args.chunk_sizes else list(CHUNK_SIZES)

    combos = build_param_grid(chunking, coverage, confidence, sizes)
    validate_grid(combos)

    ctx = load_gt_context(args.gt_file, args.device)
    if not ctx["gt_topics_by_project"]:
        print("Aviso: el GT no tiene relaciones TIENE_TOPICO; no hay con qué comparar.")
    print(
        f"Grilla: {len(combos)} combinación(es) "
        f"(chunking={chunking}, coverage={coverage}, confidence={confidence}, chunk_size={sizes})\n"
        f"Proyectos en GT: {len(ctx['docs_by_project'])} · con tópicos GT: "
        f"{len(ctx['gt_topics_by_project'])}\n"
    )

    new_configs = evaluate_combos(
        combos,
        ctx,
        args.chunks_dir,
        args.docling_dir,
        args.device,
        args.max_projects,
        use_cache=not args.no_cache and USE_CACHE,
    )
    result = merge_and_write(args.gt_file, new_configs, args.fresh or FRESH)

    print("\n" + "=" * 70)
    print(f"Configuraciones totales en el reporte: {len(result['configs'])}")
    print(f"Resumen JSON:  {RESULTS_DIR / 'topic_params_summary.json'}")
    print(f"Detalle JSON:  {RESULTS_DIR / 'topic_params_details.json'}")
    print(f"Gráficas:      {RESULTS_DIR / 'images'}")
    print(f"Reporte HTML:  {RESULTS_DIR / 'topic_params_report.html'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
