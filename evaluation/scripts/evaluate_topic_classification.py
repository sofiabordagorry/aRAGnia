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

from institutional_graphrag.config import CHUNK_MAX_TOKENS  # noqa: E402
from institutional_graphrag.extraction.bert_extractor import (  # noqa: E402
    DEFAULT_CONFIDENCE_LOGIT_THRESHOLD,
    DEFAULT_COVERAGE_LOGIT_THRESHOLD,
    DEFAULT_MIN_TOPICS,
    DEFAULT_THRESHOLD,
    BertTopicExtractor,
)

DEFAULT_GT_PATH = EVAL_DIR / "ground_truth" / "extraction" / "ground_truth_kg_evaluation.json"
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
    "min_topics",
)
PARAM_SHORT = {
    "chunking_threshold": "ck",
    "coverage_threshold": "cov",
    "confidence_threshold": "conf",
    "chunk_size": "size",
    "min_topics": "mt",
}

CHUNKING_THRESHOLDS: List[float] = [0.08]  # umbral etapa de chunking (score por chunk)
COVERAGE_THRESHOLDS: List[float] = [0.02]  # umbral final: fracción de chunks
CONFIDENCE_THRESHOLDS: List[float] = [13.5]  # umbral final: logit promedio
CHUNK_SIZES: List[int] = [300]   # max_tokens por chunk
MIN_TOPICS_VALUES: List[int] = [1]   # piso de tópicos por proyecto (siempre fallback)

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

    # id de Documento -> nombre_base (descartando tablas)
    doc_base_name: Dict[str, str] = {}
    for e in entities:
        if e.get("label") != "Documento":
            continue
        value = e.get("value", {}) or {}
        if value.get("tipo") == "tabla":
            continue
        base_name = value.get("nombre_base")
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

def _rechunk_project(
    docling_dir: Path,
    base_names: List[str],
    chunk_size: int,
) -> Dict[str, List[dict]]:
    """Re-chunkea los documentos del proyecto desde data/docling con un max_tokens dado.

    Usa el mismo chunker que producción (HybridChunker + merge_peers); con chunk_size=512
    reproduce exactamente los chunks del pipeline real (max_tokens default de e5-large-v2)."""
    from docling.chunking import HybridChunker
    from docling_core.types.doc import DoclingDocument

    from institutional_graphrag.config import EMBED_MODEL_ID
    from institutional_graphrag.ingest.chunker import chunk_document

    chunker = HybridChunker(tokenizer=EMBED_MODEL_ID, max_tokens=chunk_size, merge_peers=True)
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
    docling_dir: Path,
    chunk_size: int,
    project_title_by_id: Dict[str, str],
    device: Optional[str],
    max_projects: Optional[int],
) -> Dict[str, Any]:
    """Corre BERT una vez por proyecto y devuelve, por chunk, todos los tópicos con
    score >= RAW_SCORE_FLOOR. Re-chunkea desde docling con ``chunk_size`` como max_tokens.
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


def raw_cache_path(chunk_size: int) -> Path:
    return CACHE_DIR / f"raw_preds_size{chunk_size}.json"


def get_raw_predictions(
    docs_by_project: Dict[str, List[str]],
    docling_dir: Path,
    chunk_size: int,
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

    print(f"Corriendo BERT sobre los proyectos [chunk_size={chunk_size}]...")
    raw = compute_raw_predictions(
        docs_by_project,
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
    min_topics: int = DEFAULT_MIN_TOPICS,
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

        def coverage(topic_en: str) -> float:
            return topic_count[topic_en] / total_chunks

        def confidence(topic_en: str) -> float:
            return topic_sum_logit[topic_en] / topic_count[topic_en]

        passing = [
            topic_en
            for topic_en in topic_count
            if coverage(topic_en) >= coverage_threshold
            and confidence(topic_en) >= confidence_threshold
        ]
        selected = list(passing)
        if len(selected) < min_topics:
            fallback = sorted(
                (t for t in topic_count if t not in passing),
                key=lambda t: (coverage(t), confidence(t)),
                reverse=True,
            )
            selected.extend(fallback[: min_topics - len(selected)])

        predicted[project_id] = {normalize_topic_key(en_to_es.get(t, t)) for t in selected}

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


def build_topic_to_subfield(extractor: BertTopicExtractor) -> Dict[str, str]:
    """Mapeo clave-de-tópico (ES, normalizada) -> clave-de-subcampo (ES, normalizada),
    derivado de la jerarquía OpenAlex del extractor."""
    mapping: Dict[str, str] = {}
    for topic_en, subfield_en in extractor._topic_to_subfield.items():
        es_topic = extractor._en_to_es_topic.get(topic_en, topic_en)
        es_subfield = extractor._en_to_es_subfield.get(subfield_en, subfield_en)
        mapping[normalize_topic_key(es_topic)] = normalize_topic_key(es_subfield)
    return mapping


def to_subfields(topic_keys: set, topic_to_subfield: Dict[str, str]) -> set:
    """Colapsa un set de claves de tópico a sus claves de subcampo (tópico sin mapeo
    queda como su propia clave, para no perderlo silenciosamente)."""
    return {topic_to_subfield.get(k, k) for k in topic_keys}


def score_at_subfield(
    predicted: Dict[str, set],
    gt_topics_by_project: Dict[str, set],
    topic_to_subfield: Dict[str, str],
) -> Dict[str, Any]:
    """Igual que score_config pero comparando conjuntos de subcampos en vez de tópicos."""
    predicted_sf = {pid: to_subfields(s, topic_to_subfield) for pid, s in predicted.items()}
    gt_sf = {pid: to_subfields(s, topic_to_subfield) for pid, s in gt_topics_by_project.items()}
    return score_config(predicted_sf, gt_sf)


def build_param_grid(
    chunking: List[float],
    coverage: List[float],
    confidence: List[float],
    sizes: List[int],
    min_topics: List[int],
) -> List[Dict[str, Any]]:
    """Producto cartesiano de las listas -> lista de combinaciones de parámetros."""
    combos: List[Dict[str, Any]] = []
    for ck, cov, conf, size, mt in itertools.product(chunking, coverage, confidence, sizes, min_topics):
        combos.append(
            {
                "chunking_threshold": ck,
                "coverage_threshold": cov,
                "confidence_threshold": conf,
                "chunk_size": size,
                "min_topics": mt,
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


def config_signature(params: Dict[str, Any]) -> str:
    return (
        f"ck{params['chunking_threshold']}_cov{params['coverage_threshold']}"
        f"_conf{params['confidence_threshold']}_size{params['chunk_size']}"
        f"_mt{params['min_topics']}"
    )


def config_label(params: Dict[str, Any], varying: List[str]) -> str:
    """Etiqueta corta: muestra solo los parámetros que varían entre combinaciones.

    Si no varía ninguno (una sola combinación) muestra los cuatro valores."""
    dims = varying if varying else list(PARAM_DIMENSIONS)
    return " ".join(f"{PARAM_SHORT[d]}={params[d]}" for d in dims)


PALETTE = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#34495e"]

PARAM_LABELS_FULL = {
    "chunking_threshold": "Umbral de chunking",
    "coverage_threshold": "Umbral de cobertura",
    "confidence_threshold": "Umbral de confianza (logit)",
    "chunk_size": "Tamaño de chunk",
    "min_topics": "Mínimo de tópicos (piso)",
}
OPERATING_POINT = {
    "chunking_threshold": DEFAULT_THRESHOLD,
    "coverage_threshold": DEFAULT_COVERAGE_LOGIT_THRESHOLD,
    "confidence_threshold": float(DEFAULT_CONFIDENCE_LOGIT_THRESHOLD),
    "chunk_size": CHUNK_MAX_TOKENS,
    "min_topics": DEFAULT_MIN_TOPICS,
}


def _reference_config(configs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Config de referencia para las gráficas: el punto de operación deployado si está en
    el barrido; si no, la mejor por F1 macro a nivel tópico."""
    for c in configs:
        if all(c["params"].get(k) == v for k, v in OPERATING_POINT.items()):
            return c
    return max(configs, key=lambda c: c["metrics"]["macro"]["f1_macro"])


def _confusion_counts(per_project: Dict[str, Any]) -> Dict[str, int]:
    """Matriz de confusión 2x2 agregada sobre las celdas proyecto×tópico.

    El universo de tópicos es la unión de los esperados (GT) y predichos en todo el
    corpus; el TN cuenta las celdas de ese universo correctamente no asignadas. Ojo:
    el TN (y por ende la accuracy) queda inflado por lo esparso del espacio de etiquetas
    —cada proyecto tiene pocos tópicos de los muchos posibles—, así que P/R/F1 siguen
    siendo las métricas informativas, no la accuracy."""
    universe: set = set()
    for r in per_project.values():
        universe |= set(r.get("predicted", [])) | set(r.get("expected", []))
    tp = fp = fn = tn = 0
    for r in per_project.values():
        pred = set(r.get("predicted", []))
        gold = set(r.get("expected", []))
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
        tn += len(universe - (pred | gold))
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def generate_charts(configs: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Path]:
    """Gráficas representativas para el informe (no el barrido completo de 600+ configs).

    Toma una config de referencia (el punto de operación deployado) y produce:
      - chart_topic_sensitivity.png: P/R/F1 macro variando cada hiperparámetro de a uno.
      - chart_topic_by_project.png:  P/R/F1 por proyecto en ese punto, tópico y subcampo.
    """
    images_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    if not configs:
        return paths

    ref = _reference_config(configs)
    ref_params = ref["params"]
    metric_keys = ("precision_macro", "recall_macro", "f1_macro")
    metric_labels = ("Precisión", "Recall", "F1")
    sub_keys = ("precision", "recall", "f1")

    # 1) Sensibilidad: una gráfica por hiperparámetro. Cada una es un corte 1-D del barrido:
    #    se varía ese parámetro y se dejan los otros tres fijos en la config de referencia.
    #    La línea punteada marca el valor elegido.
    for param in PARAM_DIMENSIONS:
        sl = [c for c in configs
              if all(c["params"][k] == v for k, v in ref_params.items() if k != param)]
        sl.sort(key=lambda c: c["params"][param])
        xs = [c["params"][param] for c in sl]
        idx = np.arange(len(sl))
        fig, ax = plt.subplots(figsize=(6.0, 4.2))
        for color, mk, ml in zip(PALETTE, metric_keys, metric_labels):
            ax.plot(idx, [c["metrics"]["macro"][mk] for c in sl], marker="o", markersize=6,
                    color=color, label=ml)
        if ref_params[param] in xs:
            ax.axvline(xs.index(ref_params[param]), color="#7f8c8d", linestyle="--", linewidth=1.2,
                       label="valor elegido")
        ax.set_xticks(idx)
        ax.set_xticklabels([f"{x:g}" for x in xs], fontsize=10)
        ax.set_title(f"Sensibilidad a: {PARAM_LABELS_FULL[param]}", fontsize=12)
        ax.set_xlabel(PARAM_LABELS_FULL[param])
        ax.set_ylabel("Score macro (nivel tópico)")
        top = max([c["metrics"]["macro"][k] for c in sl for k in ("recall_macro", "f1_macro")] + [0.3])
        ax.set_ylim(0, min(1.02, top + 0.05))
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        fig.tight_layout()
        p = images_dir / f"chart_topic_sens_{PARAM_SHORT[param]}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths[f"sens_{PARAM_SHORT[param]}"] = p

    # 2) Por proyecto en la config de referencia (nivel tópico y, si está, subcampo).
    projects = list(ref["metrics"]["per_project"])
    if projects:
        levels = [("metrics", "Nivel tópico")]
        if ref.get("metrics_subfield", {}).get("per_project"):
            levels.append(("metrics_subfield", "Nivel subcampo"))
        y = np.arange(len(projects))
        bar_w = 0.26
        fig, axarr = plt.subplots(
            1, len(levels), figsize=(6 * len(levels), max(3.2, 0.55 * len(projects) + 1.8)),
            sharey=True, squeeze=False,
        )
        axs = axarr[0]
        for ax, (mk, title) in zip(axs, levels):
            per = ref[mk]["per_project"]
            for i, (color, sub, ml) in enumerate(zip(PALETTE, sub_keys, metric_labels)):
                ax.barh(y + (1 - i) * bar_w, [per[p][sub] for p in projects], bar_w, color=color, label=ml)
            ax.set_title(title, fontsize=11)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("Score (0–1)")
            ax.spines[["top", "right"]].set_visible(False)
        axs[0].set_yticks(y)
        axs[0].set_yticklabels(projects, fontsize=9)
        axs[-1].legend(fontsize=9, loc="lower right", framealpha=0.9)
        fig.suptitle(f"Desempeño por proyecto en el punto de operación ({ref['label']})", fontsize=12)
        fig.tight_layout()
        p = images_dir / "chart_topic_by_project.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["by_project"] = p

    # 3) Matriz de confusión 2x2 (nivel tópico y, si está, subcampo) en el punto de
    #    operación. Correctas (TP/TN) en verde, errores (FP/FN) en rojo; el número
    #    grande es el conteo de celdas proyecto×tópico.
    cm_levels = [("metrics", "Tópico")]
    if ref.get("metrics_subfield", {}).get("per_project"):
        cm_levels.append(("metrics_subfield", "Subcampo"))
    fig, axarr = plt.subplots(1, len(cm_levels), figsize=(4.2 * len(cm_levels), 4.0), squeeze=False)
    cell_color = np.array([["#27ae60", "#e74c3c"], ["#e74c3c", "#27ae60"]])  # TP FP / FN TN
    cell_label = np.array([["TP", "FP"], ["FN", "TN"]])
    for ax, (mk, title) in zip(axarr[0], cm_levels):
        cm = _confusion_counts(ref[mk]["per_project"])
        grid = np.array([[cm["tp"], cm["fp"]], [cm["fn"], cm["tn"]]])
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j, 1 - i), 1, 1, color=cell_color[i, j], alpha=0.18))
                ax.text(j + 0.5, 1 - i + 0.60, cell_label[i, j], ha="center", va="center",
                        fontsize=11, color=cell_color[i, j], fontweight="bold")
                ax.text(j + 0.5, 1 - i + 0.34, f"{grid[i, j]:,}", ha="center", va="center",
                        fontsize=16, color="#2c3e50")
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 2)
        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["GT: sí", "GT: no"], fontsize=9)
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(["pred: no", "pred: sí"], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(f"Matriz de confusión (celdas proyecto×tópico) — {ref['label']}", fontsize=12)
    fig.tight_layout()
    p = images_dir / "chart_topic_confusion.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["confusion"] = p

    # 4) Resumen Precisión / Recall / F1 (macro) en el punto de operación:
    #    tópico vs subcampo, barras agrupadas.
    prf_names = ["Precisión", "Recall", "F1"]
    prf_keys = ("precision_macro", "recall_macro", "f1_macro")
    groups = [("Tópico", [ref["metrics"]["macro"][k] for k in prf_keys], PALETTE[0])]
    if ref.get("metrics_subfield", {}).get("macro"):
        groups.append(
            ("Subcampo", [ref["metrics_subfield"]["macro"][k] for k in prf_keys], PALETTE[2])
        )
    xpos = np.arange(len(prf_names))
    bw = 0.8 / len(groups)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for gi, (gname, vals, color) in enumerate(groups):
        offset = (gi - (len(groups) - 1) / 2) * bw
        bars = ax.bar(xpos + offset, vals, bw, label=gname, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(prf_names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score macro (0–1)")
    ax.set_title(f"Precisión / Recall / F1 (macro) — {ref['label']}", fontsize=12)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    p = images_dir / "chart_topic_prf.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths["prf"] = p

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


def _chart_src(path: Path, rel_to: Optional[Path]) -> str:
    """Devuelve el src de una imagen como ruta relativa al reporte."""
    return str(path.relative_to(rel_to)) if rel_to else str(path)


def render_report_cards(
    summary: Dict[str, Any],
    chart_paths: Dict[str, Path],
    rel_to: Optional[Path] = None,
) -> str:
    """Genera el cuerpo del reporte (h1 + subtítulo + cards)."""
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
            f"<td>{p['chunk_size']}</td>"
            f"<td>{p['min_topics']}</td>"
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
        ", ".join(f"{PARAM_SHORT[d]}={v}" for d, v in fixed.items())
        if fixed
        else "(ninguno)"
    )
    analysis = [
        f"La mejor configuración por F1 macro (nivel tópico) es <strong>{best['label']}</strong> "
        f"(F1={best['metrics']['macro']['f1_macro']:.3f}, "
        f"P={best['metrics']['macro']['precision_macro']:.3f}, "
        f"R={best['metrics']['macro']['recall_macro']:.3f}).",
    ]
    sf_configs = [c for c in configs if c.get("metrics_subfield")]
    if sf_configs:
        best_sf = max(sf_configs, key=lambda c: c["metrics_subfield"]["macro"]["f1_macro"])
        analysis.append(
            f"A <strong>nivel subcampo</strong> (colapsando tópico → subcampo) la mejor es "
            f"<strong>{best_sf['label']}</strong> "
            f"(F1={best_sf['metrics_subfield']['macro']['f1_macro']:.3f}, "
            f"P={best_sf['metrics_subfield']['macro']['precision_macro']:.3f}, "
            f"R={best_sf['metrics_subfield']['macro']['recall_macro']:.3f}). "
            f"Suele ser bastante más alta: BERT acierta el subcampo aunque erre el tópico exacto."
        )
    analysis += [
        f"Parámetros que varían: <strong>{summary['sweep']}</strong> · "
        f"{len(configs)} combinación(es).",
        f"Parámetros fijos: <strong>{fixed_str}</strong>.",
        "El GT de tópicos está <strong>pendiente de actualización con los valores de los "
        "profes</strong>: las métricas son provisionales y sirven para comparar parámetros "
        "entre sí, no como score absoluto.",
    ]
    analysis_html = "\n".join(f"<li>{a}</li>" for a in analysis)

    sens_imgs = "".join(
        f'<div class="chart-wrap"><img src="{_chart_src(chart_paths[k], rel_to)}" alt="{k}"></div>'
        for k in ("sens_ck", "sens_cov", "sens_conf", "sens_size")
        if k in chart_paths
    )
    img_sensitivity_block = (
        f'<div class="card"><h2>Sensibilidad a cada hiperparámetro (punto de operación)</h2>{sens_imgs}</div>'
        if sens_imgs
        else ""
    )
    img_by_project_block = (
        f'<div class="card"><h2>Desempeño por proyecto (punto de operación)</h2>'
        f'<div class="chart-wrap"><img src="{_chart_src(chart_paths["by_project"], rel_to)}" alt="por proyecto"></div></div>'
        if "by_project" in chart_paths
        else ""
    )
    img_prf_block = (
        f'<div class="card"><h2>Precisión / Recall / F1 (punto de operación)</h2>'
        f'<div class="chart-wrap"><img src="{_chart_src(chart_paths["prf"], rel_to)}" alt="P/R/F1"></div></div>'
        if "prf" in chart_paths
        else ""
    )
    img_confusion_block = (
        f'<div class="card"><h2>Matriz de confusión (punto de operación)</h2>'
        f'<p class="subtitle">Celdas proyecto×tópico sobre el universo de tópicos en juego '
        f'(GT ∪ predichos). No se reporta accuracy: el TN domina el espacio esparso y la infla; '
        f'mirá Precisión / Recall / F1.</p>'
        f'<div class="chart-wrap"><img src="{_chart_src(chart_paths["confusion"], rel_to)}" alt="matriz de confusión"></div></div>'
        if "confusion" in chart_paths
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
            <th colspan="5" style="background:#1a252f;">Parámetros</th>
            <th rowspan="2">Proy.</th>
            <th colspan="3" style="background:#1a252f;">Macro</th>
            <th colspan="3" style="background:#1a252f;">Micro</th>
            <th colspan="3" style="background:#1a252f;">Conteo (micro)</th>
          </tr>
          <tr>
            <th>chunk_thr</th><th>cov</th><th>conf</th><th>size</th><th>mt</th>
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

  {img_sensitivity_block}
  {img_by_project_block}
"""


def generate_html_report(
    summary: Dict[str, Any],
    output_path: Path,
    chart_paths: Dict[str, Path],
) -> None:
    """Escribe el reporte HTML standalone (imágenes referenciadas por ruta relativa)."""
    cards = render_report_cards(summary, chart_paths, rel_to=output_path.parent)
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
            titles[e["id"]] = val.get("titulo", "") or ""
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


def parse_size_list(raw: str) -> List[int]:
    """Lista de tamaños de chunk (max_tokens). 300 = config de prod."""
    parts = [v.strip() for v in raw.split(",") if v.strip()]
    if not parts:
        raise ValueError("La lista de chunk_size no puede estar vacía.")
    try:
        return [int(v) for v in parts]
    except ValueError:
        raise ValueError(f"Valores inválidos para chunk_size: {raw!r}")


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
        "topic_to_subfield": build_topic_to_subfield(mapping_extractor),
    }


def evaluate_combos(
    combos: List[Dict[str, Any]],
    ctx: Dict[str, Any],
    docling_dir: Path,
    device: Optional[str],
    max_projects: Optional[int],
    use_cache: bool,
) -> List[Dict[str, Any]]:
    """Evalúa cada combinación de parámetros y devuelve las configuraciones con métricas."""
    new_configs: List[Dict[str, Any]] = []
    raw_by_size: Dict[int, Dict[str, Any]] = {}  # cache en memoria por chunk_size

    varying = varying_dims([{"params": c} for c in combos])

    for params in combos:
        size = params["chunk_size"]
        if size not in raw_by_size:
            raw_by_size[size] = get_raw_predictions(
                ctx["docs_by_project"],
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
            params["min_topics"],
        )
        metrics = score_config(predicted, ctx["gt_topics_by_project"])
        metrics_subfield = score_at_subfield(
            predicted, ctx["gt_topics_by_project"], ctx["topic_to_subfield"]
        )
        label = config_label(params, varying)
        new_configs.append(
            {
                "signature": config_signature(params),
                "label": label,
                "params": params,
                "metrics": metrics,
                "metrics_subfield": metrics_subfield,
            }
        )
        print(
            f"  -> {label}: "
            f"F1 macro={metrics['macro']['f1_macro']:.3f} "
            f"P={metrics['macro']['precision_macro']:.3f} "
            f"R={metrics['macro']['recall_macro']:.3f} "
            f"(n={metrics['n_projects']}) "
            f"| subcampo F1={metrics_subfield['macro']['f1_macro']:.3f}"
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
    configs.sort(key=lambda c: tuple(c["params"][d] for d in sort_dims))

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
        {k: v for k, v in c.items() if k not in ("metrics", "metrics_subfield")}
        | {
            "n_projects": c["metrics"]["n_projects"],
            "macro": c["metrics"]["macro"],
            "micro": c["metrics"]["micro"],
            "macro_subfield": c.get("metrics_subfield", {}).get("macro"),
            "micro_subfield": c.get("metrics_subfield", {}).get("micro"),
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
                    help="Lista (coma) de tamaños de chunk (max_tokens); 300 = prod. "
                         "Default: constante CHUNK_SIZES.")
    ap.add_argument("--min-topics", default=None,
                    help="Lista (coma) de pisos de tópicos por proyecto (0 = sin fallback). "
                         "Default: constante MIN_TOPICS_VALUES.")
    ap.add_argument("--gt-file", type=Path, default=DEFAULT_GT_PATH)
    ap.add_argument("--docling-dir", type=Path, default=DEFAULT_DOCLING_DIR)
    ap.add_argument("--max-projects", type=int, default=MAX_PROJECTS)
    ap.add_argument("--device", default=DEVICE, help="cuda / cpu (auto si no se especifica).")
    ap.add_argument("--no-cache", action="store_true",
                    help="Ignorar el cache de predicciones crudas (re-corre BERT).")
    ap.add_argument("--fresh", action="store_true",
                    help="Ignorar el reporte previo y empezar de cero.")
    ap.add_argument("--replot", action="store_true",
                    help="No corre el barrido: regenera solo las gráficas y el HTML "
                         "desde topic_params_details.json ya existente.")
    args = ap.parse_args()

    if args.replot:
        details_path = RESULTS_DIR / "topic_params_details.json"
        summary_path = RESULTS_DIR / "topic_params_summary.json"
        if not details_path.exists():
            raise FileNotFoundError(f"No existe {details_path}; corré la calibración primero.")
        configs = json.loads(details_path.read_text(encoding="utf-8"))["configs"]
        meta = json.loads(summary_path.read_text(encoding="utf-8"))
        full_summary = {**meta, "configs": configs}
        chart_paths = generate_charts(configs, RESULTS_DIR / "images")
        generate_html_report(full_summary, RESULTS_DIR / "topic_params_report.html", chart_paths)
        ref = _reference_config(configs)
        print(f"Punto de operación: {ref['label']} · "
              f"F1 tópico={ref['metrics']['macro']['f1_macro']:.3f} "
              f"subcampo={ref['metrics_subfield']['macro']['f1_macro']:.3f}")
        print(f"Regeneradas {len(chart_paths)} gráficas + HTML (sin re-correr el barrido).")
        return

    # Cada dimensión: del flag (string a parsear) o de la constante.
    chunking = parse_float_list(args.chunking_thresholds, "chunking_threshold") if args.chunking_thresholds else list(CHUNKING_THRESHOLDS)
    coverage = parse_float_list(args.coverage_thresholds, "coverage_threshold") if args.coverage_thresholds else list(COVERAGE_THRESHOLDS)
    confidence = parse_float_list(args.confidence_thresholds, "confidence_threshold") if args.confidence_thresholds else list(CONFIDENCE_THRESHOLDS)
    sizes = parse_size_list(args.chunk_sizes) if args.chunk_sizes else list(CHUNK_SIZES)
    min_topics = [int(v) for v in args.min_topics.split(",") if v.strip()] if args.min_topics else list(MIN_TOPICS_VALUES)

    combos = build_param_grid(chunking, coverage, confidence, sizes, min_topics)
    validate_grid(combos)

    ctx = load_gt_context(args.gt_file, args.device)
    if not ctx["gt_topics_by_project"]:
        print("Aviso: el GT no tiene relaciones TIENE_TOPICO; no hay con qué comparar.")
    print(
        f"Grilla: {len(combos)} combinación(es) "
        f"(chunking={chunking}, coverage={coverage}, confidence={confidence}, chunk_size={sizes}, min_topics={min_topics})\n"
        f"Proyectos en GT: {len(ctx['docs_by_project'])} · con tópicos GT: "
        f"{len(ctx['gt_topics_by_project'])}\n"
    )

    new_configs = evaluate_combos(
        combos,
        ctx,
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
