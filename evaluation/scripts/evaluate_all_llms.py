#!/usr/bin/env python3
"""
evaluate_all_llms.py

Evaluates all LLMs in evaluation/llm_extraction/ against the ground truth
extraction and produces a comparison report (JSON + HTML).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from util.graph_compare import compare_all_projects
from util.graph_validation import validate_graph_against_schema

GT_PATH = (
    Path(__file__).parents[1]
    / "ground_truth"
    / "extraction"
    / "ground_truth_kg.json"
)

LLM_EXTRACTION_DIR = Path(__file__).parents[1] / "llm_extraction"

RESULTS_PATH = Path(__file__).parents[1] / "results/extraction"

LLM_DISPLAY_NAMES = {
    "google_gemma-4-E4B-it": "Gemma 4 E4B",
    "Qwen_Qwen2.5-7B-Instruct": "Qwen2.5 7B",
    "Qwen_Qwen2.5-Coder-14B-Instruct": "Qwen2.5-Coder 14B",
    "meta-llama_Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "deepseek-ai_deepseek-coder-6.7b-instruct": "DeepSeek-Coder 6.7B",
}


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _aggregate_overall(projects: Dict[str, Any], key: str) -> Dict[str, float]:
    """Macro-average precision, recall, F1 across all projects for a given overall key."""
    p_vals, r_vals, f1_vals = [], [], []
    tp_total = fp_total = fn_total = 0

    for proj_data in projects.values():
        m = proj_data[key]
        p_vals.append(m["precision"])
        r_vals.append(m["recall"])
        f1_vals.append(m["f1"])
        tp_total += m["tp"]
        fp_total += m["fp"]
        fn_total += m["fn"]

    return {
        "precision_macro": mean(p_vals),
        "recall_macro": mean(r_vals),
        "f1_macro": mean(f1_vals),
        "tp_total": tp_total,
        "fp_total": fp_total,
        "fn_total": fn_total,
    }


def _aggregate_by_group(projects: Dict[str, Any], key: str) -> Dict[str, Dict[str, float]]:
    """Macro-average metrics per label/type across all projects."""
    groups: Dict[str, Dict[str, List]] = defaultdict(lambda: defaultdict(list))

    for proj_data in projects.values():
        for group_name, m in proj_data[key].items():
            groups[group_name]["precision"].append(m["precision"])
            groups[group_name]["recall"].append(m["recall"])
            groups[group_name]["f1"].append(m["f1"])
            groups[group_name]["tp"].append(m["tp"])
            groups[group_name]["fp"].append(m["fp"])
            groups[group_name]["fn"].append(m["fn"])

    return {
        group: {
            "precision_macro": mean(vals["precision"]),
            "recall_macro": mean(vals["recall"]),
            "f1_macro": mean(vals["f1"]),
            "tp_total": sum(vals["tp"]),
            "fp_total": sum(vals["fp"]),
            "fn_total": sum(vals["fn"]),
        }
        for group, vals in sorted(groups.items())
    }


def aggregate_metrics(projects: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "entities_overall": _aggregate_overall(projects, "entities_overall"),
        "relationships_overall": _aggregate_overall(projects, "relationships_overall"),
        "entities_by_label": _aggregate_by_group(projects, "entities_by_label"),
        "relationships_by_type": _aggregate_by_group(projects, "relationships_by_type"),
        "n_projects": len(projects),
    }


def evaluate_llm(display_name: str, pred_data: dict, gt_data: dict) -> Dict[str, Any]:
    print(f"  Evaluando {display_name}...")

    comparison = compare_all_projects(pred_data, gt_data, use_entity_id=False)
    agg = aggregate_metrics(comparison["projects"])

    schema_report = validate_graph_against_schema(pred_data)
    schema = schema_report.summary

    return {
        "llm": display_name,
        "n_projects": agg["n_projects"],
        "entities_overall": agg["entities_overall"],
        "relationships_overall": agg["relationships_overall"],
        "entities_by_label": agg["entities_by_label"],
        "relationships_by_type": agg["relationships_by_type"],
        "schema_score": schema["score"],
        "schema_verdict": schema["verdict"],
        "schema_errors": schema["error_count"],
        "schema_warnings": schema["warning_count"],
    }


PALETTE = [
    "#3498db",
    "#e74c3c",
    "#2ecc71",
    "#f39c12",
    "#9b59b6",
]


def _grouped_bar(
    ax: "plt.Axes",
    labels: List[str],
    series: List[Dict],
    title: str,
) -> None:
    n_groups = len(labels)
    n_series = len(series)
    bar_w = 0.8 / n_series
    x = np.arange(n_groups)

    for i, s in enumerate(series):
        offset = (i - n_series / 2 + 0.5) * bar_w
        ax.bar(x + offset, s["data"], bar_w, label=s["label"], color=PALETTE[i % len(PALETTE)])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def generate_charts(results: List[Dict[str, Any]], images_dir: Path) -> Dict[str, Path]:
    images_dir.mkdir(parents=True, exist_ok=True)

    all_labels = sorted({label for r in results for label in r.get("entities_by_label", {})})
    all_rel_types = sorted({rt for r in results for rt in r.get("relationships_by_type", {})})

    llm_names = [r["llm"] for r in results]

    series_overall = [
        {
            "label": r["llm"],
            "data": [
                r["entities_overall"]["f1_macro"],
                r["relationships_overall"]["f1_macro"],
            ],
        }
        for r in results
    ]
    series_entity = [
        {
            "label": r["llm"],
            "data": [r["entities_by_label"].get(l, {}).get("f1_macro", 0.0) for l in all_labels],
        }
        for r in results
    ]
    series_rel = [
        {
            "label": r["llm"],
            "data": [r["relationships_by_type"].get(rt, {}).get("f1_macro", 0.0) for rt in all_rel_types],
        }
        for r in results
    ]

    paths: Dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7, 4))
    _grouped_bar(ax, ["Entidades", "Relaciones"], series_overall, "F1 Macro Global por Modelo")
    fig.tight_layout()
    p = images_dir / "chart_overall.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["overall"] = p

    fig, ax = plt.subplots(figsize=(max(8, len(all_labels) * 1.1), 5))
    _grouped_bar(ax, all_labels, series_entity, "F1 por Tipo de Entidad")
    fig.tight_layout()
    p = images_dir / "chart_entities.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["entities"] = p

    fig, ax = plt.subplots(figsize=(max(8, len(all_rel_types) * 1.1), 5))
    _grouped_bar(ax, all_rel_types, series_rel, "F1 por Tipo de Relación")
    fig.tight_layout()
    p = images_dir / "chart_relations.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths["relations"] = p

    return paths


def generate_html_report(
    results: List[Dict[str, Any]],
    output_path: Path,
    chart_paths: Dict[str, Path],
) -> None:
    all_labels = sorted({
        label
        for r in results
        for label in r.get("entities_by_label", {})
    })
    all_rel_types = sorted({
        rt
        for r in results
        for rt in r.get("relationships_by_type", {})
    })

    def val_color(val: float) -> str:
        if val >= 0.7:
            return "#27ae60"
        if val >= 0.4:
            return "#e67e22"
        return "#e74c3c"

    def metric_cell(val: float) -> str:
        color = val_color(val)
        return (
            f'<td style="background:{color}18;color:{color};font-weight:600;">'
            f"{val:.3f}</td>"
        )

    def count_cell(val: int) -> str:
        return f"<td>{val}</td>"

    def schema_cell(score: int) -> str:
        color = val_color(score / 100)
        return f'<td style="color:{color};font-weight:600;">{score}/100</td>'

    def overall_rows() -> str:
        rows = []
        for r in results:
            eo = r["entities_overall"]
            ro = r["relationships_overall"]
            row = f"<tr><td class='llm-name'>{r['llm']}</td>"
            row += metric_cell(eo["precision_macro"])
            row += metric_cell(eo["recall_macro"])
            row += metric_cell(eo["f1_macro"])
            row += count_cell(eo["tp_total"])
            row += count_cell(eo["fp_total"])
            row += count_cell(eo["fn_total"])
            row += metric_cell(ro["precision_macro"])
            row += metric_cell(ro["recall_macro"])
            row += metric_cell(ro["f1_macro"])
            row += count_cell(ro["tp_total"])
            row += count_cell(ro["fp_total"])
            row += count_cell(ro["fn_total"])
            row += schema_cell(r["schema_score"])
            row += f"<td>{r['schema_errors']}</td>"
            row += f"<td>{r['schema_warnings']}</td>"
            row += "</tr>"
            rows.append(row)
        return "\n".join(rows)

    def detail_rows(group_key: str, groups: List[str]) -> str:
        rows = []
        for r in results:
            row = f"<tr><td class='llm-name'>{r['llm']}</td>"
            for g in groups:
                m = r.get(group_key, {}).get(g, {})
                row += metric_cell(m.get("precision_macro", 0.0))
                row += metric_cell(m.get("recall_macro", 0.0))
                row += metric_cell(m.get("f1_macro", 0.0))
                row += count_cell(m.get("tp_total", 0))
                row += count_cell(m.get("fp_total", 0))
                row += count_cell(m.get("fn_total", 0))
            row += "</tr>"
            rows.append(row)
        return "\n".join(rows)

    def group_headers(groups: List[str]) -> str:
        spans = "".join(
            f'<th colspan="6" style="background:#1a252f;">{g}</th>' for g in groups
        )
        sub = "".join(
            "<th>P</th><th>R</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th>" * 1
            for _ in groups
        )
        return spans, sub

    # image paths relative to the HTML file location
    img_overall = chart_paths["overall"].relative_to(output_path.parent)
    img_entities = chart_paths["entities"].relative_to(output_path.parent)
    img_relations = chart_paths["relations"].relative_to(output_path.parent)

    # Mini analysis
    best_entity = max(results, key=lambda r: r["entities_overall"]["f1_macro"])
    best_rel = max(results, key=lambda r: r["relationships_overall"]["f1_macro"])
    best_schema = max(results, key=lambda r: r["schema_score"])
    worst_entity = min(results, key=lambda r: r["entities_overall"]["f1_macro"])
    best_precision = max(results, key=lambda r: r["entities_overall"]["precision_macro"])
    best_recall = max(results, key=lambda r: r["entities_overall"]["recall_macro"])

    entity_level = "alta" if best_entity["entities_overall"]["f1_macro"] >= 0.7 else ("moderada" if best_entity["entities_overall"]["f1_macro"] >= 0.4 else "baja")
    rel_level = "alta" if best_rel["relationships_overall"]["f1_macro"] >= 0.7 else ("moderada" if best_rel["relationships_overall"]["f1_macro"] >= 0.4 else "baja")

    analysis_items = [
        f"<strong>{best_entity['llm']}</strong> obtiene el mejor F1 macro en entidades ({best_entity['entities_overall']['f1_macro']:.3f}).",
        f"<strong>{best_precision['llm']}</strong> tiene la mejor precisión en entidades ({best_precision['entities_overall']['precision_macro']:.3f}) y <strong>{best_recall['llm']}</strong> el mejor recall ({best_recall['entities_overall']['recall_macro']:.3f}).",
        f"<strong>{best_rel['llm']}</strong> obtiene el mejor F1 macro en relaciones ({best_rel['relationships_overall']['f1_macro']:.3f}).",
        f"<strong>{best_schema['llm']}</strong> genera el grafo más ajustado al esquema (score {best_schema['schema_score']}/100).",
        f"<strong>{worst_entity['llm']}</strong> tiene el rendimiento más bajo en entidades (F1 = {worst_entity['entities_overall']['f1_macro']:.3f}).",
        f"La capacidad de extracción de entidades es en general <strong>{entity_level}</strong> y la de relaciones <strong>{rel_level}</strong>.",
    ]
    analysis_html = "\n".join(f"<li>{item}</li>" for item in analysis_items)

    entity_group_spans, entity_group_sub = group_headers(all_labels)
    rel_group_spans, rel_group_sub = group_headers(all_rel_types)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <title>Evaluación LLM – Extracción KG</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0; padding: 2rem 3rem; background: #f0f2f5; color: #2c3e50;
    }}
    h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
    h2 {{ font-size: 1.2rem; color: #34495e; border-left: 4px solid #3498db; padding-left: 10px; margin-top: 2rem; }}
    p.subtitle {{ color: #666; margin-top: 0; }}
    .card {{
      background: white; border-radius: 10px; padding: 1.5rem;
      box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin: 1rem 0;
    }}
    .overflow-x {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 0.8rem; }}
    th {{
      background: #2c3e50; color: white; padding: 7px 10px;
      text-align: center; white-space: nowrap; font-size: 0.75rem;
    }}
    th:first-child {{ text-align: left; }}
    td {{ padding: 7px 10px; text-align: center; border-bottom: 1px solid #f0f0f0; }}
    td.llm-name {{ text-align: left; font-weight: 600; white-space: nowrap; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafbfc; }}
    .chart-wrap {{ max-width: 900px; }}
    .chart-wrap img {{ width: 100%; height: auto; border-radius: 6px; }}
    ul.analysis {{ line-height: 1.9; }}
    ul.analysis li {{ margin-bottom: 0.25rem; }}
  </style>
</head>
<body>
  <h1>Evaluación de Extracción – Comparación de LLMs</h1>
  <p class="subtitle">5 modelos comparados contra el ground truth de extracción (métricas macro-average sobre {results[0]['n_projects']} proyectos).</p>

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
            <th rowspan="2">Modelo</th>
            <th colspan="6" style="background:#1a252f;">Entidades (macro)</th>
            <th colspan="6" style="background:#1a252f;">Relaciones (macro)</th>
            <th rowspan="2">Schema</th>
            <th rowspan="2">Errores</th>
            <th rowspan="2">Warnings</th>
          </tr>
          <tr>
            <th>P</th><th>R</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th>
            <th>P</th><th>R</th><th>F1</th><th>TP</th><th>FP</th><th>FN</th>
          </tr>
        </thead>
        <tbody>
          {overall_rows()}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Por Tipo de Entidad</h2>
    <div class="overflow-x">
      <table>
        <thead>
          <tr>
            <th rowspan="2">Modelo</th>
            {entity_group_spans}
          </tr>
          <tr>{entity_group_sub}</tr>
        </thead>
        <tbody>
          {detail_rows("entities_by_label", all_labels)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>Por Tipo de Relación</h2>
    <div class="overflow-x">
      <table>
        <thead>
          <tr>
            <th rowspan="2">Modelo</th>
            {rel_group_spans}
          </tr>
          <tr>{rel_group_sub}</tr>
        </thead>
        <tbody>
          {detail_rows("relationships_by_type", all_rel_types)}
        </tbody>
      </table>
    </div>
  </div>

  <div class="card">
    <h2>F1 Macro Global por Modelo</h2>
    <div class="chart-wrap">
      <img src="{img_overall}" alt="F1 Macro Global">
    </div>
  </div>

  <div class="card">
    <h2>F1 por Tipo de Entidad</h2>
    <div class="chart-wrap">
      <img src="{img_entities}" alt="F1 por Tipo de Entidad">
    </div>
  </div>

  <div class="card">
    <h2>F1 por Tipo de Relación</h2>
    <div class="chart-wrap">
      <img src="{img_relations}" alt="F1 por Tipo de Relación">
    </div>
  </div>

</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main() -> None:
    gt_data = load_json(GT_PATH)

    llm_dirs = sorted([
        d for d in LLM_EXTRACTION_DIR.iterdir()
        if d.is_dir() and (d / "export_graph.json").exists()
    ])

    if not llm_dirs:
        print(f"No se encontraron directorios con export_graph.json en {LLM_EXTRACTION_DIR}")
        sys.exit(1)

    print(f"Encontrados {len(llm_dirs)} modelos para evaluar.\n")

    results: List[Dict[str, Any]] = []
    for llm_dir in llm_dirs:
        display_name = LLM_DISPLAY_NAMES.get(llm_dir.name, llm_dir.name)
        pred_data = load_json(llm_dir / "export_graph.json")
        result = evaluate_llm(display_name, pred_data, gt_data)
        results.append(result)

    results.sort(key=lambda r: r["entities_overall"]["f1_macro"], reverse=True)

    # Print summary table
    print("\n" + "=" * 82)
    print(f"{'Modelo':<25} {'Ent P':>7} {'Ent R':>7} {'Ent F1':>7} {'Rel P':>7} {'Rel R':>7} {'Rel F1':>7} {'Schema':>8}")
    print("=" * 82)
    for r in results:
        eo = r["entities_overall"]
        ro = r["relationships_overall"]
        print(
            f"{r['llm']:<25}"
            f" {eo['precision_macro']:>7.3f} {eo['recall_macro']:>7.3f} {eo['f1_macro']:>7.3f}"
            f" {ro['precision_macro']:>7.3f} {ro['recall_macro']:>7.3f} {ro['f1_macro']:>7.3f}"
            f" {r['schema_score']:>6}/100"
        )
    print("=" * 82)

    summary_path = RESULTS_PATH / "llm_comparison_summary.json"
    save_json(results, summary_path)
    print(f"\nResumen JSON guardado en: {summary_path}")

    images_dir = RESULTS_PATH / "images"
    chart_paths = generate_charts(results, images_dir)
    print(f"Gráficas guardadas en:    {images_dir}")

    html_path = RESULTS_PATH / "llm_comparison_report.html"
    generate_html_report(results, html_path, chart_paths)
    print(f"Reporte HTML guardado en:  {html_path}")


if __name__ == "__main__":
    main()
