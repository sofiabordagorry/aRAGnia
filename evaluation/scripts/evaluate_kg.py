import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from util.html_report import write_project_side_by_side_html, render_html_report
from util.graph_compare import build_rendered_projects
from util.graph_validation import validate_graph_against_schema
from typing import Any, Dict


# Cargar variables de entorno

PRED_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "entities_relations"
    / "entity_documents.json"
)
GT_PATH = (
    Path(__file__).parents[1]
    / "ground_truth"
    / "extraction"
    / "ground_truth_kg.json"
)
RESULTS_PATH = (
    Path(__file__).parents[2]
    / "evaluation"
    / "results_knowledge_graph"
)

def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data: Dict[str, Any], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    gt_path = GT_PATH
    pred_path = PRED_PATH

    output_html = RESULTS_PATH / "report_complete.html"
    Path(output_html).parent.mkdir(parents=True, exist_ok=True)
    output_json = RESULTS_PATH / "report_render_data.json"

    pred_data = load_json(pred_path)
    gt_data = load_json(gt_path)

    rendered_projects = build_rendered_projects(
        pred_data,
        gt_data,
        use_entity_id=False,
    )
    save_json(rendered_projects, output_json)

    write_project_side_by_side_html(
        rendered_projects=rendered_projects,
        path=output_html,
    )

    print(f"Reporte HTML completo guardado en: {output_html}")
    print(f"Datos intermedios guardados en JSON en: {output_json}")

    # Verificar Estabilidad del grafo.
    report = validate_graph_against_schema(pred_data)

    output_html_metrics = RESULTS_PATH/ "report_structure_graph.json"
    output_html_side_by_side = RESULTS_PATH/ "report_structure_graph.html"

    save_json(report.to_dict(), output_html_metrics)
    render_html_report(report, output_html_side_by_side)

    print(f"Score: {report.summary['score']}/100")
    print(f"Veredicto: {report.summary['verdict']}")
    print(f"Errores: {report.summary['error_count']}")
    print(f"Warnings: {report.summary['warning_count']}")