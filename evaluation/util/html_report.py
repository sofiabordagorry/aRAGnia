import html as html_lib
from typing import Any, Dict, List
from dataclasses import dataclass, field
import json
from pathlib import Path

@dataclass
class Issue:
    severity: str  # error | warning
    category: str  # structure | value | constraint
    rule: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)



@dataclass
class ValidationReport:
    summary: Dict[str, Any]
    issues: List[Issue]
    stats: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary,
            "issues": [
                {
                    "severity": i.severity,
                    "category": i.category,
                    "rule": i.rule,
                    "message": i.message,
                    "context": i.context,
                }
                for i in self.issues
            ],
            "stats": self.stats,
        }


def esc(text: Any) -> str:
    return html_lib.escape(str(text))

def write_project_side_by_side_html(
    rendered_projects: List[Dict[str, Any]],
    path: str,
) -> None:
    html = """
    <html>
    <head>
    <meta charset="utf-8">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1, h2, h3, h4 { margin-top: 28px; }
        table { border-collapse: collapse; width: 100%; margin-bottom: 20px; table-layout: fixed; }
        th, td { border: 1px solid #ccc; padding: 8px; vertical-align: top; }
        th { background: #f2f2f2; }
        pre { white-space: pre-wrap; word-wrap: break-word; margin: 0; font-size: 12px; }

        .match { background-color: #ddffdd; }
        .missing_pred { background-color: #ffe4b5; }
        .missing_gt { background-color: #ffd6d6; }

        .good { background-color: #ddffdd; }
        .bad { background-color: #ffdddd; }

        .small { color: #666; font-size: 12px; }

        details {
            border: 1px solid #ddd;
            border-radius: 8px;
            margin-bottom: 12px;
            padding: 8px 12px;
            background: #fafafa;
        }

        summary {
            cursor: pointer;
            font-weight: bold;
            font-size: 15px;
            outline: none;
        }

        .badge {
            display: inline-block;
            padding: 2px 8px;
            margin-left: 8px;
            border-radius: 999px;
            font-size: 12px;
            background: #eee;
        }

        .badge-green { background: #dff5df; }
        .badge-orange { background: #ffe7bf; }
        .badge-red { background: #ffd9d9; }

        .section-box {
            border: 1px solid #ddd;
            border-radius: 10px;
            padding: 14px;
            margin-bottom: 20px;
            background: #fcfcfc;
        }

        .metrics-table {
            table-layout: auto;
        }
    </style>
    </head>
    <body>
    <h1>Reporte completo por proyecto</h1>
    <p class="small">
        Este reporte incluye:
        <br>- métricas resumidas por proyecto
        <br>- comparación lado a lado de entidades y relaciones
        <br><br>
        Colores en comparación lado a lado:
        <br>- Verde = coincide en ambos
        <br>- Naranja = está en GT pero no en pred
        <br>- Rojo = está en pred pero no en GT
    </p>
    """

    for project in rendered_projects:
        project_id = project["project_id"]
        ent_overall = project["entities_overall"]
        rel_overall = project["relationships_overall"]
        entities_by_label = project["entities_by_label"]
        relationships_by_type = project["relationships_by_type"]
        entity_groups = project["entity_groups"]
        relationship_groups = project["relationship_groups"]

        html += f"<h2>Proyecto: {esc(project_id)}</h2>"

        html += '<div class="section-box">'
        html += "<h3>Resumen de métricas</h3>"

        html += "<h4>Resumen global</h4>"
        html += """
        <table class="metrics-table">
            <tr>
                <th>Sección</th>
                <th>Pred</th>
                <th>GT</th>
                <th>TP</th>
                <th>FN</th>
                <th>FP</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
            </tr>
        """
        html += f"""
        <tr class="{'good' if ent_overall['f1'] > 0.7 else 'bad'}">
            <td>Entidades</td>
            <td>{ent_overall['pred_count']}</td>
            <td>{ent_overall['gt_count']}</td>
            <td>{ent_overall['tp']}</td>
            <td>{ent_overall['fn']}</td>
            <td>{ent_overall['fp']}</td>
            <td>{ent_overall['precision']:.2f}</td>
            <td>{ent_overall['recall']:.2f}</td>
            <td>{ent_overall['f1']:.2f}</td>
        </tr>
        """
        html += f"""
        <tr class="{'good' if rel_overall['f1'] > 0.7 else 'bad'}">
            <td>Relaciones</td>
            <td>{rel_overall['pred_count']}</td>
            <td>{rel_overall['gt_count']}</td>
            <td>{rel_overall['tp']}</td>
            <td>{rel_overall['fn']}</td>
            <td>{rel_overall['fp']}</td>
            <td>{rel_overall['precision']:.2f}</td>
            <td>{rel_overall['recall']:.2f}</td>
            <td>{rel_overall['f1']:.2f}</td>
        </tr>
        """
        html += "</table>"

        html += "<h4>Entidades por label</h4>"
        html += """
        <table class="metrics-table">
            <tr>
                <th>Tipo</th>
                <th>Pred</th>
                <th>GT</th>
                <th>TP</th>
                <th>FN</th>
                <th>FP</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
            </tr>
        """
        for row in entities_by_label:
            html += f"""
            <tr class="{row['cls']}">
                <td>{esc(row['name'])}</td>
                <td>{row['pred_count']}</td>
                <td>{row['gt_count']}</td>
                <td>{row['tp']}</td>
                <td>{row['fn']}</td>
                <td>{row['fp']}</td>
                <td>{row['precision']:.2f}</td>
                <td>{row['recall']:.2f}</td>
                <td>{row['f1']:.2f}</td>
            </tr>
            """
        html += "</table>"

        html += "<h4>Relaciones por tipo</h4>"
        html += """
        <table class="metrics-table">
            <tr>
                <th>Tipo</th>
                <th>Pred</th>
                <th>GT</th>
                <th>TP</th>
                <th>FN</th>
                <th>FP</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1</th>
            </tr>
        """
        for row in relationships_by_type:
            html += f"""
            <tr class="{row['cls']}">
                <td>{esc(row['name'])}</td>
                <td>{row['pred_count']}</td>
                <td>{row['gt_count']}</td>
                <td>{row['tp']}</td>
                <td>{row['fn']}</td>
                <td>{row['fp']}</td>
                <td>{row['precision']:.2f}</td>
                <td>{row['recall']:.2f}</td>
                <td>{row['f1']:.2f}</td>
            </tr>
            """
        html += "</table>"
        html += "</div>"

        html += '<div class="section-box">'
        html += "<h3>Comparación lado a lado</h3>"

        html += "<h4>Entidades</h4>"
        for group in entity_groups:
            html += f"""
            <details{group['open_attr']}>
                <summary>
                    {esc(group['group_name'])} ({group['count']})
                    <span class="badge badge-green">MATCH: {group['n_match']}</span>
                    <span class="badge badge-orange">FALTA EN PRED: {group['n_missing_pred']}</span>
                    <span class="badge badge-red">SOBRA EN PRED: {group['n_missing_gt']}</span>
                </summary>
                <table>
                    <tr>
                        <th style="width: 8%;">Estado</th>
                        <th style="width: 16%;">Label</th>
                        <th style="width: 18%;">Clave comparable</th>
                        <th style="width: 29%;">Pred</th>
                        <th style="width: 29%;">GT</th>
                    </tr>
            """
            for row in group["rows"]:
                html += f"""
                <tr class="{row['cls']}">
                    <td>{esc(row['status'])}</td>
                    <td>{esc(row['label'])}</td>
                    <td><pre>{esc(row['comparable'])}</pre></td>
                    <td><pre>{esc(row['pred_txt'])}</pre></td>
                    <td><pre>{esc(row['gt_txt'])}</pre></td>
                </tr>
                """
            html += "</table></details>"

        html += "<h4>Relaciones</h4>"
        for group in relationship_groups:
            html += f"""
            <details>
                <summary>
                    {esc(group['group_name'])} ({group['count']})
                    <span class="badge badge-green">MATCH: {group['n_match']}</span>
                    <span class="badge badge-orange">FALTA EN PRED: {group['n_missing_pred']}</span>
                    <span class="badge badge-red">SOBRA EN PRED: {group['n_missing_gt']}</span>
                </summary>
                <table>
                    <tr>
                        <th style="width: 8%;">Estado</th>
                        <th style="width: 16%;">Tipo</th>
                        <th style="width: 18%;">Clave comparable</th>
                        <th style="width: 29%;">Pred</th>
                        <th style="width: 29%;">GT</th>
                    </tr>
            """
            for row in group["rows"]:
                html += f"""
                <tr class="{row['cls']}">
                    <td>{esc(row['status'])}</td>
                    <td>{esc(row['rel_type'])}</td>
                    <td><pre>{esc(row['comparable'])}</pre></td>
                    <td><pre>{esc(row['pred_txt'])}</pre></td>
                    <td><pre>{esc(row['gt_txt'])}</pre></td>
                </tr>
                """
            html += "</table></details>"

        html += "</div>"

    html += "</body></html>"

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)



# =========================================================
# HTML
# =========================================================

def render_html_report(report: ValidationReport, output_path: str | Path) -> None:
    data = report.to_dict()
    summary = data["summary"]
    stats = data["stats"]
    issues = data["issues"]

    def esc(x: Any) -> str:
        return html_lib.escape(json.dumps(x, ensure_ascii=False, indent=2) if isinstance(x, (dict, list)) else str(x))

    score = summary["score"]
    if score >= 95:
        badge = "#1f7a1f"
    elif score >= 80:
        badge = "#2d7dd2"
    elif score >= 60:
        badge = "#c98910"
    else:
        badge = "#b42318"

    rows = []
    for i in issues:
        sev_color = "#b42318" if i["severity"] == "error" else "#c98910"
        rows.append(f"""
        <tr>
            <td><span class="pill" style="background:{sev_color}">{esc(i["severity"])}</span></td>
            <td>{esc(i["category"])}</td>
            <td><code>{esc(i["rule"])}</code></td>
            <td>{esc(i["message"])}</td>
            <td><pre>{esc(i["context"])}</pre></td>
        </tr>
        """)

    issues_html = "\n".join(rows) if rows else """
    <tr>
        <td colspan="5" style="text-align:center;">No se encontraron problemas 🎉</td>
    </tr>
    """

    html_text = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<title>Reporte de validación de grafo</title>
<style>
    body {{
        font-family: Arial, Helvetica, sans-serif;
        margin: 0;
        background: #f7f8fa;
        color: #222;
    }}
    .container {{
        max-width: 1200px;
        margin: 32px auto;
        padding: 0 20px;
    }}
    .header {{
        background: white;
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }}
    .score {{
        display: inline-block;
        padding: 10px 18px;
        border-radius: 999px;
        color: white;
        font-weight: bold;
        background: {badge};
        font-size: 20px;
    }}
    .grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 20px;
    }}
    .card {{
        background: white;
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    .card h3 {{
        margin-top: 0;
        font-size: 15px;
        color: #555;
    }}
    .big {{
        font-size: 28px;
        font-weight: bold;
        margin-top: 8px;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        background: white;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
    }}
    th, td {{
        padding: 12px;
        border-bottom: 1px solid #eee;
        vertical-align: top;
        text-align: left;
    }}
    th {{
        background: #f0f3f7;
    }}
    .pill {{
        display: inline-block;
        color: white;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: bold;
    }}
    pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 12px;
    }}
    .section-title {{
        margin: 26px 0 12px 0;
    }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Reporte de validación de grafo</h1>
        <p><strong>Veredicto:</strong> {esc(summary["verdict"])}</p>
        <div class="score">Score: {esc(summary["score"])}/100</div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>Errores</h3>
            <div class="big">{esc(summary["error_count"])}</div>
        </div>
        <div class="card">
            <h3>Warnings</h3>
            <div class="big">{esc(summary["warning_count"])}</div>
        </div>
        <div class="card">
            <h3>Entidades</h3>
            <div class="big">{esc(stats["entity_count"])}</div>
        </div>
        <div class="card">
            <h3>Relaciones</h3>
            <div class="big">{esc(stats["relationship_count"])}</div>
        </div>
    </div>

    <h2 class="section-title">Distribución de entidades</h2>
    <table>
        <thead>
            <tr><th>Label</th><th>Cantidad</th></tr>
        </thead>
        <tbody>
            {''.join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in stats["entities_by_label"].items())}
        </tbody>
    </table>

    <h2 class="section-title">Distribución de relaciones</h2>
    <table>
        <thead>
            <tr><th>Tipo</th><th>Cantidad</th></tr>
        </thead>
        <tbody>
            {''.join(f"<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>" for k, v in stats["relationships_by_type"].items())}
        </tbody>
    </table>

    <h2 class="section-title">Problemas detectados</h2>
    <table>
        <thead>
            <tr>
                <th>Severidad</th>
                <th>Categoría</th>
                <th>Regla</th>
                <th>Mensaje</th>
                <th>Contexto</th>
            </tr>
        </thead>
        <tbody>
            {issues_html}
        </tbody>
    </table>
</div>
</body>
</html>
"""

    Path(output_path).write_text(html_text, encoding="utf-8")
