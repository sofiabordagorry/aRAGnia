from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple
import json
from util.comparison_utils import (
    compute_metrics,
    entity_key,
    relationship_key,
)


def _get_project_entities(entities: List[dict]) -> Dict[str, dict]:
    return {
        e["id"]: e
        for e in entities
        if e.get("label") == "Proyecto" and "id" in e
    }


def _get_chunk_entities(entities: List[dict]) -> Dict[str, dict]:
    return {
        e["id"]: e
        for e in entities
        if e.get("label") == "Chunk" and "id" in e
    }


class ProjectSubgraphBuilder:
    def __init__(self, data: dict):
        self.entities = data.get("entities", [])
        self.relationships = data.get("relationships", [])
        self.project_ids = set(_get_project_entities(self.entities).keys())
        self.chunk_ids = set(_get_chunk_entities(self.entities).keys())
        self._build_entity_to_projects_map()

    def _build_entity_to_projects_map(self) -> None:
        entity_to_projects: Dict[str, Set[str]] = defaultdict(set)

        for pid in self.project_ids:
            entity_to_projects[pid].add(pid)

        for r in self.relationships:
            s = r.get("source_id")
            t = r.get("target_id")

            if not s or not t:
                continue

            if s in self.project_ids and t not in self.chunk_ids:
                entity_to_projects[s].add(s)
                entity_to_projects[t].add(s)

            if t in self.project_ids and s not in self.chunk_ids:
                entity_to_projects[t].add(t)
                entity_to_projects[s].add(t)

        self.entity_to_projects = entity_to_projects

    def collect_project_subgraph(
        self,
        project_id: str,
        use_entity_id: bool = False,
    ) -> Dict[str, Any]:
        if not project_id:
            return {}
        table_document_ids = {
            e["id"]
            for e in self.entities
            if (
                e.get("label") == "Documento"
                and e.get("value", {}).get("tipo") == "tabla"
                and "id" in e
            )
        }
        project_entities = []
        for e in self.entities:
            eid = e.get("id")
            if eid in table_document_ids:
                continue
            if eid and project_id in self.entity_to_projects.get(eid, set()):
                project_entities.append(e)

        project_relationships = []
        for r in self.relationships:
            s = r.get("source_id")
            t = r.get("target_id")
            if s in table_document_ids or t in table_document_ids:
                continue
            if (
                (s == project_id or t == project_id)
                and not (s in self.chunk_ids or t in self.chunk_ids)
            ):
                project_relationships.append(r)
        entity_set = {entity_key(e, use_id=use_entity_id) for e in project_entities}
        rel_set = {relationship_key(r) for r in project_relationships}

        entities_by_label: Dict[str, Set[Tuple]] = defaultdict(set)
        for e in project_entities:
            entities_by_label[e.get("label", "")].add(
                entity_key(e, use_id=use_entity_id)
            )

        rels_by_type: Dict[str, Set[Tuple]] = defaultdict(set)
        for r in project_relationships:
            rels_by_type[r.get("type", "")].add(relationship_key(r))

        return {
            "project_entities_raw": project_entities,
            "project_relationships_raw": project_relationships,
            "entity_set": entity_set,
            "rel_set": rel_set,
            "entities_by_label": entities_by_label,
            "rels_by_type": rels_by_type,
        }


def compare_project(
    pred_data: dict,
    gt_data: dict,
    project_id: str,
    use_entity_id: bool = False,
) -> Dict[str, Any]:
    pred_graph = ProjectSubgraphBuilder(pred_data)
    gt_graph = ProjectSubgraphBuilder(gt_data)

    pred_sub = pred_graph.collect_project_subgraph(
        project_id,
        use_entity_id=use_entity_id,
    )
    gt_sub = gt_graph.collect_project_subgraph(
        project_id,
        use_entity_id=use_entity_id,
    )

    entities_overall = compute_metrics(pred_sub["entity_set"], gt_sub["entity_set"])
    relationships_overall = compute_metrics(pred_sub["rel_set"], gt_sub["rel_set"])

    all_entity_labels = sorted(
        set(pred_sub["entities_by_label"].keys())
        | set(gt_sub["entities_by_label"].keys())
    )

    entities_by_label = {}
    for label in all_entity_labels:
        entities_by_label[label] = compute_metrics(
            pred_sub["entities_by_label"].get(label, set()),
            gt_sub["entities_by_label"].get(label, set()),
        )

    all_rel_types = sorted(
        set(pred_sub["rels_by_type"].keys()) | set(gt_sub["rels_by_type"].keys())
    )

    relationships_by_type = {}
    for rel_type in all_rel_types:
        relationships_by_type[rel_type] = compute_metrics(
            pred_sub["rels_by_type"].get(rel_type, set()),
            gt_sub["rels_by_type"].get(rel_type, set()),
        )

    return {
        "project_id": project_id,
        "entities_overall": entities_overall,
        "relationships_overall": relationships_overall,
        "entities_by_label": entities_by_label,
        "relationships_by_type": relationships_by_type,
        "pred_sub": pred_sub,
        "gt_sub": gt_sub,
    }


def compare_all_projects(
    pred_data: dict,
    gt_data: dict,
    use_entity_id: bool = False,
) -> Dict[str, Any]:
    pred_projects = _get_project_entities(pred_data.get("entities", []))
    gt_projects = _get_project_entities(gt_data.get("entities", []))

    all_project_ids = sorted(set(pred_projects.keys()) | set(gt_projects.keys()))

    report = {
        "config": {
            "entity_comparison": "label+id" if use_entity_id else "label+normalized_value",
            "relationship_comparison": "type+source_id+target_id",
        },
        "projects": {},
    }

    for project_id in all_project_ids:
        report["projects"][project_id] = compare_project(
            pred_data,
            gt_data,
            project_id,
            use_entity_id=use_entity_id,
        )

    return report



def metric_row(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": name,
        "cls": "good" if metrics["f1"] > 0.7 else "bad",
        **metrics,
    }


def build_entity_groups(
    pred_sub: Dict[str, Any],
    gt_sub: Dict[str, Any],
    use_entity_id: bool,
) -> List[Dict[str, Any]]:
    pred_entities_map = {
        entity_key(e, use_id=use_entity_id): e
        for e in pred_sub["project_entities_raw"]
    }
    gt_entities_map = {
        entity_key(e, use_id=use_entity_id): e
        for e in gt_sub["project_entities_raw"]
    }

    all_entity_keys = sorted(
        set(pred_entities_map.keys()) | set(gt_entities_map.keys()),
        key=lambda x: tuple(str(i) for i in x),
    )

    grouped_entities = defaultdict(list)

    for k in all_entity_keys:
        pred_e = pred_entities_map.get(k)
        gt_e = gt_entities_map.get(k)

        if pred_e and gt_e:
            cls = "match"
            status = "MATCH"
        elif gt_e and not pred_e:
            cls = "missing_pred"
            status = "FALTA EN PRED"
        else:
            cls = "missing_gt"
            status = "SOBRA EN PRED"

        label = k[0] if isinstance(k, tuple) and len(k) > 0 else ""
        comparable = str(k[1:]) if isinstance(k, tuple) and len(k) > 1 else str(k)

        pred_txt = json.dumps(pred_e, ensure_ascii=False, indent=2) if pred_e else ""
        gt_txt = json.dumps(gt_e, ensure_ascii=False, indent=2) if gt_e else ""

        grouped_entities[label].append(
            {
                "cls": cls,
                "status": status,
                "label": label,
                "comparable": comparable,
                "pred_txt": pred_txt,
                "gt_txt": gt_txt,
            }
        )

    result = []
    for label in sorted(grouped_entities.keys()):
        rows = grouped_entities[label]
        result.append(
            {
                "group_name": label,
                "count": len(rows),
                "n_match": sum(1 for r in rows if r["status"] == "MATCH"),
                "n_missing_pred": sum(
                    1 for r in rows if r["status"] == "FALTA EN PRED"
                ),
                "n_missing_gt": sum(
                    1 for r in rows if r["status"] == "SOBRA EN PRED"
                ),
                "open_attr": "" if label == "Chunk" else " open",
                "rows": rows,
            }
        )

    return result


def build_relationship_groups(
    pred_sub: Dict[str, Any],
    gt_sub: Dict[str, Any],
) -> List[Dict[str, Any]]:
    pred_rels_map = {
        relationship_key(r): r
        for r in pred_sub["project_relationships_raw"]
    }
    gt_rels_map = {
        relationship_key(r): r
        for r in gt_sub["project_relationships_raw"]
    }

    all_rel_keys = sorted(
        set(pred_rels_map.keys()) | set(gt_rels_map.keys()),
        key=lambda x: (str(x[0]), str(x[1]), str(x[2])),
    )

    grouped_rels = defaultdict(list)

    for k in all_rel_keys:
        pred_r = pred_rels_map.get(k)
        gt_r = gt_rels_map.get(k)

        if pred_r and gt_r:
            cls = "match"
            status = "MATCH"
        elif gt_r and not pred_r:
            cls = "missing_pred"
            status = "FALTA EN PRED"
        else:
            cls = "missing_gt"
            status = "SOBRA EN PRED"

        rel_type = k[0] if isinstance(k, tuple) and len(k) > 0 else ""
        comparable = str(k)

        pred_txt = json.dumps(pred_r, ensure_ascii=False, indent=2) if pred_r else ""
        gt_txt = json.dumps(gt_r, ensure_ascii=False, indent=2) if gt_r else ""

        grouped_rels[rel_type].append(
            {
                "cls": cls,
                "status": status,
                "rel_type": rel_type,
                "comparable": comparable,
                "pred_txt": pred_txt,
                "gt_txt": gt_txt,
            }
        )

    result = []
    for rel_type in sorted(grouped_rels.keys()):
        rows = grouped_rels[rel_type]
        result.append(
            {
                "group_name": rel_type,
                "count": len(rows),
                "n_match": sum(1 for r in rows if r["status"] == "MATCH"),
                "n_missing_pred": sum(
                    1 for r in rows if r["status"] == "FALTA EN PRED"
                ),
                "n_missing_gt": sum(
                    1 for r in rows if r["status"] == "SOBRA EN PRED"
                ),
                "rows": rows,
            }
        )

    return result


def build_rendered_projects(
    pred_data: dict,
    gt_data: dict,
    use_entity_id: bool = False,
) -> List[Dict[str, Any]]:
    pred_projects = _get_project_entities(pred_data.get("entities", []))
    gt_projects = _get_project_entities(gt_data.get("entities", []))
    all_project_ids = sorted(set(pred_projects.keys()) | set(gt_projects.keys()))

    rendered_projects = []

    for project_id in all_project_ids:
        project_report = compare_project(
            pred_data,
            gt_data,
            project_id,
            use_entity_id=use_entity_id,
        )

        entities_by_label_rows = [
            metric_row(label, metrics)
            for label, metrics in project_report["entities_by_label"].items()
        ]

        relationships_by_type_rows = [
            metric_row(rel_type, metrics)
            for rel_type, metrics in project_report["relationships_by_type"].items()
        ]

        rendered_projects.append(
            {
                "project_id": project_id,
                "entities_overall": project_report["entities_overall"],
                "relationships_overall": project_report["relationships_overall"],
                "entities_by_label": entities_by_label_rows,
                "relationships_by_type": relationships_by_type_rows,
                "entity_groups": build_entity_groups(
                    project_report["pred_sub"],
                    project_report["gt_sub"],
                    use_entity_id,
                ),
                "relationship_groups": build_relationship_groups(
                    project_report["pred_sub"],
                    project_report["gt_sub"],
                ),
            }
        )

    return rendered_projects