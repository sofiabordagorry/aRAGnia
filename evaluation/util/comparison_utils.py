from typing import Any, Dict, Set, Tuple


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        items = sorted(
            (str(k).strip().lower(), normalize_text(v))
            for k, v in value.items()
        )
        return str(items)
    if isinstance(value, list):
        return str([normalize_text(v) for v in value])
    return str(value).strip().lower()


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(pred_set: Set[Tuple], gt_set: Set[Tuple]) -> Dict[str, Any]:
    tp = len(pred_set & gt_set)
    fp = len(pred_set - gt_set)
    fn = len(gt_set - pred_set)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pred_count": len(pred_set),
        "gt_count": len(gt_set),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def entity_key(entity: dict, use_id: bool = False) -> Tuple:
    label = entity.get("label", "").strip()

    if use_id:
        return (label, entity.get("id", "").strip())

    # Entidades estáticas (rule-based): se identifican por ID, no por valor
    if label in {"Proyecto", "Documento", "Chunk"}:
        return (label, entity.get("id", "").strip())

    value = entity.get("value")

    if isinstance(value, dict):
        if label == "Investigador":
            return (label, normalize_text(value.get("name", "")))

        if label == "Anio":
            return (label, normalize_text(value.get("year", "")))

    return (label, normalize_text(value))


def relationship_key(rel: dict) -> Tuple:
    return (
        rel.get("type", "").strip(),
        rel.get("source_id", "").strip(),
        rel.get("target_id", "").strip(),
    )
