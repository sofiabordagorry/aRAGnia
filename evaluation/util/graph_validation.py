from __future__ import annotations

import json
import html
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter

from pathlib import Path
from dotenv import load_dotenv

from util.html_report import render_html_report

# Cargar las variables de entorno
env_path = Path(__file__).parents[1] / ".env"
load_dotenv(env_path)

# Ruta al archivo JSON
JSON_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "ground_truth"
    / "extraction" / "gt_proyects.json"
)

# =========================================================
# ESQUEMA
# =========================================================

ALLOWED_ENTITY_LABELS = {
    "Proyecto",
    "Anio",
    "Investigador",
    "Topico",
    "Dominio",
    "Documento",
    "Chunk",
}

ALLOWED_REL_TYPES = {
    "PARTICIPO_EN",
    "RESPONSABLE_DE",
    "TIENE_TOPICO",
    "PERTENECE_A_DOMINIO",
    "ES_DESCRITO_POR",
    "INICIO_EN",
    "PRIMER_CHUNK",
    "SIGUIENTE_CHUNK",
    "DE_DOCUMENTO",
    "EXTRAIDO_DE",
    "TITULO_EXTRAIDO_DE",
    "POSIBLE_ALIAS",
}

VALID_REL_ENDPOINTS = {
    "PARTICIPO_EN": ("Investigador", "Proyecto"),
    "RESPONSABLE_DE": ("Investigador", "Proyecto"),
    "TIENE_TOPICO": ("Proyecto", "Topico"),
    "PERTENECE_A_DOMINIO": ("Topico", "Dominio"),
    "ES_DESCRITO_POR": ("Proyecto", "Documento"),
    "INICIO_EN": ("Proyecto", "Anio"),
    "PRIMER_CHUNK": ("Documento", "Chunk"),
    "SIGUIENTE_CHUNK": ("Chunk", "Chunk"),
    "DE_DOCUMENTO": ("Chunk", "Documento"),
    "EXTRAIDO_DE": ("Chunk", {"Topico", "Investigador"}),
    "POSIBLE_ALIAS": ("Investigador", "Investigador"),
    "TITULO_EXTRAIDO_DE": ("Proyecto", "Chunk"),
}

ALLOWED_DOCUMENT_TYPES = {"informe", "propuesta", "resumen", "tabla"}
ALLOWED_INVESTIGADOR_SOURCES = {"rule_based", "llm","human_annotation"}


# =========================================================
# MODELOS DE REPORTE
# =========================================================

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

def ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def is_non_empty_str(x: Any) -> bool:
    return isinstance(x, str) and bool(x.strip())


def add_issue(
    issues: List[Issue],
    severity: str,
    category: str,
    rule: str,
    message: str,
    **context: Any,
) -> None:
    issues.append(
        Issue(
            severity=severity,
            category=category,
            rule=rule,
            message=message,
            context=context,
        )
    )


# =========================================================
# VALIDACIÓN DE VALORES
# =========================================================

def validate_entity_value(entity: Dict[str, Any], issues: List[Issue]) -> None:
    eid = str(entity.get("id", ""))
    label = str(entity.get("label", ""))
    value = entity.get("value")

    if label == "Proyecto":
        if not is_non_empty_str(value):
            add_issue(
                issues, "error", "value", "Proyecto.value",
                "Proyecto.value debe ser string no vacío.",
                entity_id=eid,
                label=label,
                value=value,
            )

    elif label == "Topico":
        if not is_non_empty_str(value):
            add_issue(
                issues, "error", "value", "Topico.value",
                "Topico.value debe ser string no vacío.",
                entity_id=eid,
                label=label,
                value=value,
            )

    elif label == "Dominio":
        if not is_non_empty_str(value):
            add_issue(
                issues, "error", "value", "Dominio.value",
                "Dominio.value debe ser string no vacío.",
                entity_id=eid,
                label=label,
                value=value,
            )

    elif label == "Chunk":
        # Chunk hereda Entity y no impone estructura fuerte en value.
        # Permitimos cualquier value.
        pass

    elif label == "Anio":
        if not isinstance(value, dict):
            add_issue(
                issues, "error", "value", "Anio.value",
                "Anio.value debe ser dict con campo 'year'.",
                entity_id=eid,
                label=label,
                value=value,
            )
        else:
            year = value.get("year")
            if not is_non_empty_str(year):
                add_issue(
                    issues, "error", "value", "Anio.value.year",
                    "Anio.value.year debe ser string no vacío.",
                    entity_id=eid,
                    label=label,
                    value=value,
                )

    elif label == "Documento":
        if not isinstance(value, dict):
            add_issue(
                issues, "error", "value", "Documento.value",
                "Documento.value debe ser dict.",
                entity_id=eid,
                label=label,
                value=value,
            )
        else:
            required = ["base_name", "is_group", "year_publisher", "type"]
            for key in required:
                if key not in value:
                    add_issue(
                        issues, "error", "value", f"Documento.value.{key}",
                        f"Falta el campo requerido '{key}' en Documento.value.",
                        entity_id=eid,
                        label=label,
                        value=value,
                    )
            doc_type = value.get("type")
            if doc_type not in ALLOWED_DOCUMENT_TYPES:
                add_issue(
                    issues, "error", "value", "Documento.value.type",
                    f"Documento.value.type debe ser uno de {sorted(ALLOWED_DOCUMENT_TYPES)}.",
                    entity_id=eid,
                    label=label,
                    value=value,
                )

    elif label == "Investigador":
        if not isinstance(value, dict):
            add_issue(
                issues, "error", "value", "Investigador.value",
                "Investigador.value debe ser dict con campos 'name' y 'source'.",
                entity_id=eid,
                label=label,
                value=value,
            )
        else:
            if not is_non_empty_str(value.get("name")):
                add_issue(
                    issues, "error", "value", "Investigador.value.name",
                    "Investigador.value.name debe ser string no vacío.",
                    entity_id=eid,
                    label=label,
                    value=value,
                )

            source = value.get("source")
            if isinstance(source, str):
                if source not in ALLOWED_INVESTIGADOR_SOURCES:
                    add_issue(
                        issues, "error", "value", "Investigador.value.source",
                        f"Investigador.value.source debe ser uno de {sorted(ALLOWED_INVESTIGADOR_SOURCES)} o lista de esos valores.",
                        entity_id=eid,
                        label=label,
                        value=value,
                    )
            elif isinstance(source, list):
                bad = [s for s in source if s not in ALLOWED_INVESTIGADOR_SOURCES]
                if bad:
                    add_issue(
                        issues, "error", "value", "Investigador.value.source",
                        f"Investigador.value.source contiene valores inválidos: {bad}.",
                        entity_id=eid,
                        label=label,
                        value=value,
                    )
            else:
                add_issue(
                    issues, "error", "value", "Investigador.value.source",
                    "Investigador.value.source debe ser string o lista de strings válidos.",
                    entity_id=eid,
                    label=label,
                    value=value,
                )


# =========================================================
# CONSTRUCCIÓN DE ÍNDICES
# =========================================================

def build_indices(graph: Dict[str, Any]) -> Dict[str, Any]:
    entities = graph.get("entities")
    relationships = graph.get("relationships")

    entity_by_id: Dict[str, Dict[str, Any]] = {}
    label_by_id: Dict[str, str] = {}
    entities_by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    outgoing: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    incoming: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    rels_by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for e in entities:
        eid = e.get("id")
        entity_by_id[eid] = e
        label_by_id[eid] = e.get("label")
        entities_by_label[e.get("label")].append(e)

    for r in relationships:
        outgoing[r.get("source_id")].append(r)
        incoming[r.get("target_id")].append(r)
        rels_by_type[r.get("type")].append(r)

    return {
        "entities": entities,
        "relationships": relationships,
        "entity_by_id": entity_by_id,
        "label_by_id": label_by_id,
        "entities_by_label": entities_by_label,
        "outgoing": outgoing,
        "incoming": incoming,
        "rels_by_type": rels_by_type,
    }


# =========================================================
# VALIDACIÓN ESTRUCTURAL
# =========================================================

def validate_structure(graph: Dict[str, Any], issues: List[Issue]) -> Dict[str, Any]:
    idx = build_indices(graph)
    entities = idx["entities"]
    relationships = idx["relationships"]
    entity_by_id = idx["entity_by_id"]
    label_by_id = idx["label_by_id"]

    entity_ids = [e.get("id") for e in entities]
    duplicate_entity_ids = [eid for eid, count in Counter(entity_ids).items() if count > 1]

    for eid in duplicate_entity_ids:
        add_issue(
            issues, "error", "structure", "entity.unique_id",
            "ID de entidad duplicado.",
            entity_id=eid,
        )

    for e in entities:
        eid = str(e.get("id", ""))
        label = str(e.get("label", ""))

        if not is_non_empty_str(eid):
            add_issue(
                issues, "error", "structure", "entity.id",
                "La entidad debe tener id no vacío.",
                entity=e,
            )

        if label not in ALLOWED_ENTITY_LABELS:
            add_issue(
                issues, "error", "structure", "entity.label",
                f"Label de entidad desconocido: {label}.",
                entity_id=eid,
                label=label,
            )
        else:
            validate_entity_value(e, issues)

    for r in relationships:
        rtype = str(r.get("type", ""))
        sid = str(r.get("source_id", ""))
        tid = str(r.get("target_id", ""))

        if rtype not in ALLOWED_REL_TYPES:
            add_issue(
                issues, "error", "structure", "relationship.type",
                f"Tipo de relación desconocido: {rtype}.",
                relationship=r,
            )
            continue

        if sid not in entity_by_id:
            add_issue(
                issues, "error", "structure", "relationship.source_exists",
                "La relación referencia un source_id inexistente.",
                relationship=r,
            )

        if tid not in entity_by_id:
            add_issue(
                issues, "error", "structure", "relationship.target_exists",
                "La relación referencia un target_id inexistente.",
                relationship=r,
            )

        if sid not in entity_by_id or tid not in entity_by_id:
            continue

        src_label = label_by_id[sid]
        tgt_label = label_by_id[tid]
        expected = VALID_REL_ENDPOINTS[rtype]

        if rtype == "EXTRAIDO_DE":
            ok = src_label == expected[0] and tgt_label in expected[1]
        else:
            ok = src_label == expected[0] and tgt_label == expected[1]

        if not ok:
            add_issue(
                issues, "error", "structure", "relationship.endpoints",
                f"Endpoints inválidos para {rtype}: {src_label} -> {tgt_label}.",
                relationship=r,
                expected=expected,
            )

    return idx


# =========================================================
# RESTRICCIONES DE NEGOCIO
# =========================================================

def validate_constraints(idx: Dict[str, Any], issues: List[Issue]) -> None:
    rels_by_type = idx["rels_by_type"]

    # -------- Helpers --------

    def targets_of(source_id: str, rel_type_name: str) -> List[str]:
        return [
            r["target_id"]
            for r in rels_by_type.get(rel_type_name, [])
            if r["source_id"] == source_id
        ]

    def sources_of(target_id: str, rel_type_name: str) -> List[str]:
        return [
            r["source_id"]
            for r in rels_by_type.get(rel_type_name, [])
            if r["target_id"] == target_id
        ]

    def chunk_document(chunk_id: str) -> Optional[str]:
        docs = targets_of(chunk_id, "DE_DOCUMENTO")
        return docs[0] if docs else None

    def project_documents(project_id: str) -> Set[str]:
        return set(targets_of(project_id, "ES_DESCRITO_POR"))

    def document_first_chunks(document_id: str) -> List[str]:
        return targets_of(document_id, "PRIMER_CHUNK")

    def chunk_next(chunk_id: str) -> List[str]:
        return targets_of(chunk_id, "SIGUIENTE_CHUNK")

    def chunks_of_document(document_id: str) -> Set[str]:
        return {
            r["source_id"]
            for r in rels_by_type.get("DE_DOCUMENTO", [])
            if r["target_id"] == document_id
        }

    def extracted_entities(chunk_id: str) -> Set[str]:
        return set(targets_of(chunk_id, "EXTRAIDO_DE"))

    # 1) Máximo 3 tópicos por proyecto
    for project in idx["entities_by_label"].get("Proyecto", []):
        pid = project["id"]
        topicos = targets_of(pid, "TIENE_TOPICO")
        if len(topicos) > 3:
            add_issue(
                issues, "error", "constraint", "project.max_3_topics",
                "Un proyecto tiene más de 3 tópicos asociados.",
                project_id=pid,
                topic_count=len(topicos),
                topic_ids=topicos,
            )
        if len(topicos) == 0:
            add_issue(
                issues, "error", "constraint", "project.0_topics",
                "Un proyecto debe tener al menos 1 tópico asociados.",
                project_id=pid,
                topic_count=len(topicos),
                topic_ids=topicos,
            )
    
    # 1.b) Todo proyecto debe tener al menos una relación TITULO_EXTRAIDO_DE
    for project in idx["entities_by_label"].get("Proyecto", []):
        pid = project["id"]
        title_chunks = targets_of(pid, "TITULO_EXTRAIDO_DE")

        if len(title_chunks) == 0:
            add_issue(
                issues, "error", "constraint", "project.requires_title_extracted_from",
                "El proyecto no tiene ninguna relación TITULO_EXTRAIDO_DE hacia un Chunk.",
                project_id=pid,
            )
    # 2) Si un chunk tiene siguiente, ambos deben pertenecer al mismo documento
    for rel in rels_by_type.get("SIGUIENTE_CHUNK", []):
        c1 = rel["source_id"]
        c2 = rel["target_id"]
        d1 = chunk_document(c1)
        d2 = chunk_document(c2)

        if d1 is None or d2 is None:
            add_issue(
                issues, "error", "constraint", "siguiente_chunk.same_document",
                "Chunks conectados por SIGUIENTE_CHUNK deben tener DE_DOCUMENTO.",
                chunk_from=c1,
                chunk_to=c2,
                doc_from=d1,
                doc_to=d2,
            )
        elif d1 != d2:
            add_issue(
                issues, "error", "constraint", "siguiente_chunk.same_document",
                "Chunks conectados por SIGUIENTE_CHUNK pertenecen a documentos distintos.",
                chunk_from=c1,
                chunk_to=c2,
                doc_from=d1,
                doc_to=d2,
            )
    # 2.b) Todo documento debe tener al menos un PRIMER_CHUNK y al menos un DE_DOCUMENTO
    for document in idx["entities_by_label"].get("Documento", []):
        did = document["id"]
        doc_type = document.get("value", {}).get("type")
        if doc_type == "tabla":
            continue
        first_chunks = document_first_chunks(did)
        doc_chunks = chunks_of_document(did)

        # Verifica PRIMER_CHUNK
        if not first_chunks:
            add_issue(
                issues, "error", "constraint", "document.requires_first_chunk",
                "El documento debe tener al menos una relación PRIMER_CHUNK.",
                document_id=did,
            )

        # Verifica DE_DOCUMENTO (chunks asociados)
        if not doc_chunks:
            add_issue(
                issues, "error", "constraint", "document.requires_chunks",
                "El documento debe tener al menos un chunk asociado mediante DE_DOCUMENTO.",
                document_id=did,
            )
    # 3) Todo chunk de un documento debe ser alcanzable desde el PRIMER_CHUNK
    for document in idx["entities_by_label"].get("Documento", []):
        did = document["id"]
        all_chunks = chunks_of_document(did)
        first_chunks = document_first_chunks(did)

        if not all_chunks and not first_chunks:
            continue

        if not first_chunks:
            add_issue(
                issues, "error", "constraint", "document.first_chunk_exists",
                "El documento tiene chunks pero no tiene PRIMER_CHUNK.",
                document_id=did,
                chunk_ids=sorted(all_chunks),
            )
            continue

        if len(first_chunks) > 1:
            add_issue(
                issues, "warning", "constraint", "document.multiple_first_chunks",
                "El documento tiene más de un PRIMER_CHUNK.",
                document_id=did,
                first_chunk_ids=first_chunks,
            )

        visited: Set[str] = set()
        stack = list(first_chunks)

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for nxt in chunk_next(current):
                if chunk_document(nxt) == did:
                    stack.append(nxt)

        unreachable = sorted(all_chunks - visited)
        if unreachable:
            add_issue(
                issues, "error", "constraint", "document.chunk_reachability",
                "Hay chunks del documento no alcanzables desde PRIMER_CHUNK.",
                document_id=did,
                unreachable_chunk_ids=unreachable,
                visited_chunk_ids=sorted(visited),
            )
    for topic in idx["entities_by_label"].get("Topico", []):
        tid = topic["id"]

        projects = sources_of(tid, "TIENE_TOPICO")
        if not projects:
            add_issue(
                issues,
                "error",
                "constraint",
                "topic.requires_project",
                "El tópico no está asociado a ningún proyecto.",
                topic_id=tid,
            )

        domains = targets_of(tid, "PERTENECE_A_DOMINIO")
        if len(domains) == 0:
            add_issue(
                issues,
                "error",
                "constraint",
                "topic.requires_domain",
                "El tópico no está asociado a ningún dominio.",
                topic_id=tid,
            )
        elif len(domains) > 1:
            add_issue(
                issues,
                "error",
                "constraint",
                "topic.single_domain",
                "El tópico está asociado a más de un dominio.",
                topic_id=tid,
                domain_ids=domains,
            )


    # 4) No pueden existir ciclos en SIGUIENTE_CHUNK
    next_graph: Dict[str, List[str]] = defaultdict(list)
    all_chunk_nodes: Set[str] = set()

    for rel in rels_by_type.get("SIGUIENTE_CHUNK", []):
        s = rel["source_id"]
        t = rel["target_id"]
        next_graph[s].append(t)
        all_chunk_nodes.add(s)
        all_chunk_nodes.add(t)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {n: WHITE for n in all_chunk_nodes}
    parent: Dict[str, Optional[str]] = {n: None for n in all_chunk_nodes}
    cycles_found: List[List[str]] = []

    def dfs_cycle(u: str) -> None:
        color[u] = GRAY
        for v in next_graph.get(u, []):
            if color[v] == WHITE:
                parent[v] = u
                dfs_cycle(v)
            elif color[v] == GRAY:
                cycle = [v]
                cur = u
                while cur is not None and cur != v:
                    cycle.append(cur)
                    cur = parent[cur]
                cycle.append(v)
                cycle.reverse()
                cycles_found.append(cycle)
        color[u] = BLACK

    for node in list(all_chunk_nodes):
        if color[node] == WHITE:
            dfs_cycle(node)

    if cycles_found:
        for cycle in cycles_found:
            add_issue(
                issues, "error", "constraint", "siguiente_chunk.acyclic",
                "Se detectó un ciclo en SIGUIENTE_CHUNK.",
                cycle=cycle,
            )

    # 5) Si un investigador participa en un proyecto, debe existir al menos
    #    un chunk asociado a ese investigador y ese chunk debe pertenecer
    #    a un documento que describa ese mismo proyecto.
    for rel in rels_by_type.get("PARTICIPO_EN", []):
        iid = rel["source_id"]
        pid = rel["target_id"]
        docs = project_documents(pid)

        evidence_ok = False
        supporting_chunks = []

        for extraido in rels_by_type.get("EXTRAIDO_DE", []):
            if extraido["target_id"] != iid:
                continue
            chunk_id = extraido["source_id"]
            did = chunk_document(chunk_id)
            if did in docs:
                evidence_ok = True
                supporting_chunks.append(chunk_id)

        if not evidence_ok:
            add_issue(
                issues, "error", "constraint", "participo_en.requires_evidence",
                "Existe PARTICIPO_EN sin chunk de evidencia del investigador en un documento del proyecto.",
                investigador_id=iid,
                project_id=pid,
                project_document_ids=sorted(docs),
            )

    # 6) Si un proyecto tiene un tópico, debe existir al menos un chunk asociado
    #    a ese tópico, y ese chunk debe pertenecer a un documento del proyecto.
    for rel in rels_by_type.get("TIENE_TOPICO", []):
        pid = rel["source_id"]
        tid = rel["target_id"]
        docs = project_documents(pid)

        evidence_ok = False
        supporting_chunks = []

        for extraido in rels_by_type.get("EXTRAIDO_DE", []):
            if extraido["target_id"] != tid:
                continue
            chunk_id = extraido["source_id"]
            did = chunk_document(chunk_id)
            if did in docs:
                evidence_ok = True
                supporting_chunks.append(chunk_id)

        if not evidence_ok:
            add_issue(
                issues, "error", "constraint", "tiene_topico.requires_evidence",
                "Existe TIENE_TOPICO sin chunk de evidencia del tópico en un documento del proyecto.",
                topico_id=tid,
                project_id=pid,
                project_document_ids=sorted(docs),
            )

    # 7) Si un proyecto tiene TITULO_EXTRAIDO_DE hacia un chunk, ese chunk debe
    #    pertenecer a un documento asociado al mismo proyecto.
    for rel in rels_by_type.get("TITULO_EXTRAIDO_DE", []):
        pid = rel["source_id"]
        cid = rel["target_id"]
        docs = project_documents(pid)
        did = chunk_document(cid)

        if did is None:
            add_issue(
                issues, "error", "constraint", "titulo_extraido_de.chunk_has_document",
                "El chunk apuntado por TITULO_EXTRAIDO_DE no tiene DE_DOCUMENTO.",
                project_id=pid,
                chunk_id=cid,
            )
        elif did not in docs:
            add_issue(
                issues, "error", "constraint", "titulo_extraido_de.same_project_document",
                "El chunk de TITULO_EXTRAIDO_DE no pertenece a un documento del mismo proyecto.",
                project_id=pid,
                chunk_id=cid,
                chunk_document_id=did,
                project_document_ids=sorted(docs),
            )


# =========================================================
# SCORE
# =========================================================

def compute_score(issues: List[Issue]) -> Dict[str, Any]:
    # Podés ajustar estos pesos si querés castigar más ciertas cosas.
    error_weights = {
        "structure": 5,
        "value": 3,
        "constraint": 4,
    }
    warning_weights = {
        "structure": 2,
        "value": 1,
        "constraint": 2,
    }

    penalty = 0
    for i in issues:
        if i.severity == "error":
            penalty += error_weights.get(i.category, 3)
        else:
            penalty += warning_weights.get(i.category, 1)

    score = max(0, 100 - penalty)

    error_count = sum(1 for i in issues if i.severity == "error")
    warning_count = sum(1 for i in issues if i.severity == "warning")

    if score >= 95:
        verdict = "Excelente ajuste al esquema"
    elif score >= 80:
        verdict = "Buen ajuste al esquema"
    elif score >= 60:
        verdict = "Ajuste parcial al esquema"
    else:
        verdict = "Mal ajuste al esquema"

    return {
        "score": score,
        "verdict": verdict,
        "error_count": error_count,
        "warning_count": warning_count,
        "penalty": penalty,
    }


# =========================================================
# REPORTE PRINCIPAL
# =========================================================

def validate_graph_against_schema(graph: Dict[str, Any]) -> ValidationReport:
    issues: List[Issue] = []

    idx = validate_structure(graph, issues)

    # Solo validamos restricciones si los nodos y relaciones mínimamente existen.
    validate_constraints(idx, issues)

    stats = {
        "entity_count": len(idx["entities"]),
        "relationship_count": len(idx["relationships"]),
        "entities_by_label": {
            label: len(items) for label, items in idx["entities_by_label"].items()
        },
        "relationships_by_type": {
            rtype: len(items) for rtype, items in idx["rels_by_type"].items()
        },
    }

    summary = compute_score(issues)

    return ValidationReport(
        summary=summary,
        issues=issues,
        stats=stats,
    )