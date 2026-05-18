# src/extraction/ie.py
from __future__ import annotations

import json
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import ijson

from institutional_graphrag.document_naming import PATTERN_DOCUMENT, PATTERN_TABLE
from institutional_graphrag.extraction.llm_extractor import (
    LLMEntityExtractor,
    load_all_topics_and_domains,
)
from institutional_graphrag.extraction.rule_based_extractor import RuleBasedExtractor
from institutional_graphrag.extraction.tabular_extractor import TabularExtractor
from institutional_graphrag.graph.schema import (
    Documento,
    Entity,
    GraphSchema,
    Relationship,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[4] / "data"

ALLOWED_SUFFIXES = {".parquet", ".pdf"}


@dataclass
class ExtractionResult:
    entities: List[Entity]
    relationships: List[Relationship]
    errors: List[Dict[str, Any]]


class EntityExtractor:
    def __init__(
        self,
        llm_model: Optional[str] = None,
        data_dir: Path | None = None,
    ):
        base = data_dir if data_dir is not None else DATA_DIR
        self.data_dir = base
        self.documents_dir = base / "corpus"
        self.chunks_dir = base / "chunks"
        self.table_dir = base / "tables"
        self.input_dir = base / "entities_relations"
        self.res: ExtractionResult = ExtractionResult([], [], [])
        self.rule_based = RuleBasedExtractor()
        self.tabular = TabularExtractor()

        self.doc_by_basename: dict[str, Documento] = {}
        self.doc_by_id: dict[str, Documento] = {}
        self.docs_by_group_year: dict[tuple[str, str], list[Documento]] = defaultdict(list)

        # Configuración LLM
        self.llm_model = llm_model

        self._seen_entities: set[tuple[str, str]] = set()
        self._seen_rels: set[tuple[str, str, str, str]] = set()
        self._rel_index: dict[tuple[str, str, str], int] = {}
        self.reg: Dict[str, List[str]] = {}
        self._registry_dirty: bool = False

    def cleanup(self):
        self._seen_entities = set()
        self._seen_rels = set()
        self._rel_index = {}
        self.reg = {}
        self._registry_dirty = False
        self.res = ExtractionResult(entities=[], relationships=[], errors=[])
        self.rule_based.cleanup()
        self.doc_by_basename = {}
        self.doc_by_id = {}
        self.docs_by_group_year = defaultdict(list)

    def run(
        self,
        max_docs: int | None = None,
        llm_topics: bool = True,
        include_headings: bool = True,
        checkpoint_every: int = 5,
    ) -> ExtractionResult:
        topic_entities, topic_rels = load_all_topics_and_domains()
        self.add_entities(topic_entities)
        self.add_relationship(topic_rels)

        entities_json = self.input_dir / "entity_documents.json"
        if entities_json.exists():
            self.load_subset_from_graph_json(
                entities_json,
                label="Investigador",
                value_filter={"source": "llm"},
            )
            self.load_subset_from_graph_json(entities_json, label="Topico")
            self.load_subset_from_graph_json(entities_json, label="Dominio")

        self._extract_documents()
        self._build_doc_indexes()
        res = self.rule_based.associate_tables_with_documents(
            self.docs_by_group_year, self.table_dir
        )
        self.res.errors.extend(res.errors)

        self._extract_chunks()
        self._extract_projects_and_responsible()
        self._extract_researchers_from_tabular()
        self._extract_with_llm(
            max_docs=max_docs,
            llm_topics=llm_topics,
            include_headings=include_headings,
            checkpoint_every=checkpoint_every,
        )
        return self.res

    def load_subset_from_graph_json(
        self,
        json_path: str | Path,
        *,
        label: str,
        value_filter: Optional[dict[str, Any]] = None,
    ) -> None:
        json_path = Path(json_path).resolve()
        if not json_path.is_file():
            self.res.errors.append(
                {"type": "MissingFile", "message": f"No existe el archivo: {json_path}"}
            )
            return

        matched_ids: set[str] = set()
        matched_entities: list[Entity] = []

        # -------- 1) ENTITIES (streaming) --------
        try:
            with json_path.open("rb") as f:
                for raw in ijson.items(f, "entities.item"):
                    if not isinstance(raw, dict):
                        continue
                    if raw.get("label") != label:
                        continue

                    match = True

                    v = raw.get("value")
                    if value_filter is not None:
                        if not isinstance(v, dict):
                            continue
                        for k, expected in value_filter.items():
                            val = v.get(k)

                            if k == "source" and isinstance(val, list):
                                if expected not in val:
                                    match = False
                                    break
                            elif val != expected:
                                match = False
                                break

                        if not match:
                            continue

                    entity_id = raw.get("id")
                    if not isinstance(entity_id, str) or not entity_id:
                        continue

                    cls = GraphSchema.ENTITIES.get(label)
                    if cls is None:
                        self.res.errors.append(
                            {"type": "UnknownEntityType", "message": f"Label desconocido: {label}"}
                        )
                        return

                    try:
                        ent = cls(id=entity_id, value=v)
                    except Exception as exc:
                        self.res.errors.append(
                            {"type": "InvalidEntity", "message": f"{label}({entity_id}): {exc}"}
                        )
                        continue

                    matched_entities.append(ent)
                    matched_ids.add(entity_id)

        except Exception as exc:
            self.res.errors.append(
                {"type": "JsonReadError", "message": f"Error leyendo entities: {exc}"}
            )
            return

        self.add_entities(matched_entities)

        # -------- 2) RELATIONSHIPS (segunda pasada) --------
        try:
            with json_path.open("rb") as f:
                for raw in ijson.items(f, "relationships.item"):
                    if not isinstance(raw, dict):
                        continue

                    rel_type = raw.get("type")
                    source_id = raw.get("source_id")
                    target_id = raw.get("target_id")
                    props = raw.get("properties") or {}

                    if (
                        not isinstance(rel_type, str)
                        or not isinstance(source_id, str)
                        or not isinstance(target_id, str)
                    ):
                        continue

                    if source_id not in matched_ids and target_id not in matched_ids:
                        continue

                    if not isinstance(props, dict):
                        props = {}

                    try:
                        self.add_relationship(
                            [
                                Relationship(
                                    type=rel_type,
                                    source_id=source_id,
                                    target_id=target_id,
                                    properties=props,
                                )
                            ]
                        )
                    except Exception as exc:
                        self.res.errors.append(
                            {
                                "type": "InvalidRelationship",
                                "message": f"{rel_type}({source_id}->{target_id}): {exc}",
                            }
                        )

        except Exception as exc:
            self.res.errors.append(
                {"type": "JsonReadError", "message": f"Error leyendo relationships: {exc}"}
            )

    def _build_doc_indexes(self) -> None:
        docs = [cast(Documento, e) for e in self.res.entities if e.label == "Documento"]
        self.doc_by_id = {str(d.id): d for d in docs}

        self.doc_by_basename = {}
        self.docs_by_group_year = defaultdict(list)
        for d in docs:
            base = d.value["base_name"]
            self.doc_by_basename[base] = d
            key = (d.value["is_group"], d.value["year_publisher"])
            self.docs_by_group_year[key].append(d)

    def add_entities(self, entities: List[Entity]) -> bool:
        for e in entities:
            key = (e.label, str(e.id))

            if key in self._seen_entities:
                # reemplazar la entidad existente
                for i, existing in enumerate(self.res.entities):
                    if existing.label == e.label and str(existing.id) == str(e.id):
                        if (
                            e.label == "Investigador"
                            and isinstance(existing.value, dict)
                            and isinstance(e.value, dict)
                        ):
                            for prop in (
                                "documento",
                                "tipo_documento",
                                "pais_documento",
                                "sexo",
                            ):
                                if not e.value.get(prop) and existing.value.get(prop):
                                    e.value[prop] = existing.value[prop]
                        self.res.entities[i] = e
                        continue
                continue

            self._seen_entities.add(key)
            self.res.entities.append(e)
        return True

    def add_relationship(self, relations: List[Relationship]) -> bool:
        for r in relations:
            key = (r.type, str(r.source_id), str(r.target_id))

            if key in self._rel_index:
                idx = self._rel_index[key]
                self.res.relationships[idx] = r
                continue

            self._rel_index[key] = len(self.res.relationships)
            self.res.relationships.append(r)
        return True

    def _extract_documents(self) -> None:

        def ensure_dir(d: Path) -> bool:
            if d.exists():
                return True
            self.res.errors.append(
                {"type": "MissingFolder", "message": f"No existe la carpeta: {d}"}
            )
            return False

        if not ensure_dir(self.documents_dir) or not ensure_dir(self.table_dir):
            return

        def add_docs_from_dir(
            d: Path, pattern, value_builder, create_year_entity: bool = False
        ) -> None:
            for path in sorted(p for p in d.iterdir() if p.is_file()):
                res = self.rule_based.extract_document(
                    path, pattern, value_builder, create_year_entity
                )
                self.add_entities(res.entities)
                self.add_relationship(res.relationships)
                self.res.errors.extend(res.errors)

        add_docs_from_dir(
            self.documents_dir,
            PATTERN_DOCUMENT,
            lambda base, m: {
                "base_name": base,
                "is_group": m.group("group"),
                "year_publisher": m.group("year"),
                "sub_id": m.group("doc_id"),
                "type": m.group("kind"),
            },
        )

        add_docs_from_dir(
            self.table_dir,
            PATTERN_TABLE,
            lambda base, m: {
                "base_name": base,
                "is_group": m.group("group"),
                "year_publisher": m.group("year"),
                "type": "tabla",
            },
            create_year_entity=True,
        )

    def _extract_chunks(self) -> None:
        if not self.chunks_dir.exists():
            self.res.errors.append(
                {
                    "type": "MissingFolder",
                    "message": f"No existe la carpeta: {self.chunks_dir.as_posix()}",
                }
            )
            return

        all_paths = sorted(
            [p for p in self.chunks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"]
        )

        for path in all_paths:
            res = self.rule_based.extract_chunk(path, self.doc_by_basename)
            self.add_entities(res.entities)
            self.add_relationship(res.relationships)
            self.res.errors.extend(res.errors)

    def _extract_projects_and_responsible(self) -> None:
        res = self.rule_based.extract_projects_and_responsible_from_tables(
            self.doc_by_id, self.chunks_dir
        )
        self.add_entities(res.entities)
        self.add_relationship(res.relationships)
        self.res.errors.extend(res.errors)

    def _extract_researchers_from_tabular(self) -> None:
        res = self.tabular.extract_from_directory(self.table_dir)
        self.add_entities(res.entities)
        self.add_relationship(res.relationships)
        self.res.errors.extend(res.errors)

    def _result_to_json(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.res.entities],
            "relationships": [r.to_dict() for r in self.res.relationships],
            "errors": self.res.errors,
        }

    def _read_json(self, path: Path) -> Optional[dict[str, Any]]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            self.res.errors.append({"type": "InvalidJson", "message": f"{path.name}: {e}"})
            return None

        if not isinstance(data, dict):
            self.res.errors.append(
                {"type": "InvalidJson", "message": f"{path.name}: JSON root is not an object"}
            )
            return None

        return cast(dict[str, Any], data)

    def save_in_file(self, filename: str) -> None:

        self.input_dir.mkdir(parents=True, exist_ok=True)

        out = self._result_to_json()
        out_path = self.input_dir / filename
        out_path.write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load_from_json(self, filename: str) -> Optional[ExtractionResult]:
        path = self.input_dir / filename
        if not path.exists():
            return None

        data = self._read_json(path)
        if data is None:
            return None

        entities_data = data.get("entities", [])
        relationships_data = data.get("relationships", [])
        errors_data = data.get("errors", [])

        entities: list[Entity] = []
        relationships: list[Relationship] = []
        errors: list[dict[str, Any]] = errors_data if isinstance(errors_data, list) else []

        if not isinstance(entities_data, list):
            errors.append({"type": "InvalidFormat", "message": "'entities' no es lista"})
            return ExtractionResult(entities=entities, relationships=relationships, errors=errors)

        # --- reconstruir entidades ---
        for e in entities_data:
            if not isinstance(e, dict):
                errors.append(
                    {"type": "InvalidEntity", "message": f"Entidad no es dict: {repr(e)[:200]}"}
                )
                continue

            entity_id = e.get("id")
            label = e.get("label")
            value = e.get("value")
            if not isinstance(entity_id, str) or not entity_id:
                errors.append(
                    {"type": "InvalidEntity", "message": f"Entidad sin id válido: {repr(e)[:200]}"}
                )
                continue
            if not isinstance(label, str) or not label:
                errors.append(
                    {
                        "type": "InvalidEntity",
                        "message": f"Entidad sin label válido: {repr(e)[:200]}",
                    }
                )
                continue

            cls = GraphSchema.ENTITIES.get(label)

            if cls is None:
                errors.append(
                    {"type": "UnknownEntityType", "message": f"Label desconocido: {label}"}
                )
                continue

            # Algunos LLMs devuelven {"value": "string"} en vez de "string" para
            # entidades cuyo schema espera un str (Proyecto, Grupo, Topico, Dominio).
            if label in {"Proyecto", "Grupo", "Topico", "Dominio"} and isinstance(value, dict):
                value = value.get("value", value)

            try:
                entities.append(cls(id=entity_id, value=value))
            except Exception as exc:
                errors.append({"type": "InvalidEntity", "message": f"{label}({entity_id}): {exc}"})

        # --- reconstruir relaciones ---
        if not isinstance(relationships_data, list):
            self.res.errors.append(
                {"type": "InvalidFormat", "message": "'relationships' no es lista"}
            )
            return None

        for r in relationships_data:
            if not isinstance(r, dict):
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"Relación no es dict: {repr(r)[:200]}",
                    }
                )
                continue
            rel_type = r.get("type")
            source_id = r.get("source_id")
            target_id = r.get("target_id")
            props = r.get("properties", {})

            if not isinstance(rel_type, str) or not rel_type:
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"Relación sin type válido: {repr(r)[:200]}",
                    }
                )
                continue
            if (
                not isinstance(source_id, str)
                or not source_id
                or not isinstance(target_id, str)
                or not target_id
            ):
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"Relación sin endpoints: {repr(r)[:200]}",
                    }
                )
                continue
            if not isinstance(props, dict):
                props = {}  # normalizar

            try:
                relationships.append(
                    Relationship(
                        type=rel_type, source_id=source_id, target_id=target_id, properties=props
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "type": "InvalidRelationship",
                        "message": f"{rel_type}({source_id}->{target_id}): {exc}",
                    }
                )

        return ExtractionResult(entities=entities, relationships=relationships, errors=errors)

    def load_registry(self, path: Path) -> Dict[str, List[str]]:
        if not path.exists():
            return {}
        data_any: Any = json.loads(path.read_text(encoding="utf-8"))
        data = cast(dict[str, list[str]], data_any)
        return data

    def atomic_write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.reg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)

    def already_run(self, doc_id: str, entity_label: str) -> bool:
        return entity_label in self.reg.get(doc_id, [])

    def mark_success(self, doc_id: str, entity_label: str) -> None:
        self.reg.setdefault(doc_id, [])
        if entity_label not in self.reg[doc_id]:
            self.reg[doc_id].append(entity_label)
            self.reg[doc_id].sort()
            self._registry_dirty = True

    def _flush_registry(self, registry_path: Path) -> None:
        if self._registry_dirty:
            self.atomic_write(registry_path)
            self._registry_dirty = False

    def _save_checkpoint(
        self,
        filename: str = "entity_documents.json",
        registry_path: Path | None = None,
    ) -> None:
        """Guarda el estado actual a disco (checkpoint intermedio)."""
        self.save_in_file(filename)
        if registry_path is not None:
            self._flush_registry(registry_path)
        logger.info("[Checkpoint] Guardado intermedio en %s", filename)

    def _extract_with_llm(
        self,
        max_docs: int | None = None,
        llm_topics: bool = True,
        include_headings: bool = True,
        checkpoint_every: int = 5,
    ) -> None:
        """Extraer tópicos usando LLM.
        Args:
            max_docs: Límite opcional de documentos a procesar.
            checkpoint_every: Guardar a disco cada N documentos procesados.
        """
        existing_topic_ids = {e.id for e in self.res.entities if e.label == "Topico"}

        llm_extractor = LLMEntityExtractor(
            llm_model=self.llm_model,
            temperature=0.1,
            max_tokens=1024,
        )
        registry_path = self.input_dir / "llm_registry.json"
        self.reg = self.load_registry(registry_path)
        self._registry_dirty = False

        # Procesar cada proyecto y grupo
        projects = [e for e in self.res.entities if e.label in ("Proyecto", "Grupo")]

        docs_processed = 0
        self.docs_by_project: dict[str, list[str]] = defaultdict(list)
        for r in self.res.relationships:
            if r.type == "ES_DESCRITO_POR":
                self.docs_by_project[r.source_id].append(r.target_id)

        for project in projects:
            project_id = project.id

            # Obtener documentos del proyecto
            project_docs = self.docs_by_project.get(project_id, [])

            if not project_docs:
                continue

            # Procesar chunks de cada documento del proyecto
            for doc_id in project_docs:
                # Verificar límite de documentos
                if max_docs is not None and docs_processed >= max_docs:
                    logger.info(f"[LLM] Límite de {max_docs} documentos alcanzado")
                    return

                # Skipear documentos de tabla
                if doc_id.endswith("_table"):
                    continue

                doc = self.doc_by_id.get(doc_id)
                if doc is None:
                    continue
                topic_cache = self.already_run(doc_id, "Topico")

                # Cargar chunks del documento
                base_name = doc.value.get("base_name", "")
                if not base_name:
                    continue

                chunks_file = self.chunks_dir / f"{base_name}_chunks.json"
                if not chunks_file.exists():
                    continue

                try:
                    # Cargar chunks del archivo
                    payload = self._read_json(chunks_file)
                    if payload is None:
                        continue

                    chunks = payload.get("chunks", [])
                    if not isinstance(chunks, list):
                        continue

                    if topic_cache or not llm_topics:
                        logger.info(f"[LLM Topics] Archivo en cache: {doc_id}")
                    else:
                        # Extraer tópicos usando LLM de todos los chunks
                        logger.info(
                            f"[LLM Topics] Procesando {len(chunks)} chunks de {base_name}..."
                        )
                        llm_result_topic = llm_extractor.extract_topics_from_chunks(
                            chunks, max_chunks=None, include_headings=include_headings
                        )
                        # Agregar errores
                        self.res.errors.extend(llm_result_topic.errors)
                        # Crear entidades y relaciones chunk->topico
                        new_entities, new_relationships = (
                            llm_extractor.create_topics_from_llm_extraction(
                                llm_result_topic, existing_topic_ids
                            )
                        )
                        # Agregar al resultado
                        self.add_entities(new_entities)
                        self.add_relationship(new_relationships)

                        # Actualizar el set de IDs globales
                        existing_topic_ids.update(e.id for e in new_entities)
                        self.mark_success(doc_id, "Topico")

                        logger.info(
                            f"[LLM Topics] ✓ {base_name}: encontrados {len(llm_result_topic.topics)} tópicos, {len(llm_result_topic.errors)} errores"
                        )

                    # Incrementar contador de documentos procesados
                    docs_processed += 1

                    # Checkpoint periódico para no perder progreso
                    if checkpoint_every > 0 and docs_processed % checkpoint_every == 0:
                        self._save_checkpoint(registry_path=registry_path)

                except Exception as e:
                    self.res.errors.append(
                        {
                            "type": "LLMExtractionError",
                            "document": base_name,
                            "message": f"Error procesando documento con LLM: {str(e)}",
                        }
                    )
                    docs_processed += 1
            self._aggregate_topics_for_project(project_id)

        # Checkpoint final tras toda la extracción LLM
        if docs_processed > 0:
            self._save_checkpoint(registry_path=registry_path)
            logger.info("[LLM] Extracción completada — %d documentos procesados", docs_processed)

    def _aggregate_topics_for_project(self, project_id: str) -> None:
        """Agregar tópicos a nivel de proyecto basándose en los chunks.

        Cuenta los tópicos de todos los chunks del proyecto y crea
        relaciones Proyecto TIENE_TOPICO Topico para todos los tópicos mencionados.
        """
        # Obtener todos los chunks del proyecto (a través de documentos)
        project_docs = self.docs_by_project.get(project_id, [])
        topic_entity_ids = {e.id for e in self.res.entities if e.label == "Topico"}
        chunk_by_doc: dict[str, list[str]] = defaultdict(list)
        topic_by_chunk: dict[str, list[str]] = defaultdict(list)
        for r in self.res.relationships:
            if r.type == "DE_DOCUMENTO":
                chunk_by_doc[r.target_id].append(r.source_id)
            elif r.type == "EXTRAIDO_DE":
                topic_by_chunk[r.source_id].append(r.target_id)

        project_chunks = set()
        for doc_id in project_docs:
            doc_chunks = chunk_by_doc.get(doc_id, [])
            project_chunks.update(doc_chunks)

        # Contar tópicos de los chunks del proyecto
        topic_counts: Counter[str] = Counter()
        for chunk_id in project_chunks:
            chunk_topics = topic_by_chunk.get(chunk_id, [])
            topic_ids = [tid for tid in chunk_topics if tid in topic_entity_ids]

            topic_counts.update(topic_ids)

        # Crear relaciones proyecto->topico para los top 3 tópicos más mencionados
        top_topics = topic_counts.most_common(3)
        relationships_to_add = []
        for topic_id, count in top_topics:
            rel = Relationship(
                type="TIENE_TOPICO",
                source_id=project_id,
                target_id=topic_id,
                properties={"mention_count": count},
            )
            relationships_to_add.append(rel)

        # 3️⃣ Llamar una sola vez al método
        self.add_relationship(relationships_to_add)
