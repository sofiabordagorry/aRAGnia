from __future__ import annotations

import csv
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from docling_core.types.doc import DoclingDocument
from dotenv import load_dotenv

from institutional_graphrag.config import EMBED_MODEL_ID
from institutional_graphrag.document_naming import (
    PATTERN_DOCUMENT_WITH_OPTIONAL_PDF as PATTERN_DOCUMENT,
)
from institutional_graphrag.document_naming import PATTERN_TABLE
from institutional_graphrag.extraction.ie import EntityExtractor
from institutional_graphrag.extraction.rule_based_extractor import RuleBasedExtractor
from institutional_graphrag.graph.builder import GraphBuilder
from institutional_graphrag.graph.graph_loader import load_graph_json
from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker
from institutional_graphrag.ingest.docling_parser import parse_single_document
from institutional_graphrag.ingest.file_namer import (
    PdfKind,
    classify_pdf,
    generate_new_filename,
    save_temp_file,
)
from institutional_graphrag.ingest.table_extractors import clean_table, convert_tables_to_chunks
from institutional_graphrag.ingest.type_converter import odt_bytes_to_pdf

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
_PROYECTOS_YEAR_RE = re.compile(r"^proyectos[_\s]?(\d{4})", re.IGNORECASE)


class MissingNeo4jCredentialsError(Exception):
    pass


class IngestService:
    def __init__(
        self,
        *,
        data_dir: Path,
        env_path: Optional[Path] = None,
        keep_debug_artifacts: bool,
        include_headings: bool = True,
    ):
        self.keep_debug_artifacts = keep_debug_artifacts
        self.include_headings = include_headings

        self.data_dir = data_dir
        self.tables_dir = data_dir / "tables"
        self.output_dir = data_dir / "corpus"
        self.docling_dir = data_dir / "docling"
        self.chunks_dir = data_dir / "chunks"
        self.entities_dir = data_dir / "entities_relations"

        for d in (
            self.tables_dir,
            self.output_dir,
            self.docling_dir,
            self.chunks_dir,
            self.entities_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        if env_path is not None:
            load_dotenv(env_path)

        self.cache_file = data_dir / "cache_paths.csv"
        self.cache_file.touch(exist_ok=True)
        self.tokenizer = EMBED_MODEL_ID
        self.chunker = get_native_chunker(tokenizer=self.tokenizer)
        self.rule_based_extractor = RuleBasedExtractor()
        self.entity_extractor = EntityExtractor()
        self.processed_files: List[str] = []

        self.cache_dict: Dict[str, int] = {}

        neo4j_host = os.getenv("HOST", "localhost")
        neo4j_port = os.getenv("NEO4J_BOLT_PORT", "7687")
        self.neo4j_uri = f"bolt://{neo4j_host}:{neo4j_port}"

        self.user = os.getenv("NEO4J_USER")
        self.password = os.getenv("NEO4J_PASSWORD")
        if not self.user or not self.password:
            error = {
                "type": "MissingNeo4jCredentials",
                "message": "Faltan NEO4J_USER o NEO4J_PASSWORD en el entorno.",
            }
            self.entity_extractor.res.errors.append(error)
            raise MissingNeo4jCredentialsError(error["message"])
        self.builder = GraphBuilder(self.neo4j_uri, self.user, self.password)

    def cleanup(self):
        self.builder = GraphBuilder(self.neo4j_uri, self.user, self.password)
        self.rule_based_extractor.cleanup()
        self.entity_extractor.cleanup()
        self.processed_files = []
        self.cache_dict = {}

    async def ingest_from_uploads(
        self,
        folder_files: List[Tuple[str, bytes]],
        csv_bytes: Optional[bytes] = None,
        csv_filename: Optional[str] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        Procesa archivos individuales subidos desde el frontend.
        folder_files: lista de (relative_path, bytes) donde el path preserva la
        estructura de carpetas completa (ej: '1_GRUPOS I+D.../2010_.../152/...').
        """
        self.cache_dict = self.load_cache()

        if csv_bytes is not None and csv_filename is not None:
            self._merge_proyectos_csv(csv_bytes, csv_filename)

        skipped: List[Dict[str, str]] = []
        processed: List[str] = []

        if folder_files:
            has_proyectos_csv = any(
                "proyectos" in p.name.lower() for p in self.tables_dir.glob("*.csv")
            )
            if not has_proyectos_csv:
                self.entity_extractor.res.errors.append(
                    {
                        "type": "MissingProyectosCSV",
                        "message": (
                            "No hay CSV de proyectos en el sistema. "
                            "Suba un CSV con 'proyectos' en el nombre."
                        ),
                    }
                )
            else:
                valid_pairs = self._load_valid_project_pairs()
                total_files = sum(
                    1
                    for rel_path, _ in folder_files
                    if (
                        self._extract_project_key_from_path(rel_path) is None
                        or self._extract_project_key_from_path(rel_path) in valid_pairs
                    )
                )
                processed_count = 0
                for rel_path, content in folder_files:
                    project_key = self._extract_project_key_from_path(rel_path)
                    if project_key is not None and project_key not in valid_pairs:
                        skipped.append(
                            {
                                "path": rel_path,
                                "year": project_key[0],
                                "id_formulario": project_key[1],
                            }
                        )
                        continue
                    processed_count += 1
                    try:
                        full_cloud_path = "\\" + rel_path.replace("/", "\\")
                        base_name = await self._process_file(full_cloud_path, content, len(content))
                        if base_name:
                            processed.append(base_name)
                    except Exception as e:
                        self.entity_extractor.res.errors.append(
                            {
                                "type": "ProcessFileError",
                                "file": rel_path,
                                "message": str(e),
                            }
                        )
                    if progress_callback:
                        progress_callback(processed_count, total_files)

        convert_tables_to_chunks()
        self._extract_projects_and_responsibles()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        self.entity_extractor._extract_entities(include_headings=self.include_headings)
        entity_dicts, rel_dicts = self._collect_entity_dicts()
        entity_json_path = self._save_entities_json(entity_dicts, rel_dicts)

        if entity_json_path is not None:
            self._ingest_neo4j(entity_json_path)

        self.save_cache()

        if not self.keep_debug_artifacts:
            for base in processed:
                self._cleanup_processed_file(base)
            if entity_json_path is not None:
                entity_json_path.unlink(missing_ok=True)

        self.builder.close()
        return {
            "processed": processed,
            "skipped": skipped,
            "errors": self.entity_extractor.res.errors,
            "csv_loaded": csv_bytes is not None,
        }

    async def _process_file(
        self,
        full_cloud_path: str,
        content: bytes,
        file_size: int,
    ) -> Optional[str]:
        prev_size = self.cache_dict.get(full_cloud_path)
        if prev_size is not None and prev_size == file_size:
            print(f"Sin cambios (mismo tamaño): {full_cloud_path}")
            return None
        if prev_size is not None and prev_size != file_size:
            print(f"Actualizado (cambió tamaño): {full_cloud_path}")

        new_filename = generate_new_filename(full_cloud_path)
        base_name = Path(new_filename).stem

        final_pdf_path = self.output_dir / new_filename
        docling_json_path = self.docling_dir / f"{base_name}.json"
        chunk_json_path = self.chunks_dir / f"{base_name}_chunks.json"

        bytes_source = content
        if Path(new_filename).suffix.lower() == ".odt":
            print(f"Convirtiendo archivo odt a pdf: {new_filename}")
            bytes_source = odt_bytes_to_pdf(bytes_source)
            new_filename = str(Path(new_filename).with_suffix(".pdf"))
            base_name = Path(new_filename).stem
            final_pdf_path = self.output_dir / new_filename

        tmp_path = save_temp_file(bytes_source, new_filename)

        try:
            kind = classify_pdf(new_filename)

            def value_builder(base, m):
                return {
                    "base_name": base,
                    "is_group": m.group("group"),
                    "year_publisher": m.group("year"),
                    "sub_id": m.group("doc_id"),
                    "type": m.group("kind"),
                }

            doc_dict = parse_single_document(tmp_path)
            if doc_dict is None:
                raise ValueError(f"No se pudo extraer texto del documento: {new_filename}")

            with open(final_pdf_path, "wb") as out:
                out.write(bytes_source)

            docling_json_path.write_text(
                json.dumps(doc_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            doc = DoclingDocument.model_validate(doc_dict)
            chunks = chunk_document(doc=doc, chunker=self.chunker)
            chunk_json_path.write_text(
                json.dumps(
                    {
                        "source": doc.name,
                        "total_chunks": len(chunks),
                        "tokenizer": self.tokenizer,
                        "chunks": chunks,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            self._extract_doc_and_chunks(
                file_path=final_pdf_path,
                kind=kind,
                value_builder=value_builder,
                chunk_file=chunk_json_path,
            )

            self.cache_dict[full_cloud_path] = file_size
            return base_name

        finally:
            tmp_path.unlink(missing_ok=True)

    def _merge_proyectos_csv(self, csv_bytes: bytes, csv_filename: str) -> None:
        try:
            raw_content = csv_bytes.decode("utf-8")
            content = self._clean_csic_csv_if_needed(raw_content)

            reader = csv.DictReader(io.StringIO(content))
            new_rows = list(reader)
            fieldnames = list(reader.fieldnames or [])

            if not new_rows:
                return

            existing_csv_path: Optional[Path] = None
            for p in self.tables_dir.glob("*.csv"):
                if "proyectos" in p.name.lower():
                    existing_csv_path = p
                    break

            if existing_csv_path is None:
                dest = self.tables_dir / Path(csv_filename).name
                dest.write_text(content, encoding="utf-8")
                return

            existing_rows: List[Dict[str, Any]] = []
            existing_fieldnames: List[str] = []
            with open(existing_csv_path, encoding="utf-8", newline="") as f:
                reader2 = csv.DictReader(f)
                existing_rows = list(reader2)
                existing_fieldnames = list(reader2.fieldnames or [])

            _GENERATED_COLS = {"row_id", "file_type", "file_id"}

            def _row_key(row: Dict[str, Any]) -> frozenset:
                return frozenset((k, v) for k, v in row.items() if k not in _GENERATED_COLS)

            existing_row_set = {_row_key(row) for row in existing_rows}
            rows_to_add = [row for row in new_rows if _row_key(row) not in existing_row_set]

            if not rows_to_add:
                return

            merged_fieldnames = existing_fieldnames[:]
            for fn in fieldnames:
                if fn not in merged_fieldnames:
                    merged_fieldnames.append(fn)

            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=merged_fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(existing_rows + rows_to_add)

            existing_csv_path.write_text(out.getvalue(), encoding="utf-8")

        except Exception as e:
            self.entity_extractor.res.errors.append({"type": "CSVMergeError", "message": str(e)})

    def _load_valid_project_pairs(self) -> Set[Tuple[str, str]]:
        """Return the set of (anio, id_formulario) pairs from all proyectos CSVs."""
        pairs: Set[Tuple[str, str]] = set()
        for csv_path in self.tables_dir.glob("*.csv"):
            if "proyectos" not in csv_path.name.lower():
                continue
            try:
                with open(csv_path, encoding="utf-8", newline="") as f:
                    for row in csv.DictReader(f):
                        year = (row.get("anio") or "").strip()
                        id_form = (row.get("id_formulario") or "").strip()
                        if year and id_form:
                            pairs.add((year, id_form))
            except Exception:
                pass
        return pairs

    def _extract_project_key_from_path(self, rel_path: str) -> Optional[Tuple[str, str]]:
        """Extract (year, id_formulario) from a path like 'proyectos_2018/.../63/...'."""
        parts = rel_path.replace("\\", "/").split("/")
        year_match = _PROYECTOS_YEAR_RE.match(parts[0])
        if not year_match:
            return None
        year = year_match.group(1)
        id_formulario: Optional[str] = None
        for part in parts[1:]:
            if part.isdigit():
                id_formulario = part
                break
        if id_formulario is None:
            return None
        return (year, id_formulario)

    def _clean_csic_csv_if_needed(self, content: str) -> str:
        """Aplica clean_table si el CSV tiene el formato CSIC (primera columna numérica).
        clean_table une filas partidas por saltos de línea dentro de celdas."""
        lines = content.strip().splitlines()
        is_csic_format = False
        for line in lines[1:]:
            stripped = line.strip()
            if stripped:
                first_col = stripped.split(",", 1)[0].strip().strip('"').strip("'")
                is_csic_format = first_col.isdigit()
                break

        if not is_csic_format:
            return content

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", encoding="utf-8", delete=False
        ) as tmp_in:
            tmp_in.write(content)
            tmp_in_path = Path(tmp_in.name)

        tmp_out_path = tmp_in_path.with_suffix(".out.csv")
        cleaned_path = tmp_out_path.with_name(f"{tmp_out_path.stem}_table{tmp_out_path.suffix}")
        try:
            actual_out = clean_table(tmp_in_path, tmp_out_path, "Proyecto")
            cleaned = actual_out.read_text(encoding="utf-8")
            cleaned_lines = [line for line in cleaned.splitlines() if line.strip()]
            if len(cleaned_lines) > 1:
                return cleaned
            return content
        except Exception:
            return content
        finally:
            tmp_in_path.unlink(missing_ok=True)
            tmp_out_path.unlink(missing_ok=True)
            cleaned_path.unlink(missing_ok=True)

    def load_cache(self) -> Dict[str, int]:
        paths: Dict[str, int] = {}
        if self.cache_file.exists():
            with open(self.cache_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if len(row) >= 2:
                        paths[row[0]] = int(row[1])
        return paths

    def save_cache(self) -> None:
        with open(self.cache_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for path, size in self.cache_dict.items():
                print(f"Guardado en CSV: {path} ({size} bytes)")
                writer.writerow([path, size])

    def _extract_document_only(
        self,
        *,
        file_path: Path,
        kind: PdfKind,
        value_builder: Any,
    ) -> None:
        pattern = PATTERN_TABLE if kind == PdfKind.TABULAR else PATTERN_DOCUMENT
        res = self.rule_based_extractor.extract_document(
            path=file_path,
            pattern=pattern,
            value_builder=value_builder,
            create_year_entity=True,
        )
        self.entity_extractor.add_entities(res.entities)
        self.entity_extractor.add_relationship(res.relationships)
        self.entity_extractor.res.errors.extend(res.errors)
        self.entity_extractor._build_doc_indexes()
        self.processed_files.append(str(file_path))

    def _extract_doc_and_chunks(
        self,
        *,
        file_path: Path,
        kind: PdfKind,
        value_builder: Any,
        chunk_file: Path,
    ) -> None:
        pattern = PATTERN_DOCUMENT if kind == PdfKind.NARRATIVE else PATTERN_TABLE

        res = self.rule_based_extractor.extract_document(
            path=file_path,
            pattern=pattern,
            value_builder=value_builder,
            create_year_entity=True,
        )
        self.entity_extractor.add_entities(res.entities)
        self.entity_extractor.add_relationship(res.relationships)
        self.entity_extractor.res.errors.extend(res.errors)

        self.entity_extractor._build_doc_indexes()

        res2 = self.rule_based_extractor.extract_chunk(
            chunk_file,
            self.entity_extractor.doc_by_basename,
        )
        self.entity_extractor.add_entities(res2.entities)
        self.entity_extractor.add_relationship(res2.relationships)
        self.entity_extractor.res.errors.extend(res2.errors)

        self.processed_files.append(str(file_path))

    def _extract_projects_and_responsibles(self) -> None:
        try:
            res = self.entity_extractor.tabular.extract_from_directory(
                self.entity_extractor.table_dir, self.entity_extractor.id_projects
            )
            self.entity_extractor.add_entities(res.entities)
            self.entity_extractor.add_relationship(res.relationships)
            self.entity_extractor.res.errors.extend(res.errors)

        except Exception as e:
            self.entity_extractor.res.errors.append(
                {
                    "type": "ProjectsResponsiblesError",
                    "message": str(e),
                }
            )

    def _collect_entity_dicts(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entity_dicts = [e.to_dict() for e in self.entity_extractor.res.entities]
        rel_dicts = [r.to_dict() for r in self.entity_extractor.res.relationships]
        return entity_dicts, rel_dicts

    def _save_entities_json(
        self,
        entities: List[Dict[str, Any]],
        rels: List[Dict[str, Any]],
    ) -> Optional[Path]:
        if not self.entity_extractor.doc_by_id:
            if not entities:
                print("No hay entidades para guardar.")
                return None
            filename = "entity_extraction_web_tabular.json"
        else:
            first_key = next(iter(self.entity_extractor.doc_by_id.keys()))
            filename = f"entity_extraction_web_{first_key}.json"
        path = self.entities_dir / filename

        new_data = {
            "entities": entities,
            "relationships": rels,
            "errors": self.entity_extractor.res.errors,
        }

        if path.exists():
            try:
                existing_data = json.loads(path.read_text(encoding="utf-8"))

                if not isinstance(existing_data, dict):
                    existing_data = {}

            except (json.JSONDecodeError, OSError):
                existing_data = {}

            existing_entities = existing_data.get("entities", [])
            existing_relationships = existing_data.get("relationships", [])
            existing_errors = existing_data.get("errors", [])

            if not isinstance(existing_entities, list):
                existing_entities = []
            if not isinstance(existing_relationships, list):
                existing_relationships = []
            if not isinstance(existing_errors, list):
                existing_errors = []

            merged_data = {
                "entities": existing_entities + new_data["entities"],
                "relationships": existing_relationships + new_data["relationships"],
                "errors": existing_errors + new_data["errors"],
            }
        else:
            merged_data = new_data

        path.write_text(
            json.dumps(merged_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    def _ingest_neo4j(self, entity_json_path: Path) -> None:
        try:
            if self.user is None or self.password is None:
                raise ValueError("Faltan credenciales de Neo4j: user/password")

            entities, relationships = load_graph_json(
                entity_json_path,
                self.neo4j_uri,
                self.user,
                self.password,
            )
            self.builder.ingest(entities=entities, relationships=relationships)

            print("\n" + "=" * 80)
            print("PIPELINE FINALIZADO")
            print("Archivos procesados:", self.processed_files)
            print("=" * 80)

        except Exception as e:
            self.entity_extractor.res.errors.append(
                {
                    "type": "Neo4jIngestError",
                    "message": str(e),
                }
            )

    def _cleanup_processed_file(self, base_name: str) -> None:
        """
        Limpieza robusta: borra todo lo que empiece con base_name en cada carpeta.
        Así se evitan problemas de .pdf.pdf, etc.
        """
        dirs = (self.docling_dir, self.chunks_dir)
        for d in dirs:
            for p in d.glob(f"{base_name}*"):
                try:
                    p.unlink(missing_ok=True)
                except Exception as e:
                    print(f"[cleanup] No se pudo borrar {p}: {e}")
