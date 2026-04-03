# institutional_graphrag/services/ingest_service.py
from __future__ import annotations

import csv
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import numpy as np
import requests
from docling_core.types.doc import DoclingDocument
from dotenv import load_dotenv

from institutional_graphrag.config import EMBED_MODEL_ID
from institutional_graphrag.document_naming import (
    PATTERN_DOCUMENT_WITH_OPTIONAL_PDF as PATTERN_DOCUMENT,
)
from institutional_graphrag.document_naming import (
    PATTERN_TABLE,
)
from institutional_graphrag.extraction.ie import EntityExtractor
from institutional_graphrag.extraction.rule_based_extractor import RuleBasedExtractor
from institutional_graphrag.graph.builder import GraphBuilder
from institutional_graphrag.graph.graph_loader import load_graph_json
from institutional_graphrag.ingest.chunker import chunk_document, get_native_chunker
from institutional_graphrag.ingest.docling_parser import parse_single_document
from institutional_graphrag.ingest.embedder import E5Embedder
from institutional_graphrag.ingest.file_namer import (
    PdfKind,
    classify_pdf,
    generate_new_filename,
    save_temp_file,
)
from institutional_graphrag.ingest.postprocess_entities import Postprocessor
from institutional_graphrag.ingest.table_extractors import extract_table


class MissingNeo4jCredentialsError(Exception):
    pass


class IngestService:
    def __init__(
        self,
        *,
        data_dir: Path,
        env_path: Optional[Path] = None,
        enable_researcher_consolidation: bool,
        keep_debug_artifacts: bool,
        include_headings: bool = True,
    ):
        self.enable_researcher_consolidation = enable_researcher_consolidation
        self.keep_debug_artifacts = keep_debug_artifacts
        self.data_dir = data_dir
        self.tables_dir = data_dir / "tables"
        self.output_dir = data_dir / "corpus"
        self.embedding_dir = data_dir / "embeddings"
        self.docling_dir = data_dir / "docling"
        self.chunks_dir = data_dir / "chunks"
        self.entities_dir = data_dir / "entities_relations"

        for d in (
            self.tables_dir,
            self.output_dir,
            self.embedding_dir,
            self.docling_dir,
            self.chunks_dir,
            self.entities_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

        if env_path is not None:
            load_dotenv(env_path)

        self.corpus_token = os.getenv("FING_TOKEN")
        self.cache_file = data_dir / "cache_paths.csv"
        self.cache_file.touch(exist_ok=True)
        self.tokenizer = EMBED_MODEL_ID
        self.chunker = get_native_chunker(tokenizer=self.tokenizer)
        self.embedder = E5Embedder()

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

    async def ingest_items(self, path: str) -> Dict[str, Any]:
        """
        Descarga una carpeta/archivo desde la nube (zip o archivo individual),
        procesa cada PDF y al final ingesta al grafo y hace cleanup.
        """
        self.cache_dict = self.load_cache()

        zf = self.download_path(path)
        if zf is None:
            return {"processed": [], "errors": ["No se pudo descargar el contenido."]}

        print(f"Procesando carpeta/archivo {path} ...")

        processed: List[str] = []
        try:
            for zi in zf.infolist():
                try:
                    base_name = await self._process_one(zi, zf, path)
                    if base_name:
                        processed.append(base_name)
                except Exception as e:
                    # mantener formato consistente (strings)
                    self.entity_extractor.res.errors.append(
                        {
                            "type": "ProcessOneError",
                            "file": zi.filename,
                            "message": str(e),
                        }
                    )
        finally:
            try:
                zf.close()
            except Exception:
                pass

        # Post-loop: proyectos + responsables
        self._extract_projects_and_responsibles()

        # Extracción con LLM (si tu EntityExtractor lo soporta)
        self.entity_extractor._extract_with_llm(include_headings=self.include_headings)

        # Normalización
        entity_dicts, rel_dicts = self._postprocess_entities()

        # Guardar JSON normalizado
        entity_json_path = self._save_entities_json(entity_dicts, rel_dicts)

        # Persistir a Neo4j
        if entity_json_path is not None:
            self._ingest_neo4j(entity_json_path)

        # Guardar en cache los documentos extraidos
        self.save_cache()

        # Cleanup al final
        if not self.keep_debug_artifacts:
            for base in processed:
                self._cleanup_processed_file(base)

            if entity_json_path is not None:
                entity_json_path.unlink(missing_ok=True)

        self.builder.close()
        return {"processed": processed, "errors": self.entity_extractor.res.errors}

    async def _process_one(
        self,
        zi: zipfile.ZipInfo,
        zf: zipfile.ZipFile,
        path_encoded: str,
    ) -> Optional[str]:
        if zi.is_dir():
            return None

        # bytes reales del archivo dentro del zip
        content = zf.read(zi)

        relative_path = Path(zi.filename)
        if relative_path.parts and relative_path.parts[0] == "CSIC VALIDACION INFORMES":
            relative_path = Path(*relative_path.parts[1:])
        full_cloud_path = "\\" + str(
            Path(unquote(path_encoded).strip("/")) / relative_path
        ).replace("/", "\\")
        file_size = zi.file_size

        prev_size = self.cache_dict.get(full_cloud_path)
        if prev_size is not None and prev_size == file_size:
            print(f"Sin cambios (mismo tamaño): {full_cloud_path}")
            return None
        if prev_size is not None and prev_size != file_size:
            print(f"Actualizado (cambió tamaño): {full_cloud_path}")

        self.cache_dict[full_cloud_path] = file_size

        # nombre nuevo (idealmente incluye .pdf)
        new_filename = generate_new_filename(full_cloud_path)
        print("NOMBRE", new_filename)
        base_name = Path(new_filename).stem  # clave: TODO se guarda con base_name

        # paths de salida
        final_pdf_path = self.output_dir / new_filename
        docling_json_path = self.docling_dir / f"{base_name}.json"
        chunk_json_path = self.chunks_dir / f"{base_name}_chunks.json"
        emb_npy_path = self.embedding_dir / f"{base_name}.npy"
        emb_meta_path = self.embedding_dir / f"{base_name}_metadata.json"

        # tmp siempre con bytes
        tmp_path = save_temp_file(content, new_filename)
        try:
            kind = classify_pdf(new_filename)

            # ---------------------------
            # TABLAS
            # ---------------------------
            if kind == PdfKind.TABULAR:
                # Extraer tablas desde el PDF temporal
                extract_table(tmp_path, self.tables_dir)

                # Crear entidad Documento(tabla) (sin chunks)
                def value_builder(base, m):
                    return {
                        "base_name": base,
                        "is_group": m.group("group"),
                        "year_publisher": m.group("year"),
                        "type": "tabla",
                    }

                # Para el extractor por reglas, pasamos un path "representativo".
                # Si tu extractor matchea por nombre, esto funciona.
                self._extract_document_only(
                    file_path=Path(new_filename),
                    kind=kind,
                    value_builder=value_builder,
                )
                return base_name

            # ---------------------------
            # NARRATIVE
            # ---------------------------
            def value_builder(base, m):
                return {
                    "base_name": base,
                    "is_group": m.group("group"),
                    "year_publisher": m.group("year"),
                    "sub_id": m.group("doc_id"),
                    "type": m.group("kind"),
                }

            # Guardar PDF definitivo
            with open(final_pdf_path, "wb") as out:
                out.write(content)

            # Docling
            doc_dict = parse_single_document(final_pdf_path)
            docling_json_path.write_text(
                json.dumps(doc_dict, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # Chunking
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

            # Embeddings
            embeddings = self.embedder.embed_passages([c["text"] for c in chunks])
            np.save(emb_npy_path, embeddings)
            emb_meta_path.write_text(
                json.dumps(chunks, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            # Extracción estática: Documento + Año + Chunk + relaciones
            self._extract_doc_and_chunks(
                file_path=final_pdf_path,  # path real
                kind=kind,
                value_builder=value_builder,
                chunk_file=chunk_json_path,
            )

            return base_name

        finally:
            tmp_path.unlink(missing_ok=True)

    def download_path(self, path_encoded: str) -> Optional[zipfile.ZipFile]:
        """
        Descarga carpeta (zip) o archivo individual y devuelve un ZipFile.
        Si es archivo individual, crea un zip en memoria con ese archivo.
        """
        base_url = f"https://nube.fing.edu.uy/index.php/s/{self.corpus_token}/download"
        url = f"{base_url}?path={path_encoded}&files="
        original_folder_path = unquote(path_encoded).strip("/")

        response = requests.get(url, verify=False, timeout=120)
        if response.status_code != 200:
            print(f"Error al descargar: {unquote(path_encoded)} (status={response.status_code})")
            return None

        try:
            return zipfile.ZipFile(io.BytesIO(response.content))
        except zipfile.BadZipFile:
            mem_zip = io.BytesIO()
            with zipfile.ZipFile(mem_zip, mode="w") as z:
                filename = Path(original_folder_path).name
                z.writestr(filename, response.content)
            mem_zip.seek(0)
            return zipfile.ZipFile(mem_zip)

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
            res = self.rule_based_extractor.associate_tables_with_documents(
                self.entity_extractor.docs_by_group_year,
                self.tables_dir,
            )
            self.entity_extractor.res.errors.extend(res.errors)

            res = self.rule_based_extractor.extract_projects_and_responsible_from_tables(
                self.entity_extractor.doc_by_id,
                self.entity_extractor.chunks_dir,
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

    def _postprocess_entities(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        entity_dicts = [e.to_dict() for e in self.entity_extractor.res.entities]
        rel_dicts = [r.to_dict() for r in self.entity_extractor.res.relationships]
        post_processor = Postprocessor(
            enable_researcher_consolidation=self.enable_researcher_consolidation,
            similarity_threshold=0.85,
        )

        entity_dicts, rel_dicts, _ = post_processor.consolidate_researchers(entity_dicts, rel_dicts)
        entities_db = self.builder.fetch_researchers()
        rel_dicts.extend(
            post_processor.build_possible_alias_relationships(entities_db, entity_dicts)
        )

        rel_dicts, _ = post_processor.add_missing_evidence_text(rel_dicts)
        return entity_dicts, rel_dicts

    def _save_entities_json(
        self,
        entities: List[Dict[str, Any]],
        rels: List[Dict[str, Any]],
    ) -> Optional[Path]:
        if not self.entity_extractor.doc_by_id:
            print(
                "No hay documentos indexados (doc_by_id vacío). No se guarda entity_extraction_web."
            )
            return None

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
        Así evitás problemas de .pdf.pdf, etc.
        """
        dirs = (self.output_dir, self.embedding_dir, self.docling_dir, self.chunks_dir)
        for d in dirs:
            for p in d.glob(f"{base_name}*"):
                try:
                    p.unlink(missing_ok=True)
                except Exception as e:
                    print(f"[cleanup] No se pudo borrar {p}: {e}")
