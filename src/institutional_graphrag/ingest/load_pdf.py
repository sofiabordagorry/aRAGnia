import logging
from pathlib import Path
from typing import Iterator, List

from langchain_core.documents import Document
from langchain_docling import DoclingLoader

DEFAULT_CORPUS_DIR = Path("data/corpus")

class DocumentLoadError(Exception):
    """Error al cargar o convertir un documento."""
    pass


class LoadedDoc:
    """Representa un documento cargado con su metadata."""
    def __init__(self, path: Path, documents: List[Document]):
        self.path = path
        self.documents = documents

    def __repr__(self) -> str:
        return f"LoadedDoc(path={self.path!s}, chunks={len(self.documents)})"


def iter_document_paths(corpus_dir: Path, recursive: bool = False) -> Iterator[Path]:
    """Itera sobre archivos de documentos soportados en un directorio."""
    corpus_dir = Path(corpus_dir)
    if not corpus_dir.exists():
        raise FileNotFoundError(f"Directorio de corpus no encontrado: {corpus_dir}")
    if not corpus_dir.is_dir():
        raise NotADirectoryError(f"La ruta del corpus no es un directorio: {corpus_dir}")

    extensions = ["*.pdf", "*.docx", "*.pptx", "*.xlsx", "*.html", "*.md", "*.txt"]
    pattern_prefix = "**/" if recursive else ""
    
    files = []
    for ext in extensions:
        pattern = pattern_prefix + ext
        files.extend(p for p in corpus_dir.glob(pattern) if p.is_file())
    
    yield from sorted(files)


def load_document(path: Path) -> LoadedDoc:
    """
    Carga un documento usando DoclingLoader de LangChain.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Documento no encontrado: {path}")
    
    supported_extensions = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"}
    if path.suffix.lower() not in supported_extensions:
        raise DocumentLoadError(
            f"Formato de archivo no soportado: {path.suffix}. "
            f"Formatos soportados: {', '.join(sorted(supported_extensions))}"
        )

    try:
        loader = DoclingLoader(file_path=str(path))
        documents = loader.load()
        return LoadedDoc(path=path, documents=documents)
    except Exception as e:
        raise DocumentLoadError(f"Error al convertir documento con Docling: {path}") from e


def load_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    recursive: bool = False,
    skip_errors: bool = True,
) -> Iterator[LoadedDoc]:
    """
    Carga todos los documentos de un directorio.
    """
    for doc_path in iter_document_paths(corpus_dir, recursive=recursive):
        try:
            yield load_document(doc_path)
        except Exception as e:
            print(f"Error al cargar {doc_path}: {e}")
            if not skip_errors:
                raise