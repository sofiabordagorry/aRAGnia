from pathlib import Path

from docling.document_converter import DocumentConverter

DATA_DIR = Path(__file__).resolve().parents[4] / "data"
DEFAULT_CORPUS_DIR = DATA_DIR / Path("corpus")
DEFAULT_DOCLING_DIR = DATA_DIR / Path("docling")
SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md", ".txt"]
GLOB_PATTERNS = [f"*{ext}" for ext in SUPPORTED_EXTENSIONS]

class DocumentAlreadyProcessed(Exception):
    """El documento ya estaba en cache."""

    pass

def get_input_paths(corpus_dir: Path, recursive: bool = False) -> list[Path]:
    """Devuelve la lista de archivos de documentos soportados en un directorio."""
    corpus_path = Path(corpus_dir)
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Directorio de corpus no encontrado: {corpus_path}")
    if not corpus_path.is_dir():
        raise NotADirectoryError(f"La ruta del corpus no es un directorio: {corpus_path}")

    files: list[Path] = []
    
    for pattern in (f"**/{p}" if recursive else p for p in GLOB_PATTERNS):
        files.extend(p for p in corpus_path.glob(pattern) if p.is_file())

    return sorted(files)

def exists_docling(path: Path) -> bool:
    """Devuelve True si la version JSON del documento ya existe en el directiorio data/docling."""
    json_twin = DEFAULT_DOCLING_DIR / path.with_suffix(".json").name
    if json_twin.exists():
        raise DocumentAlreadyProcessed(f"El documento {path.name} ya había sido convertido.")
    else:
        return False
    
def parse_single_document(source: Path) -> dict:
    """Para pruebas: intenta parsear un único documento sin importar su formato."""
    if not exists_docling(path=source):
        try:
            # Inicializar el conversor
            converter = DocumentConverter()
            # Convertir el archivo y devolverlo
            result = converter.convert(source)
            doc_dict = result.document.export_to_dict()
            return doc_dict
        except Exception as e:
            print(f"✗ Error al convertir {source.name}: {e}")

def parse_corpus(
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
    recursive: bool = False,
    skip_errors: bool = True,
) -> list[dict]:
    
    try:
        # Obtener todos los archivos válidos
        input_paths = get_input_paths(corpus_dir=corpus_dir, recursive=recursive)
        # Filtrar archivos ya procesados
        filtered_paths = []
        for p in input_paths:
            try:
                if not exists_docling(p):
                    filtered_paths.append(p)
            except DocumentAlreadyProcessed:
                continue

        input_paths = filtered_paths

        # Inicializar el conversor
        doc_converter = DocumentConverter()

        # Convertir los archivos y devolverlos
        conv_results = doc_converter.convert_all(source = input_paths, raises_on_error = not skip_errors)
        return [res.document.export_to_dict() for res in conv_results]
    except Exception as e:
            print(f"✗ Error al convertir el directiorio {corpus_dir.name}: {e}")
