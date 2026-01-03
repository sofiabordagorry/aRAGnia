"""
Módulo para generar chunks a partir de la estructura del documento.

Implementa chunking por sección cuando es posible, con fallback por tamaño
para prevenir chunks demasiado grandes que excedan límites de tokens.
"""

from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, cast

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from institutional_graphrag.ingest.load_pdf import LoadedDoc

DEFAULT_MAX_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 200


class SectionBasedChunker:
    """
    Chunker que respeta la estructura de secciones del documento.

    Si los chunks por sección son muy grandes (superan max_chunk_size),
    se dividen adicionalmente usando RecursiveCharacterTextSplitter.
    """

    def __init__(
        self,
        max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        length_function: Optional[Callable[[str], int]] = None,
    ):
        """
        Inicializa el chunker con parámetros de tamaño.

        Args:
            max_chunk_size:
                Tamaño máximo de chunk en unidades de `length_function`.
                - Si length_function es len (por defecto): caracteres.
                - Si length_function cuenta tokens: tokens.
            chunk_overlap:
                Solapamiento entre chunks (según len o tokens).
            length_function:
                Función que mide la longitud de un texto (len por defecto).
                Idealmente, una función que cuente tokens del modelo que se está usando (esto depende del modelo de embedding y de las funciones que se tengan disponibles, por ej usando Hugging Face).
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function or len

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.length_function,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Genera chunks a partir de una lista de documentos.

        Args:
            documents: Lista de documentos de LangChain con metadata.

        Returns:
            Lista de chunks como documentos de LangChain.
        """
        chunks: List[Document] = []

        # Agrupar documentos por sección basándose en headings
        sections = self._group_by_section(documents)

        for section_heading, section_docs in sections:
            # Intentar crear uno o más chunks por sección
            section_chunks = self._chunk_section(section_heading, section_docs)
            chunks.extend(section_chunks)

        return chunks

    def _group_by_section(
        self, documents: List[Document]
    ) -> List[Tuple[Optional[str], List[Document]]]:
        """
        Agrupa documentos por sección basándose en headings.

        - Los documentos con el mismo heading se agrupan juntos.
        - Documentos sin heading se tratan individualmente.

        Returns:
            Lista de tuplas (heading, documentos).
        """
        sections: List[Tuple[Optional[str], List[Document]]] = []
        current_heading: Optional[str] = None
        current_docs: List[Document] = []

        for doc in documents:
            # Extraer heading de la metadata de Docling
            heading = self._extract_heading(doc)

            # Si no hay heading, se trata el doc individualmente
            if heading is None:
                # Cerrar la sección actual, si había
                if current_docs:
                    sections.append((current_heading, current_docs))
                    current_docs = []
                    current_heading = None

                sections.append((None, [doc]))
                continue

            # Caso normal: documento con heading
            if heading != current_heading:
                # Nueva sección
                if current_docs:
                    sections.append((current_heading, current_docs))
                current_heading = heading
                current_docs = [doc]
            else:
                # Misma sección, agregar documento
                current_docs.append(doc)

        # Agregar última sección abierta
        if current_docs:
            sections.append((current_heading, current_docs))

        return sections

    def _extract_heading(self, doc: Document) -> Optional[str]:
        """
        Extrae el heading de la metadata de Docling.

        Returns:
            El primer heading encontrado o None.
        """
        try:
            dl_meta = doc.metadata.get("dl_meta", {})
            headings = dl_meta.get("headings", [])
            if headings and isinstance(headings[0], str):
                return headings[0]
        except (AttributeError, KeyError, TypeError, IndexError):
            pass

        return None

    def _chunk_section(
        self, heading: Optional[str], section_docs: List[Document]
    ) -> List[Document]:
        """
        Genera chunks para una sección.

        Si el contenido de la sección cabe en max_chunk_size, se crea un solo chunk.
        Si no, se divide usando el text splitter.

        Args:
            heading: Título de la sección (puede ser None).
            section_docs: Documentos que forman la sección.

        Returns:
            Lista de chunks para esta sección.
        """
        if not section_docs:
            return []

        # limpiamos headings duplicados dentro de la sección (o sea, que no estén en el "texto" del chunk pero si en la metadata)
        cleaned_contents = []

        for idx, doc in enumerate(section_docs):
            content = doc.page_content

            # Si tenemos heading y NO es el primer doc de la sección,
            # chequeamos si la primera línea es exactamente el heading y la removemos.
            if heading and idx > 0:
                lines = content.splitlines()
                if lines and lines[0].strip() == heading.strip():
                    lines = lines[1:]
                    content = "\n".join(lines)

            cleaned_contents.append(content)

        # Combinar contenido de la sección ya limpiado
        section_content = "\n\n".join(cleaned_contents)

        # Preparar metadata base (del primer documento, sin mutarlo)
        base_metadata = dict(section_docs[0].metadata) if section_docs[0].metadata else {}

        # Agregar información de la sección
        if heading:
            base_metadata["section_heading"] = heading

        # Si la sección cabe en el límite, retornar como un solo chunk
        if self.length_function(section_content) <= self.max_chunk_size:
            return [
                Document(
                    page_content=section_content,
                    metadata={
                        **base_metadata,
                        "chunk_method": "section",
                        "section_doc_count": len(section_docs),
                    },
                )
            ]

        # Fallback: dividir por tamaño usando text splitter
        temp_doc = Document(page_content=section_content, metadata=base_metadata)
        split_docs = cast(List[Document], self.text_splitter.split_documents([temp_doc]))

        # Agregar metadata adicional a los chunks
        total_chunks = len(split_docs)
        for i, chunk_doc in enumerate(split_docs):
            chunk_doc.metadata.update(
                {
                    "chunk_method": "section_split",
                    "section_heading": heading,
                    "section_doc_count": len(section_docs),
                    "section_chunk_index": i,
                    "section_chunk_total": total_chunks,
                }
            )

        return split_docs


def chunk_loaded_doc(
    loaded_doc: "LoadedDoc",
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    length_function: Optional[Callable[[str], int]] = None,
) -> List[Document]:
    """
    Función de conveniencia para hacer chunking de un LoadedDoc.

    Args:
        loaded_doc: Instancia de LoadedDoc con documentos cargados.
        max_chunk_size: Tamaño máximo de chunk en unidades de length_function.
        chunk_overlap: Solapamiento entre chunks.
        length_function: Función que mide la longitud del texto.

    Returns:
        Lista de chunks como documentos.
    """
    chunker = SectionBasedChunker(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    documents: List[Document] = loaded_doc.documents
    return chunker.chunk_documents(documents)


def chunk_documents(
    documents: List[Document],
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    length_function: Optional[Callable[[str], int]] = None,
) -> List[Document]:
    """
    Función de conveniencia para hacer chunking de una lista de documentos.

    Args:
        documents: Lista de documentos de LangChain.
        max_chunk_size: Tamaño máximo de chunk en unidades de length_function.
        chunk_overlap: Solapamiento entre chunks.
        length_function: Función que mide la longitud del texto.

    Returns:
        Lista de chunks como documentos.
    """
    chunker = SectionBasedChunker(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
    )
    return chunker.chunk_documents(documents)
