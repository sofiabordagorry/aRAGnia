"""Prueba rápida de carga de PDFs."""

from pathlib import Path
from institutional_graphrag.ingest.load_pdf import load_document

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
print(DATA_DIR)
pdf = list((DATA_DIR / Path("corpus")).glob("*.pdf"))[0]
print(f"Cargando: {pdf.name}")

doc = load_document(pdf)

print(f"Chunks: {len(doc.documents)}")
print(f"\nPrimeros 200 chars:\n{doc.documents[0].page_content[:200]}")
