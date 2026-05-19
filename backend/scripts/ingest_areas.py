import os
import unicodedata
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from institutional_graphrag.graph.schema import Area
from institutional_graphrag.graph.builder import GraphBuilder

def normalize_id(text: str) -> str:
    """Convierte texto a un ID en minúsculas, sin acentos y con guiones bajos."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip().replace(" ", "_")

def normalize_value(text: str) -> str:
    """Convierte texto a su normalización en minúsculas y sin acentos"""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text.lower().strip()

def main():
    BASE_DIR = Path(__file__).resolve().parents[2]
    tables_dir = BASE_DIR / "data" / "tables"
    clean_csv_path = tables_dir / "equipos_i+d_2012_2018_pinco_clean.csv"

    load_dotenv(BASE_DIR / ".env")
    NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

    df = pd.read_csv(clean_csv_path)

    # Validar que las columnas existan
    if 'area' not in df.columns or 'id_unico' not in df.columns:
        raise ValueError("El CSV no contiene las columnas esperadas ('area' y 'id_unico').")
    
    unique_areas = df['area'].dropna().unique()

    areas_entities = {}
    for area_name in unique_areas:
        area_id = normalize_id(area_name)
        normalized_value = normalize_value(area_name)
        if area_id:
            # Value e id en minúscula y sin acentos
            areas_entities[area_id] = Area(id=area_id, value=normalized_value)

    print(f"Se encontraron {len(areas_entities)} áreas únicas.")

    # 5. Ingestar en Neo4j usando GraphBuilder
    print("Iniciando ingesta en Neo4j...")
    builder = GraphBuilder(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        builder.ingest(
            entities=areas_entities.values(),
            relationships=[]
        )
        print("Ingesta finalizada con éxito")
    except Exception as e:
        print(f"Error durante la ingesta: {e}")
    finally:
        builder.close()

if __name__ == "__main__":
    main()