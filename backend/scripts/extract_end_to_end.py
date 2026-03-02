import asyncio
from pathlib import Path
from typing import Optional

from institutional_graphrag.services.ingest_service import IngestService

# ==============================
# CONFIGURACIÓN
# ==============================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
ENV_PATH: Optional[Path] = Path(BACKEND_DIR / ".env") 
ENABLE_RESEARCHER_CONSOLIDATION = True

INPUT_PATH = "%2F2_PROYECTOS%20I%2BD_2012_2014_2016_2018_2020%2Fid2014_informes_vs_propuestas%2Finformes_propuestas_2014%2F21"  


# ==============================
# MAIN ASYNC
# ==============================

async def main(keep_debug_artifacts: bool):
    print("🚀 Inicializando IngestService...")

    service = IngestService(
        data_dir=DATA_DIR,
        env_path=ENV_PATH,
        enable_researcher_consolidation=ENABLE_RESEARCHER_CONSOLIDATION,
        keep_debug_artifacts=keep_debug_artifacts,
    )

    print("📥 Ejecutando ingest...")

    result = await service.ingest_items(INPUT_PATH)

    print("✅ Resultado:")
    print(result)


# ==============================
# ENTRYPOINT
# ==============================

if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Ejecutar end to end")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mantiene archivos intermedios para debugging (no limpia)",
    )

    args = parser.parse_args()

    keep_debug_artifacts = args.debug

    print(f"[CONFIG] MODO DEBUG: {'ACTIVO' if keep_debug_artifacts else 'DESACTIVADO'}")

    asyncio.run(main(keep_debug_artifacts))