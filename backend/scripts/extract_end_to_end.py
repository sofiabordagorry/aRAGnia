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

INPUT_PATH = "%2F2_PROYECTOS%20I%2BD_2012_2014_2016_2018_2020%2Fid2014_informes_vs_propuestas%2Finformes_propuestas_2014%2F21"  # <-- CAMBIAR


# ==============================
# MAIN ASYNC
# ==============================

async def main():
    print("🚀 Inicializando IngestService...")

    service = IngestService(
        data_dir=DATA_DIR,
        env_path=ENV_PATH,
        enable_researcher_consolidation=ENABLE_RESEARCHER_CONSOLIDATION,
    )

    print("📥 Ejecutando ingest...")

    result = await service.ingest_items(INPUT_PATH)

    print("✅ Resultado:")
    print(result)


# ==============================
# ENTRYPOINT
# ==============================

if __name__ == "__main__":
    asyncio.run(main())