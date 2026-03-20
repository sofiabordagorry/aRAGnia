import asyncio
from pathlib import Path
from typing import Optional
import urllib.parse

from institutional_graphrag.services.ingest_service import IngestService


# ==============================
# CONFIGURACIÓN
# ==============================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
BACKEND_DIR = Path(__file__).resolve().parents[2] / "backend"
ENV_PATH: Optional[Path] = Path(BACKEND_DIR / ".env") 

# Lista de paths originales
paths = [
    r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2020_informes_vs_propuestas\informe_propusetas_2020\513",
    r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2020_informes_vs_propuestas\id2018p_informes_vs__propuestas\informes_propuestas_2018\413",
    r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2020_informes_vs_propuestas\id2018p_informes_vs__propuestas\informes_propuestas_2018\260",
    r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2016_informes_vs_propuestas\informes_propuestas_2016\729",
    r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2012_Informes_vs_propuestas\informes_propuestas_2012\638",
]

# ==============================
# MAIN ASYNC
# ==============================

async def main():
    print("Inicializando IngestService...")

    print("Ejecutando ingest...")
    url_encoded_paths = []
    for path in paths:
        path_normalized = path.replace("\\", "/")
        path_encoded = urllib.parse.quote(path_normalized)
        url_encoded_paths.append(path_encoded)

    # Imprimir los paths codificados
    for encoded_path in url_encoded_paths:
        service = IngestService(
            data_dir=DATA_DIR,
            env_path=ENV_PATH,
            enable_researcher_consolidation=False,
            keep_debug_artifacts=True,
        )
        result = await service.ingest_items(encoded_path)
        print("Resultados del path ", encoded_path, ":")
        print(result)
    
    
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())