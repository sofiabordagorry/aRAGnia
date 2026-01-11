import unittest
import sys
from pathlib import Path

from institutional_graphrag.ingest.file_namer import generate_new_filename

# Add project root to sys.path to allow importing from src
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))


class TestNamingConvention(unittest.TestCase):

    def test_gi_informe(self):
        path = r"\1_GRUPOS I+D_2010_2014_2018\2014_Informes finales Grupos I+D\140\Informe final\INFORME FINAL GRUPOS 2019 versión final.pdf"
        self.assertEqual(generate_new_filename(path), "gi_2014_140_informe.pdf")

    def test_gi_propuesta(self):
        path = r"\1_GRUPOS I+D_2010_2014_2018\2014_Informes finales Grupos I+D\140\Propuesta postulación_2014\Propuesta.pdf"
        self.assertEqual(generate_new_filename(path), "gi_2014_140_propuesta.pdf")

    def test_proy_informe(self):
        path = r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2012_Informes_vs_propuestas\informes_propuestas_2012\28\Informe publicable\362_documentoarchivo.pdf"
        self.assertEqual(generate_new_filename(path), "proy_2012_28_informe.pdf")

    def test_proy_propuesta(self):
        path = r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2012_Informes_vs_propuestas\informes_propuestas_2012\28\Propuesta_postulación\1305_avales.pdf"
        self.assertEqual(generate_new_filename(path), "proy_2012_28_propuesta.pdf")

    def test_admin_simple(self):
        path = r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2014_informes_vs_propuestas\Resultado llamado_Informe CSIC_gruposI+D_2014.pdf"
        self.assertEqual(
            generate_new_filename(path),
            "admin_2014_Resultado llamado_Informe CSIC_gruposI+D_2014.pdf",
        )

    def test_admin_closest_year(self):
        # Checks that 2018 is picked over 2020
        path = r"\2_PROYECTOS I+D_2012_2014_2016_2018_2020\id2020_informes_vs_propuestas\id2018p_informes_vs__propuestas\planilla financiados_proyectosI+D.pdf"
        self.assertEqual(
            generate_new_filename(path), "admin_2018_planilla financiados_proyectosI+D.pdf"
        )


if __name__ == "__main__":
    unittest.main()
