import unittest
from unittest.mock import patch

import pandas as pd

import config
import selector_engine


def sample_universe():
    rows = []
    code = 1
    for cda in ("CDA ALFA", "CDA BETA"):
        for channel, count in (("ON", 25), ("OFF", 35)):
            for position in range(count):
                rows.append({
                    "CODIGO": code,
                    "NOMBRE": "ICA",
                    "DES LOC_COM": cda,
                    "LOC.COM": cda,
                    "Subcanal2": channel,
                    "CANAL": channel,
                    "RUT.COM": f"{cda}-{channel}",
                    "ESTRATEGICA": "GR",
                    "Fenix": int(channel == "OFF" and position == 0),
                    "Programa de Valor": "0",
                    "ESPECIALIZADA": "SI" if channel == "ON" and position < 18 else "NO",
                    "SEL_MES_ANT": "NO",
                    "Esparta_Flag": 0,
                    "Ventas": float(count - position),
                    "VISITAS_ANT": 0,
                })
                code += 1
    return pd.DataFrame(rows)


class CdaSelectionTests(unittest.TestCase):
    def setUp(self):
        self.quotas = {
            "CDAALFA": {"name": "CDA ALFA", "ON": 11, "OFF": 15},
            "CDABETA": {"name": "CDA BETA", "ON": 11, "OFF": 18},
        }
        self.historical = pd.DataFrame([{
            "NOMBRE": "ICA", "Subcanal2": channel, "ESTRATEGICA": "GR",
            "ES_TITULAR_HISTORICO": True,
        } for channel in ("ON", "OFF")])

    def test_exact_cda_targets_keep_route_and_substitute_rules(self):
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                sample_universe(), self.historical, cda_quotas=self.quotas,
            )
        selected = final.loc[final[config.SELECTION_COL].eq("T")]
        counts = selected.groupby(["DES LOC_COM", "_CANAL_SELECCION"]).size().to_dict()
        self.assertEqual(counts, {
            ("CDA ALFA", "OFF"): 15,
            ("CDA ALFA", "ON"): 11,
            ("CDA BETA", "OFF"): 18,
            ("CDA BETA", "ON"): 11,
        })
        self.assertEqual(len(selected), 55)
        self.assertNotIn("S4", set(final[config.SELECTION_COL]))
        self.assertEqual(
            int(controls.loc[controls["Control"].eq("Selecciones OFF solo en rutas Titán/Fénix"), "Titulares"].iloc[0]), 0,
        )
        self.assertTrue(controls.loc[controls["Control"].eq("Cuota CDA por canal"), "Estado"].eq("OK").all())
        self.assertFalse(controls["Estado"].eq("REVISAR").any())

    def test_unavailable_cda_quota_fails_before_selection(self):
        quotas = {**self.quotas, "NOEXISTE": {"name": "NO EXISTE", "ON": 1, "OFF": 0}}
        with patch.object(config, "load_lima_quotas", return_value={}):
            with self.assertRaisesRegex(ValueError, "NO EXISTE ON"):
                selector_engine.run_selection_process(
                    sample_universe(), self.historical, cda_quotas=quotas,
                )

    def test_incompatible_route_balance_fails_instead_of_exporting(self):
        quotas = {
            **self.quotas,
            "CDAALFA": {"name": "CDA ALFA", "ON": 12, "OFF": 15},
            "CDABETA": {"name": "CDA BETA", "ON": 10, "OFF": 18},
        }
        with patch.object(config, "load_lima_quotas", return_value={}):
            with self.assertRaisesRegex(ValueError, "Titulares especializados por ruta ON"):
                selector_engine.run_selection_process(
                    sample_universe(), self.historical, cda_quotas=quotas,
                )


if __name__ == "__main__":
    unittest.main()
