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
            int(controls.loc[controls["Control"].eq("Apoyo OFF en rutas sin Titán/Fénix para cuota CDA"), "Titulares"].iloc[0]), 0,
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

    def test_insufficient_cda_capacity_keeps_download_and_exact_warning(self):
        quotas = {
            **self.quotas,
            "CDAALFA": {"name": "CDA ALFA", "ON": 11, "OFF": 40},
        }
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                sample_universe(), self.historical, cda_quotas=quotas,
            )
        selected = final.loc[final[config.SELECTION_COL].eq("T")]
        self.assertEqual(len(selected), 75)
        alfa = controls.loc[
            controls["Control"].eq("Cuota CDA por canal")
            & controls["CDA"].eq("CDA ALFA")
            & controls["Canal"].eq("OFF")
        ].iloc[0]
        self.assertEqual((int(alfa["Objetivo"]), int(alfa["Titulares"])), (40, 35))
        self.assertEqual(int(alfa["Capacidad_seleccionable"]), 35)
        self.assertEqual(alfa["Estado"], "REVISAR")
        total = controls.loc[controls["Control"].eq("Muestra total solicitada")].iloc[0]
        self.assertEqual((int(total["Objetivo"]), int(total["Titulares"])), (80, 75))
        self.assertEqual(
            total["Motivo"],
            "Las cuotas CDA no caben en el universo y las reglas de rutas. "
            "CDA ALFA OFF: cuota 40, elegibles 35, capacidad elegible 35",
        )

    def test_on_route_capacity_shortfall_preserves_substitute_reserve(self):
        quotas = {
            **self.quotas,
            "CDAALFA": {"name": "CDA ALFA", "ON": 30, "OFF": 15},
        }
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                sample_universe(), self.historical, cda_quotas=quotas,
            )
        alfa_on = final.loc[
            final["DES LOC_COM"].eq("CDA ALFA") & final["_CANAL_SELECCION"].eq("ON")
        ]
        self.assertEqual(int(alfa_on[config.SELECTION_COL].eq("T").sum()), 24)
        self.assertTrue(alfa_on[config.SELECTION_COL].isin(["S1", "S2", "S3"]).any())
        total = controls.loc[controls["Control"].eq("Muestra total solicitada")].iloc[0]
        self.assertIn("CDA ALFA ON: cuota 30, elegibles 25, capacidad según rutas 24", total["Motivo"])

    def test_blank_nombre_keeps_mandatory_fenix_but_excludes_other_points(self):
        universe = sample_universe()
        blank = universe["DES LOC_COM"].eq("CDA ALFA") & universe["Subcanal2"].eq("OFF")
        first, second = universe.index[blank][:2]
        universe.loc[[first, second], "NOMBRE"] = "nan"
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                universe, self.historical, cda_quotas=self.quotas,
            )
        self.assertEqual(final.loc[final["CODIGO"].eq(universe.loc[first, "CODIGO"]), config.SELECTION_COL].iloc[0], "T")
        self.assertEqual(final.loc[final["CODIGO"].eq(universe.loc[second, "CODIGO"]), config.SELECTION_COL].iloc[0], "NO")
        selected_without_name = final.loc[
            final[config.SELECTION_COL].eq("T") & ~selector_engine._has_nombre(final["NOMBRE"])
        ]
        self.assertEqual(selected_without_name["CODIGO"].tolist(), [universe.loc[first, "CODIGO"]])
        self.assertTrue(controls.loc[controls["Control"].eq("Cuota CDA por canal"), "Estado"].eq("OK").all())
        unnamed_control = controls.loc[
            controls["Control"].eq("Titán/Fénix sin NOMBRE por CDA")
        ]
        self.assertEqual(unnamed_control["CDA"].tolist(), ["CDA ALFA"])
        self.assertEqual(unnamed_control["Titulares"].tolist(), [1])

    def test_blank_nombre_shortfall_is_visible_without_redistribution(self):
        universe = sample_universe()
        alfa_off = universe.index[
            universe["DES LOC_COM"].eq("CDA ALFA") & universe["Subcanal2"].eq("OFF")
        ]
        universe.loc[alfa_off[14:], "NOMBRE"] = "nan"
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                universe, self.historical, cda_quotas=self.quotas,
            )
        selected = final.loc[final[config.SELECTION_COL].eq("T")]
        self.assertEqual(len(selected), 54)
        self.assertTrue(selector_engine._has_nombre(selected["NOMBRE"]).all())
        alfa_control = controls.loc[
            controls["Control"].eq("Cuota CDA por canal")
            & controls["CDA"].eq("CDA ALFA")
            & controls["Canal"].eq("OFF")
        ].iloc[0]
        self.assertEqual((int(alfa_control["Objetivo"]), int(alfa_control["Titulares"])), (15, 14))
        self.assertEqual(int(alfa_control["Elegibles_con_NOMBRE"]), 14)
        self.assertEqual(int(alfa_control["Excluidos_sin_NOMBRE"]), 21)
        self.assertEqual(alfa_control["Estado"], "REVISAR")
        self.assertIn("sin NOMBRE", alfa_control["Motivo"])

    def test_off_cda_without_any_titan_or_fenix_uses_scoped_exception(self):
        universe = sample_universe()
        alfa_off = universe["DES LOC_COM"].eq("CDA ALFA") & universe["Subcanal2"].eq("OFF")
        universe.loc[alfa_off, "Fenix"] = 0
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                universe, self.historical, cda_quotas=self.quotas,
            )
        off_alfa = final.loc[
            final["DES LOC_COM"].eq("CDA ALFA") & final["_CANAL_SELECCION"].eq("OFF")
        ]
        self.assertEqual(int(off_alfa[config.SELECTION_COL].eq("T").sum()), 15)
        self.assertTrue(off_alfa[config.SELECTION_COL].isin(["S1", "S2", "S3"]).any())
        exceptions = controls.loc[controls["Control"].eq("Apoyo OFF sin Titán/Fénix por CDA")]
        self.assertEqual(exceptions["CDA"].tolist(), ["CDA ALFA"])
        self.assertFalse(controls["Estado"].eq("REVISAR").any())
        self.assertNotIn("CDA_OFF_ROUTE_EXCEPTION", final.columns)

    def test_cda_with_one_fenix_uses_unqualified_routes_only_for_remaining_quota(self):
        universe = sample_universe()
        alfa_off = universe.loc[
            universe["DES LOC_COM"].eq("CDA ALFA") & universe["Subcanal2"].eq("OFF")
        ].index
        universe.loc[alfa_off[10:], "RUT.COM"] = "CDA ALFA-OFF-UNQUALIFIED"
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                universe, self.historical, cda_quotas=self.quotas,
            )
        alfa_titles = final.loc[
            final["DES LOC_COM"].eq("CDA ALFA")
            & final["_CANAL_SELECCION"].eq("OFF")
            & final[config.SELECTION_COL].eq("T")
        ]
        self.assertEqual(int(alfa_titles["RUT.COM"].eq("CDA ALFA-OFF").sum()), 10)
        self.assertEqual(int(alfa_titles["RUT.COM"].eq("CDA ALFA-OFF-UNQUALIFIED").sum()), 5)
        support = controls.loc[controls["Control"].eq("Apoyo OFF sin Titán/Fénix por CDA")]
        self.assertEqual(support["CDA"].tolist(), ["CDA ALFA"])
        self.assertEqual(int(support["Titulares"].iloc[0]), 5)
        self.assertFalse(controls["Estado"].eq("REVISAR").any())

    def test_qualified_routes_suffice_without_using_fallback(self):
        universe = sample_universe()
        alfa_off = universe.loc[
            universe["DES LOC_COM"].eq("CDA ALFA") & universe["Subcanal2"].eq("OFF")
        ].index
        universe.loc[alfa_off[20:], "RUT.COM"] = "CDA ALFA-OFF-UNQUALIFIED"
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                universe, self.historical, cda_quotas=self.quotas,
            )
        alfa_titles = final.loc[
            final["DES LOC_COM"].eq("CDA ALFA")
            & final["_CANAL_SELECCION"].eq("OFF")
            & final[config.SELECTION_COL].eq("T")
        ]
        self.assertTrue(alfa_titles["RUT.COM"].eq("CDA ALFA-OFF").all())
        self.assertFalse(controls["Control"].eq("Apoyo OFF sin Titán/Fénix por CDA").any())

    def test_lima_minimum_counts_unlabeled_points_in_same_cda(self):
        universe = sample_universe().iloc[:3].copy()
        universe["NOMBRE"] = "78MARTINEZVENTANILLA"
        universe["CDA_ORIG_CITY"] = ["LIMA", "nan", "LIMA"]
        universe["CDA_QUOTA_KEY"] = "78MARTINEZVENTANILLA"
        universe["Subcanal2"] = "OFF"
        universe["CANAL"] = "OFF"
        universe["LOC.COM"] = "78H"
        universe["Fenix"] = [1, 1, 0]
        prepared = selector_engine._prepare(universe)
        selector_engine._set_selected(prepared, prepared.index[prepared["_mandatory"]])
        selector_engine._ensure_lima_minimums(
            prepared, {"78H": 2}, allow_off_route_fallback=True,
        )
        self.assertEqual(int(prepared[config.SELECTION_COL].eq("T").sum()), 2)

    def test_default_off_mode_ignores_exception_column_from_input(self):
        universe = sample_universe()
        universe = universe.loc[universe["Subcanal2"].eq("OFF")].copy()
        alfa = universe["DES LOC_COM"].eq("CDA ALFA")
        universe.loc[alfa, "Fenix"] = 0
        universe["CDA_OFF_ROUTE_EXCEPTION"] = alfa
        index = pd.MultiIndex.from_tuples([("ICA", "OFF")], names=selector_engine.CITY_GROUP)
        targets = pd.Series([40], index=index)
        strategy_index = pd.MultiIndex.from_tuples(
            [("ICA", "OFF", "GR")], names=selector_engine.STRATEGY_GROUP,
        )
        strategy_targets = pd.Series([1], index=strategy_index)
        with self.assertRaisesRegex(ValueError, "no cabe completamente en rutas"):
            selector_engine.select_canal_off(
                universe, targets, strategy_targets, {},
            )

    def test_on_fenix_in_cda_does_not_block_off_quota_fallback(self):
        universe = sample_universe()
        alfa_off = universe["DES LOC_COM"].eq("CDA ALFA") & universe["Subcanal2"].eq("OFF")
        alfa_on = universe.loc[
            universe["DES LOC_COM"].eq("CDA ALFA") & universe["Subcanal2"].eq("ON")
        ].index
        universe.loc[alfa_off, "Fenix"] = 0
        universe.loc[alfa_on[0], "Fenix"] = 1
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                universe, self.historical, cda_quotas=self.quotas,
            )
        alfa_off_selected = final.loc[
            final["DES LOC_COM"].eq("CDA ALFA")
            & final["_CANAL_SELECCION"].eq("OFF")
            & final[config.SELECTION_COL].eq("T")
        ]
        self.assertEqual(len(alfa_off_selected), 15)
        self.assertFalse(controls["Estado"].eq("REVISAR").any())

    def test_minor_on_route_balance_is_reported_without_blocking_download(self):
        quotas = {
            **self.quotas,
            "CDAALFA": {"name": "CDA ALFA", "ON": 12, "OFF": 15},
            "CDABETA": {"name": "CDA BETA", "ON": 10, "OFF": 18},
        }
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                sample_universe(), self.historical, cda_quotas=quotas,
            )
        self.assertEqual(int(final[config.SELECTION_COL].eq("T").sum()), 55)
        balance = controls.loc[
            controls["Control"].eq("Titulares especializados por ruta ON (10–30 y balance)")
            & controls["Estado"].eq("REVISAR")
        ]
        self.assertFalse(balance.empty)
        self.assertTrue(balance["Titulares"].between(10, 30).all())
        self.assertTrue(balance["Suplentes"].gt(0).all())

    def test_historical_mix_deviation_is_reported_without_blocking_download(self):
        historical = self.historical.copy()
        historical["ESTRATEGICA"] = "EG"
        with patch.object(config, "load_lima_quotas", return_value={}):
            final, controls = selector_engine.run_selection_process(
                sample_universe(), historical, cda_quotas=self.quotas,
            )
        self.assertEqual(int(final[config.SELECTION_COL].eq("T").sum()), 55)
        self.assertTrue(controls["Control"].eq("Variación por estrategia (máximo +/- 1 pp)").any())
        self.assertTrue(controls.loc[
            controls["Control"].eq("Variación por estrategia (máximo +/- 1 pp)"), "Estado"
        ].eq("REVISAR").any())
        self.assertTrue(controls.loc[
            controls["Control"].eq("Cuota CDA por canal"), "Estado"
        ].eq("OK").all())

    def test_mandatory_control_still_blocks_cda_download(self):
        real_build_controls = selector_engine.build_controls

        def mark_mandatory_failure(*args, **kwargs):
            controls = real_build_controls(*args, **kwargs)
            mask = controls["Control"].eq("Titán/Titán Plus/Fénix obligatorios")
            controls.loc[mask, "Estado"] = "REVISAR"
            controls.loc[mask, "Motivo"] = "Falta un titular obligatorio"
            return controls

        with patch.object(config, "load_lima_quotas", return_value={}), patch.object(
            selector_engine, "build_controls", side_effect=mark_mandatory_failure,
        ):
            with self.assertRaisesRegex(ValueError, "Titán/Titán Plus/Fénix obligatorios"):
                selector_engine.run_selection_process(
                    sample_universe(), self.historical, cda_quotas=self.quotas,
                )


if __name__ == "__main__":
    unittest.main()
