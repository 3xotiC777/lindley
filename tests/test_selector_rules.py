import unittest
from unittest.mock import patch

import pandas as pd

import config
import selector_engine


def route_group(specs):
    rows = []
    code = 1
    for route, available, current, mandatory, non_specialized in specs:
        for position in range(available):
            rows.append({
                "CODIGO": code,
                "RUT.COM": route,
                "Ventas": float(available - position),
                config.SELECTION_COL: "T" if position < current else "NO",
                "_mandatory": position < mandatory,
                "_is_specialized": position >= non_specialized,
            })
            code += 1
    return pd.DataFrame(rows)


def scope_universe(extra_on=20, off_capacity=140):
    """Build enough eligible capacity to exercise all three increase scopes."""
    rows = []
    code = 1
    for city, quota in config.ON_CITY_QUOTAS.items():
        for position in range(quota + extra_on):
            rows.append({
                "CODIGO": code, "NOMBRE": city, "Subcanal2": "ON", "CANAL": "ON",
                "RUT.COM": f"{city}-ON", "ESTRATEGICA": "GR", "Fenix": 0,
                "Programa de Valor": "0", "SEL_MES_ANT": "NO", "ESPECIALIZADA": "NO",
                "LOC.COM": "",
            })
            code += 1
    for position in range(off_capacity):
        rows.append({
            "CODIGO": code, "NOMBRE": "LIMA", "Subcanal2": "OFF", "CANAL": "OFF",
            "RUT.COM": "LIMA-OFF", "ESTRATEGICA": "GR", "Fenix": 1 if position == 0 else 0,
            "Programa de Valor": "0", "SEL_MES_ANT": "NO", "ESPECIALIZADA": "NO",
            "LOC.COM": "OTRA",
        })
        code += 1
    index = pd.MultiIndex.from_tuples(
        [(city, "ON") for city in config.ON_CITY_QUOTAS] + [("LIMA", "OFF")],
        names=selector_engine.CITY_GROUP,
    )
    historical = pd.Series(
        [*config.ON_CITY_QUOTAS.values(), 100], index=index, dtype=int,
    )
    return pd.DataFrame(rows), historical


class OnRouteTargetTests(unittest.TestCase):
    def test_fixed_on_quotas_sum_to_approved_total(self):
        index = pd.MultiIndex.from_tuples(
            [("AREQUIPA", "ON"), ("LIMA", "OFF")],
            names=selector_engine.CITY_GROUP,
        )
        historical = pd.Series([999, 100], index=index, dtype=int)
        fixed = selector_engine._with_fixed_on_quotas(historical)
        on_total = int(fixed[fixed.index.get_level_values("Subcanal2") == "ON"].sum())
        self.assertEqual(on_total, 1285)
        self.assertEqual(int(fixed.get(("HUARAZ", "ON"))), 25)

    def test_increase_scope_off_keeps_on_at_approved_base(self):
        universe, historical = scope_universe()
        targets = selector_engine._scaled_city_targets(
            universe, historical, config.ON_BASE_TOTAL + 112, {}, "OFF",
        )
        target_on = int(sum(
            value for (_city, subchannel), value in targets.items() if subchannel == "ON"
        ))
        self.assertEqual(target_on, config.ON_BASE_TOTAL)
        self.assertEqual(int(targets[("LIMA", "OFF")]), 112)

    def test_on_capacity_reserves_a_substitute_per_optional_route(self):
        universe, historical = scope_universe(extra_on=2)
        _groups, capacity, _lower = selector_engine._city_group_bounds(
            universe, historical, {},
        )
        self.assertEqual(
            capacity[("HUARAZ", "ON")],
            config.ON_CITY_QUOTAS["HUARAZ"] + 1,
        )

    def test_increase_scope_on_keeps_off_at_historical_base(self):
        universe, historical = scope_universe()
        targets = selector_engine._scaled_city_targets(
            universe, historical, config.ON_BASE_TOTAL + 112, {}, "ON",
        )
        target_on = int(sum(
            value for (_city, subchannel), value in targets.items() if subchannel == "ON"
        ))
        self.assertEqual(target_on, config.ON_BASE_TOTAL + 12)
        self.assertEqual(int(targets[("LIMA", "OFF")]), 100)

    def test_increase_scope_both_distributes_above_both_bases(self):
        universe, historical = scope_universe()
        targets = selector_engine._scaled_city_targets(
            universe, historical, config.ON_BASE_TOTAL + 120, {}, "AMBOS",
        )
        target_on = int(sum(
            value for (_city, subchannel), value in targets.items() if subchannel == "ON"
        ))
        self.assertEqual(int(targets.sum()), config.ON_BASE_TOTAL + 120)
        self.assertGreater(target_on, config.ON_BASE_TOTAL)
        self.assertGreater(int(targets[("LIMA", "OFF")]), 100)

    def test_arequipa_is_levelled_across_three_routes(self):
        group = route_group([
            ("ICHE0", 40, 10, 0, 10),
            ("ICHE1", 40, 27, 0, 10),
            ("ICHE2", 40, 28, 0, 10),
        ])
        targets, target_si = selector_engine._route_specialized_targets(group, 65)
        self.assertEqual(target_si, 65)
        self.assertEqual(sorted(targets.values()), [21, 22, 22])

    def test_chiclayo_is_levelled_twenty_five_each(self):
        group = route_group([
            ("I7HE1", 40, 20, 0, 10),
            ("I7HE2", 40, 30, 0, 10),
        ])
        targets, target_si = selector_engine._route_specialized_targets(group, 50)
        self.assertEqual(target_si, 50)
        self.assertEqual(targets, {"I7HE1": 25, "I7HE2": 25})

    def test_trujillo_caps_five_specialized_routes_at_thirty(self):
        group = route_group([
            ("43H08", 2, 1, 1, 2),
            ("43HE0", 45, 32, 0, 10),
            ("43HE1", 45, 32, 0, 10),
            ("43HE2", 45, 31, 0, 10),
            ("43HE3", 45, 34, 0, 10),
            ("43HE4", 45, 30, 0, 10),
            ("43HN0", 40, 0, 0, 40),
        ])
        targets, target_si = selector_engine._route_specialized_targets(group, 160)
        active = [value for value in targets.values() if value > 0]
        self.assertEqual(target_si, 150)
        self.assertEqual(sum(active), 150)
        self.assertLessEqual(max(active), 30)
        self.assertEqual(sorted(active), [30, 30, 30, 30, 30])

    def test_iquitos_requests_non_specialized_fill_after_specialized_cap(self):
        group = route_group([
            ("JCHE0", 56, 50, 0, 0),
            ("JCH04", 1, 0, 0, 1),
            ("JCH24", 1, 0, 0, 1),
            ("JCH25", 1, 0, 0, 1),
            ("JCH26", 1, 0, 0, 1),
        ])
        targets, target_si = selector_engine._route_specialized_targets(group, 50)
        self.assertEqual(target_si, 30)
        self.assertEqual(targets["JCHE0"], 30)

    def test_optional_tiny_specialized_route_does_not_force_overflow(self):
        group = route_group([
            ("BIG", 50, 0, 0, 0),
            ("TINY", 5, 0, 0, 0),
            ("NO", 20, 0, 0, 20),
        ])
        targets, target_si = selector_engine._route_specialized_targets(group, 35)
        self.assertEqual(target_si, 30)
        self.assertEqual(targets["BIG"], 30)
        self.assertEqual(targets["TINY"], 0)

    def test_quota_below_ten_uses_non_specialized_points(self):
        group = route_group([
            ("BIG", 50, 0, 0, 0),
            ("NO", 20, 0, 0, 20),
        ])
        targets, target_si = selector_engine._route_specialized_targets(group, 5)
        self.assertEqual(target_si, 0)
        self.assertEqual(targets["BIG"], 0)

    def test_specialized_target_reserves_a_substitute(self):
        group = route_group([
            ("SI", 14, 0, 0, 0),
            ("NO", 20, 0, 0, 20),
        ])
        targets, target_si = selector_engine._route_specialized_targets(group, 14)
        self.assertEqual(target_si, 13)
        self.assertEqual(targets["SI"], 13)

    def test_strategy_mix_is_repaired_inside_same_route(self):
        frame = pd.DataFrame([
            {
                "CODIGO": 1, "NOMBRE": "CITY", "Subcanal2": "ON", "RUT.COM": "R1",
                "Ventas": 100, config.SELECTION_COL: "T", "_segment": "GR",
                "_mandatory": False, "_is_specialized": True, "_is_titan": False,
                "_prev_t": False, "_preferred_on_route": True, "Esparta_Flag": 0,
            },
            {
                "CODIGO": 2, "NOMBRE": "CITY", "Subcanal2": "ON", "RUT.COM": "R1",
                "Ventas": 90, config.SELECTION_COL: "T", "_segment": "GR",
                "_mandatory": False, "_is_specialized": True, "_is_titan": False,
                "_prev_t": False, "_preferred_on_route": True, "Esparta_Flag": 0,
            },
            {
                "CODIGO": 3, "NOMBRE": "CITY", "Subcanal2": "ON", "RUT.COM": "R1",
                "Ventas": 80, config.SELECTION_COL: "NO", "_segment": "EG",
                "_mandatory": False, "_is_specialized": True, "_is_titan": False,
                "_prev_t": False, "_preferred_on_route": True, "Esparta_Flag": 0,
            },
        ])
        index = pd.MultiIndex.from_tuples(
            [("CITY", "ON", "EG"), ("CITY", "ON", "GR")],
            names=selector_engine.STRATEGY_GROUP,
        )
        targets = pd.Series([1, 1], index=index, dtype=int)
        selector_engine._rebalance_on_strategy_within_routes(frame, targets)
        selected = frame.loc[frame[config.SELECTION_COL].eq("T"), "_segment"].value_counts()
        self.assertEqual(selected.to_dict(), {"GR": 1, "EG": 1})
        self.assertEqual(int(frame[config.SELECTION_COL].eq("T").sum()), 2)


class OffAndSubstituteTests(unittest.TestCase):
    def test_eg_and_gr_have_priority_inside_the_exact_mix_band_in_both_channels(self):
        for channel in ("ON", "OFF"):
            with self.subTest(channel=channel):
                rows = []
                for route in range(4):
                    for segment in ("EG", "GR", "ME", "PE"):
                        for position in range(8):
                            rows.append({
                                "CODIGO": len(rows) + 1, "NOMBRE": "CITY", "Subcanal2": channel,
                                "CANAL": channel, "RUT.COM": str(route), "ESTRATEGICA": segment,
                                "Fenix": int(channel == "OFF" and segment == "ME" and position == 0),
                                "Programa de Valor": "0", "SEL_MES_ANT": "NO", "Esparta_Flag": 0,
                                "ESPECIALIZADA": "SI" if channel == "ON" else "NO",
                                "Ventas": 100 - position, "LOC.COM": "",
                            })
                frame = pd.DataFrame(rows)
                city = pd.Series({("CITY", channel): 100})
                city.index = city.index.set_names(selector_engine.CITY_GROUP)
                strategy = pd.Series({("CITY", channel, segment): 25 for segment in ("EG", "GR", "ME", "PE")})
                select = selector_engine.select_canal_on if channel == "ON" else selector_engine.select_canal_off
                args = () if channel == "ON" else ({},)
                with patch.object(selector_engine, "_prioritize_eg_gr_titles"):
                    baseline = select(frame, city, strategy, *args)
                result = select(frame, city, strategy, *args)
                titles = result.loc[result[config.SELECTION_COL].eq("T")]
                self.assertEqual(titles["ESTRATEGICA"].value_counts().to_dict(), {
                    "EG": 26, "GR": 26, "ME": 24, "PE": 24,
                })
                by_route = ["RUT.COM", "ESPECIALIZADA", config.SELECTION_COL]
                pd.testing.assert_series_equal(result.groupby(by_route).size(), baseline.groupby(by_route).size())
                self.assertTrue(result.loc[result["Fenix"].gt(0), config.SELECTION_COL].eq("T").all())

    def test_cda_priority_also_respects_the_original_city_mix(self):
        frame = pd.DataFrame([{
            "CODIGO": index, "NOMBRE": "CDA", "CDA_ORIG_CITY": "CITY", "Subcanal2": "ON",
            "RUT.COM": "R1", "_segment": "GR" if index <= 50 else "ME",
            "_is_specialized": True, "_is_titan": False, "_mandatory": False,
            "Esparta_Flag": 0, "_prev_t": False, "Ventas": 100 - index,
            config.SELECTION_COL: "NO" if index == 0 else "T",
        } for index in range(101)])
        cda_strategy = pd.Series({("CDA", "ON", "GR"): 50, ("CDA", "ON", "ME"): 50})
        city_strategy = pd.Series({("CITY", "ON", "GR"): 49, ("CITY", "ON", "ME"): 51})
        selector_engine._prioritize_eg_gr_titles(frame, cda_strategy, city_strategy)
        self.assertEqual(frame.at[0, config.SELECTION_COL], "NO")
        city_strategy = pd.Series({("CITY", "ON", "GR"): 50, ("CITY", "ON", "ME"): 50})
        selector_engine._prioritize_eg_gr_titles(frame, cda_strategy, city_strategy)
        self.assertEqual(frame.at[0, config.SELECTION_COL], "T")
        self.assertEqual(int(frame[config.SELECTION_COL].eq("T").sum()), 100)

    def test_off_route_is_qualified_by_route_even_when_nombre_is_zero(self):
        frame = pd.DataFrame([
            {
                "CODIGO": 1, "NOMBRE": "AREQUIPA", "RUT.COM": "R1", "Subcanal2": "OFF", "CANAL": "OFF",
                "ESTRATEGICA": "GR", "Fenix": 1, "Programa de Valor": "0", "SEL_MES_ANT": "NO",
                "ESPECIALIZADA": "NO",
            },
            {
                "CODIGO": 2, "NOMBRE": "0", "RUT.COM": "R1", "Subcanal2": "OFF", "CANAL": "OFF",
                "ESTRATEGICA": "GR", "Fenix": 0, "Programa de Valor": "0", "SEL_MES_ANT": "NO",
                "ESPECIALIZADA": "NO",
            },
            {
                "CODIGO": 3, "NOMBRE": "AREQUIPA", "RUT.COM": "R2", "Subcanal2": "OFF", "CANAL": "OFF",
                "ESTRATEGICA": "GR", "Fenix": 0, "Programa de Valor": "0", "SEL_MES_ANT": "NO",
                "ESPECIALIZADA": "NO",
            },
        ])
        prepared = selector_engine._prepare(frame)
        self.assertEqual(prepared["_off_qualified_route"].tolist(), [True, True, False])

    def test_substitutes_keep_level_sizes_and_order_by_sales_before_strategy(self):
        frame = pd.DataFrame([
            {
                "CODIGO": code, "NOMBRE": "LIMA", "RUT.COM": "R1", "Ventas": 100 - code,
                config.SELECTION_COL: "T" if code == 1 else "NO", "_is_specialized": True,
                "Esparta_Flag": 0, "_segment": "ME" if code > 9 else "GR",
                "_mandatory": code == 1, "_prev_t": code < 6,
            }
            for code in range(1, 14)
        ])
        selector_engine._assign_substitutes(frame)
        self.assertEqual(frame[config.SELECTION_COL].tolist(), ["T"] + ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4)


if __name__ == "__main__":
    unittest.main()
