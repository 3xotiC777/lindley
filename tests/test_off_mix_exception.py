import unittest

import pandas as pd

import config
import selector_engine as engine


def make_tumbes():
    rows = []
    def add(route, segment, count, titan_first=False):
        for i in range(count):
            rows.append({
                'CODIGO': len(rows) + 1, 'NOMBRE': 'TUMBES', 'Subcanal2': 'OFF',
                'CANAL': 'OFF', 'RUT.COM': route, 'ESTRATEGICA': segment,
                'Fenix': 0, 'Programa de Valor': 'TITAN' if titan_first and i == 0 else '0',
                'SEL_MES_ANT': 'NO', 'ESPECIALIZADA': 'NO', 'LOC.COM': '',
                'Esparta_Flag': 0, 'Ventas': float(1000 - len(rows)),
            })
    add('QUALIFIED1', 'EG', 3)
    add('QUALIFIED1', 'GR', 2, titan_first=True)
    add('QUALIFIED1', 'ME', 5)
    add('QUALIFIED1', 'PE', 14)
    add('QUALIFIED2', 'GR', 2)
    add('QUALIFIED2', 'ME', 1)
    add('QUALIFIED2', 'PE', 4, titan_first=True)
    add('UNQUALIFIED', 'EG', 17)
    add('UNQUALIFIED', 'GR', 25)
    return pd.DataFrame(rows)


def history(eg=27, gr=37):
    index = pd.MultiIndex.from_tuples([('TUMBES','OFF','EG'),('TUMBES','OFF','GR')],names=engine.STRATEGY_GROUP)
    return pd.Series([eg, gr], index=index)


class OffMixExceptionTests(unittest.TestCase):
    def test_tumbes_exception_improves_mix_without_losing_titles_or_mandatory(self):
        data = make_tumbes()
        idx = pd.MultiIndex.from_tuples([('TUMBES','OFF')],names=engine.CITY_GROUP)
        result = engine.select_canal_off(data, pd.Series([31],index=idx), history(), {})
        selected = result[result[config.SELECTION_COL].eq('T')]
        counts = selected['ESTRATEGICA'].value_counts().to_dict()
        self.assertEqual(len(selected),31)
        self.assertEqual((counts['EG'],counts['GR'],counts['PE']),(13,17,1))
        self.assertEqual(int(selected['Programa de Valor'].eq('TITAN').sum()),2)
        self.assertTrue(selected['RUT.COM'].eq('UNQUALIFIED').any())
        self.assertTrue(result.loc[result['RUT.COM'].eq('UNQUALIFIED'),config.SELECTION_COL].isin(['S1','S2','S3']).any())

    def test_no_exception_when_mix_within_five_points(self):
        data = make_tumbes()
        prepared = engine._prepare(data)
        current = prepared['RUT.COM'].ne('UNQUALIFIED') & prepared['_segment'].isin(['EG','GR'])
        prepared.loc[current, config.SELECTION_COL] = 'T'
        engine._improve_off_mix_with_route_exception(prepared, history(3,4), {}, set())
        self.assertFalse(prepared.loc[prepared[config.SELECTION_COL].eq('T'),'RUT.COM'].eq('UNQUALIFIED').any())

    def test_exactly_five_pp_does_not_activate_exception(self):
        prepared = engine._prepare(make_tumbes())
        selected_ids = prepared.index[prepared['RUT.COM'].ne('UNQUALIFIED')][:20]
        prepared.loc[selected_ids, config.SELECTION_COL] = 'T'
        prepared.loc[selected_ids[:19], '_segment'] = 'GR'
        prepared.loc[selected_ids[19:], '_segment'] = 'EG'
        engine._improve_off_mix_with_route_exception(prepared, history(0, 20), {}, set())
        self.assertFalse(prepared.loc[prepared[config.SELECTION_COL].eq('T'),'RUT.COM'].eq('UNQUALIFIED').any())

    def test_cda_keeps_qualified_routes_when_mix_is_exactly_five_pp(self):
        template = make_tumbes().iloc[0].to_dict()
        rows = []
        for i in range(30):
            row = dict(template, CODIGO=i+1, RUT_COM='unused')
            row['RUT.COM'] = 'QUALIFIED' if i < 20 else 'UNQUALIFIED'
            row['ESTRATEGICA'] = 'PE' if i == 19 else 'GR'
            row['Programa de Valor'] = 'TITAN' if i == 0 else '0'
            rows.append(row)
        idx = pd.MultiIndex.from_tuples([('TUMBES','OFF')],names=engine.CITY_GROUP)
        result = engine.select_canal_off(pd.DataFrame(rows), pd.Series([20],index=idx), history(0,20), {}, allow_off_route_fallback=True)
        self.assertFalse(result.loc[result[config.SELECTION_COL].eq('T'),'RUT.COM'].eq('UNQUALIFIED').any())

    def test_additional_strategy_is_counted_in_historical_mix(self):
        frame = engine._prepare(make_tumbes())
        selected_ids = frame.index[frame['RUT.COM'].ne('UNQUALIFIED')][:20]
        frame.loc[selected_ids, config.SELECTION_COL] = 'T'
        frame.loc[selected_ids[:10], '_segment'] = 'GR'
        frame.loc[selected_ids[10:], '_segment'] = 'EG'
        candidates = frame.index[frame['RUT.COM'].eq('UNQUALIFIED')][:3]
        frame.loc[candidates, '_segment'] = 'MI'
        idx = pd.MultiIndex.from_tuples([('TUMBES','OFF','EG'),('TUMBES','OFF','GR'),('TUMBES','OFF','MI')],names=engine.STRATEGY_GROUP)
        engine._improve_off_mix_with_route_exception(frame, pd.Series([9,9,2],index=idx), {}, set())
        selected = frame.loc[frame[config.SELECTION_COL].eq('T'),'_segment'].value_counts().to_dict()
        self.assertEqual(selected.get('MI'),2)


if __name__ == '__main__': unittest.main()
