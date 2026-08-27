"""Tests for Hansen GFC asset resolution (argument > config > init > default)."""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from hansen_gfc import (
    DEFAULT_HANSEN_GFC_ASSET,
    hansen_lossyear_max,
    resolve_hansen_gfc_asset,
)


class TestResolveHansenGfcAsset(unittest.TestCase):
    def test_default_when_nothing_provided(self):
        self.assertEqual(
            resolve_hansen_gfc_asset(),
            DEFAULT_HANSEN_GFC_ASSET,
        )
        self.assertIn('global_forest_change_', DEFAULT_HANSEN_GFC_ASSET)

    def test_argument_wins_over_config_and_instance(self):
        result = resolve_hansen_gfc_asset(
            argument='UMD/hansen/global_forest_change_2024_v1_12',
            config={'hansen_gfc_asset': 'UMD/hansen/global_forest_change_2023_v1_11'},
            instance_config={'hansen_gfc_asset': 'UMD/hansen/global_forest_change_2022_v1_10'},
        )
        self.assertEqual(result, 'UMD/hansen/global_forest_change_2024_v1_12')

    def test_flat_config_used_when_no_argument(self):
        result = resolve_hansen_gfc_asset(
            config={'hansen_gfc_asset': 'UMD/hansen/global_forest_change_2023_v1_11'},
            instance_config={'hansen_gfc_asset': 'UMD/hansen/global_forest_change_2022_v1_10'},
        )
        self.assertEqual(result, 'UMD/hansen/global_forest_change_2023_v1_11')

    def test_nested_hansen_gfc_asset(self):
        result = resolve_hansen_gfc_asset(
            config={'hansen': {'gfc_asset': 'UMD/hansen/global_forest_change_2024_v1_12'}},
        )
        self.assertEqual(result, 'UMD/hansen/global_forest_change_2024_v1_12')

    def test_instance_config_used_when_run_config_omits_it(self):
        result = resolve_hansen_gfc_asset(
            config={'AOI_path': './aoi.shp'},
            instance_config={'hansen_gfc_asset': 'UMD/hansen/global_forest_change_2023_v1_11'},
        )
        self.assertEqual(result, 'UMD/hansen/global_forest_change_2023_v1_11')

    def test_blank_argument_falls_through(self):
        result = resolve_hansen_gfc_asset(
            argument='   ',
            config={'hansen_gfc_asset': 'UMD/hansen/global_forest_change_2023_v1_11'},
        )
        self.assertEqual(result, 'UMD/hansen/global_forest_change_2023_v1_11')


class TestHansenLossyearMax(unittest.TestCase):
    def test_parses_year_from_asset_id(self):
        self.assertEqual(
            hansen_lossyear_max('UMD/hansen/global_forest_change_2025_v1_13'),
            25,
        )
        self.assertEqual(
            hansen_lossyear_max('UMD/hansen/global_forest_change_2023_v1_11'),
            23,
        )

    def test_unknown_asset_uses_default_max(self):
        self.assertEqual(hansen_lossyear_max('not-a-hansen-id'), 25)


class TestRunEligibilityHansenSignature(unittest.TestCase):
    def test_run_eligibility_accepts_hansen_gfc_asset(self):
        source = (Path(__file__).parent / 'core' / 'main.py').read_text()
        tree = ast.parse(source)
        fn = None
        init_fn = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == 'ForestryCarbonARR':
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'run_eligibility':
                        fn = item
                    if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        init_fn = item
        self.assertIsNotNone(fn)
        self.assertIn('hansen_gfc_asset', [arg.arg for arg in fn.args.args])
        self.assertIsNotNone(init_fn)
        self.assertIn('hansen_gfc_asset', [arg.arg for arg in init_fn.args.args])


class TestHistoricalLossLoadsFromConfig(unittest.TestCase):
    def test_initiate_tcl_uses_instance_gfc_asset(self):
        source = (
            Path(__file__).parent.parent
            / 'GEE_notebook_Forestry'
            / 'osi'
            / 'hansen'
            / 'historical_loss.py'
        ).read_text()
        self.assertIn('ee.Image(self.gfc_asset)', source)
        self.assertNotIn(
            'ee.Image("UMD/hansen/global_forest_change_2025_v1_13")',
            source,
        )
        self.assertIn('def resolve_gfc_asset', source)


if __name__ == '__main__':
    unittest.main()
