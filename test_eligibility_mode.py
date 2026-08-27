"""Tests for FCD-only vs ML training mode in run_eligibility."""

import ast
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from exceptions import ForestryCarbonError
from eligibility_mode import resolve_eligibility_mode


BASE_CONFIG = {
    'AOI_path': './aoi.shp',
    'OID_field_name': 'id',
    'date_start_end': ['2024-1-1', '2024-5-31'],
}


class TestResolveEligibilityMode(unittest.TestCase):
    def test_fcd_mode_does_not_require_training(self):
        result = resolve_eligibility_mode(True, BASE_CONFIG)
        self.assertEqual(result['mode'], 'fcd')
        self.assertFalse(result['use_training'])
        self.assertNotIn('input_training', result['required_keys'])
        self.assertEqual(
            result['required_keys'],
            ['AOI_path', 'OID_field_name', 'date_start_end'],
        )

    def test_fcd_mode_ignores_provided_training(self):
        config = {**BASE_CONFIG, 'input_training': './points.shp'}
        result = resolve_eligibility_mode(True, config)
        self.assertEqual(result['mode'], 'fcd')
        self.assertFalse(result['use_training'])

    def test_ml_mode_requires_training(self):
        config = {**BASE_CONFIG, 'input_training': './points.shp'}
        result = resolve_eligibility_mode(False, config)
        self.assertEqual(result['mode'], 'ml')
        self.assertTrue(result['use_training'])
        self.assertIn('input_training', result['required_keys'])

    def test_ml_mode_without_training_raises(self):
        with self.assertRaises(ForestryCarbonError) as ctx:
            resolve_eligibility_mode(False, BASE_CONFIG)
        self.assertIn('input_training', str(ctx.exception))
        self.assertIn('is_only_fcd=False', str(ctx.exception))

    def test_ml_mode_empty_training_raises(self):
        config = {**BASE_CONFIG, 'input_training': '   '}
        with self.assertRaises(ForestryCarbonError):
            resolve_eligibility_mode(False, config)


class TestRunEligibilitySignature(unittest.TestCase):
    def test_is_only_fcd_defaults_true(self):
        source = (Path(__file__).parent / 'core' / 'main.py').read_text()
        tree = ast.parse(source)
        fn = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == 'ForestryCarbonARR':
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == 'run_eligibility':
                        fn = item
                        break
        self.assertIsNotNone(fn, 'run_eligibility not found')
        arg_names = [arg.arg for arg in fn.args.args]
        self.assertIn('is_only_fcd', arg_names)
        self.assertIn('use_gee', arg_names)
        # defaults map to the last N positional-or-keyword args
        defaults = fn.args.defaults
        default_map = dict(zip(arg_names[-len(defaults):], defaults))
        self.assertIn('is_only_fcd', default_map)
        self.assertIsInstance(default_map['is_only_fcd'], ast.Constant)
        self.assertIs(default_map['is_only_fcd'].value, True)


if __name__ == '__main__':
    unittest.main()
