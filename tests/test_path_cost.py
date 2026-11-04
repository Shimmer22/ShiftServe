import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.cost_model import path_cost


class PathCost(unittest.TestCase):
    def test_gap_hides_only_prearrival_state(self):
        row = {'path': 'HC', 'entry_export_ms': 20, 'entry_clear_ms': 0, 'entry_import_ms': 80,
               'prefill_ms': 30, 'switch_export_ms': 10, 'switch_clear_ms': 0,
               'switch_import_ms': 40, 'decode_prefix_ms': [5, 11]}
        self.assertEqual(path_cost(row, 0, 1)['total_ms'], 185)
        self.assertEqual(path_cost(row, 2000, 1)['total_ms'], 85)
        self.assertEqual(path_cost(row, 2000)['total_ms'], 91)
        self.assertEqual(path_cost(row, 0)['end_backend'], 'cpu')
        with self.assertRaises(ValueError):
            path_cost(row, 0, 3)
