import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.scheduling.cost_model import route_times


class CostModel(unittest.TestCase):
    def test_gap_can_change_route_but_cannot_hide_return(self):
        row = {'cpu': {'prefill_ms': 250}, 'htp': {
            'state_export_ms': 20, 'state_clear_ms': 0, 'state_import_ms': 80,
            'tiling_ms': None, 'prefill_ms': 100,
            'return_export_ms': 30, 'return_clear_ms': 0, 'return_import_ms': 40}}
        self.assertEqual(route_times(row, 0)['choice'], 'cpu')
        self.assertEqual(route_times(row, 100)['choice'], 'htp')
        self.assertEqual(route_times(row, 2000)['htp_ms'], 170)


if __name__ == '__main__':
    unittest.main()
