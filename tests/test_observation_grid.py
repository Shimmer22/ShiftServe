import unittest
from a2o.validation.observation import measurement_grid


class ObservationGridTest(unittest.TestCase):
    def config(self):
        return dict(new_tokens=[16, 24, 31, 32, 33, 40, 48, 56, 64, 128, 256, 512, 1024],
                    gap_ms=[0, 50, 200], strategies=['after_return', 'always', 'cost_aware'],
                    routes=['CC', 'HC'], observation_scan=True, system_inputs=[24, 31, 32], scan_gap_ms=200)

    def test_complete_replays_and_no_duplicate_scan(self):
        grid = measurement_grid(self.config())
        self.assertEqual(len(grid), 101)
        self.assertEqual(len(grid), len(set(grid)))
        for p in [24, 31, 32]:
            self.assertEqual(sum(x[0] == p for x in grid), 27)
        for p in [16, 33, 40, 48, 56, 64, 128, 256, 512, 1024]:
            self.assertEqual([x[1:] for x in grid if x[0] == p],
                             [(200, 'after_return', 'CC'), (200, 'always', 'HC')])
        self.assertEqual(2 * (13 * 6 + len(grid) * 3), 762)

    def test_unknown_system_input_rejected(self):
        cfg = self.config(); cfg['system_inputs'] = [99]
        with self.assertRaises(ValueError):
            measurement_grid(cfg)
