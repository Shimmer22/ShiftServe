"""有限等待的经济准入、预算边界与未完成future不可选HTP。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling.pending_wait import trigger_rates


class PendingWaitTest(unittest.TestCase):
    def test_trigger_denominator_excludes_calibration_and_forced_paths(self):
        common = dict(H=4096, P=32, gap_requested_ms=0, strategy='cost_aware',
                      prepared_early=True, ready_at_return=False, route='CC')
        rows = [dict(common, phase='calibration', requested_route='router'),
                dict(common, phase='measurement', requested_route='HC'),
                dict(common, phase='measurement', requested_route='router')]
        rate = trigger_rates(rows)['by_strategy'][0]
        self.assertEqual(rate['router_decisions'], 1)
        self.assertEqual(rate['prepare'], dict(numerator=1, denominator=1, rate=1.0))
        self.assertIsNone(rate['wait_success']['rate'])

    def test_wait_success_and_timeout_use_attempts_as_denominator(self):
        common = dict(H=4096, P=32, gap_requested_ms=0, strategy='cost_aware',
                      prepared_early=True, ready_at_return=False, phase='measurement',
                      requested_route='router', wait_policy='bounded')
        rows = [dict(common, route='HC', pending_wait_attempted=True, pending_wait_succeeded=True),
                dict(common, route='CC', pending_wait_attempted=True, pending_wait_timed_out=True),
                dict(common, route='CC')]
        rate = trigger_rates(rows)['by_strategy'][0]
        self.assertEqual(rate['wait_attempt']['denominator'], 3)
        self.assertEqual(rate['wait_success'], dict(numerator=1, denominator=2, rate=.5))
        self.assertEqual(rate['wait_timeout'], dict(numerator=1, denominator=2, rate=.5))

    def test_native_wait_guards_and_timeout(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as d:
            binary = Path(d) / 'pending-wait-test'
            subprocess.run([compiler, '-std=c++17', '-pthread', '-I' + str(ROOT / 'src/a2o/native'),
                            str(ROOT / 'tests/pending_wait_test.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == '__main__':
    unittest.main()
