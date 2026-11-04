import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runner import check_async_preparation, validate
from a2o.router_table import export_table


class AsyncPreparationTest(unittest.TestCase):
    def test_native_state_machine(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler, 'C++ compiler required for runtime mechanism validation')
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / 'async-test')
            subprocess.run([compiler, '-std=c++17', '-pthread', '-I' + str(ROOT / 'src/a2o'),
                            str(ROOT / 'tests/async_preparation_test.cpp'), '-o', binary], check=True)
            subprocess.run([binary], check=True, timeout=15)

    def test_explicit_configuration(self):
        config = json.loads((ROOT / 'configs/async-mechanism-smoke.json').read_text())
        validate(config)
        for change in [{'compare_prefill': True}, {'decode_routes': ['htp'] * 4},
                       {'preparation_cases': ['early'] * 4}, {'pre_move': True}]:
            with self.assertRaises(ValueError):
                validate({**config, **change})

    def test_cross_round_lane_and_cost_router(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as tmp:
            binary = str(Path(tmp) / 'lane-test')
            subprocess.run([compiler, '-std=c++17', '-pthread', '-I' + str(ROOT / 'src/a2o'),
                            str(ROOT / 'tests/lane_router_test.cpp'), '-o', binary], check=True)
            table = Path(tmp) / 'measured.tsv'
            export_table(ROOT / 'results/20261103-213658-342299-four-path-structured-grid', table)
            subprocess.run([binary, str(table)], check=True, timeout=15)

    def test_event_contract_rejects_join_before_response_and_missing_checks(self):
        names = ['tool_started', 'observation_arrived', 'bind_pending', 'cpu_prefill_started',
                 'prepare_started', 'prepare_ready', 'action_complete', 'cpu_state_committed',
                 'bind_stale', 'drain_started_after_response', 'drain_finished']
        result = {'timing_valid_for_performance': False, 'turns': [{
            'prefill_backend': 'cpu', 'decode_backend': 'cpu', 'async_preparation': {
                'case': 'early', 'reason': 'pending', 'snapshot_version': 0, 'committed_version': 1,
                'authority_checked': True, 'stale_rejected': True, 'status': 'ready',
                'events': [{'event': n, 'ms': 0} for n in names]}}]}
        self.assertTrue(check_async_preparation(result, ['early'])['passed'])
        for field in ['authority_checked', 'stale_rejected']:
            bad = copy.deepcopy(result)
            bad['turns'][0]['async_preparation'][field] = False
            self.assertFalse(check_async_preparation(bad, ['early'])['passed'])
        bad = copy.deepcopy(result)
        events = bad['turns'][0]['async_preparation']['events']
        events.insert(0, events.pop(-2))
        self.assertFalse(check_async_preparation(bad, ['early'])['passed'])
        self.assertFalse(check_async_preparation(result, ['early', 'cancel'])['passed'])


if __name__ == '__main__':
    unittest.main()
