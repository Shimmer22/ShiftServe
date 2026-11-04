"""Oracle 只能选择合格、实际可用的完整重放条件。"""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('oracle_analysis', ROOT / 'scripts/analyze/oracle.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class OracleAnalysisTest(unittest.TestCase):
    def fixture(self, directory, ready=False):
        config = {'new_tokens': [16, 64], 'gap_ms': [0, 50, 200, 500],
                  'strategies': ['after_return', 'always', 'cost_aware'], 'routes': ['CC', 'HC'], 'repetitions': 3}
        (directory / 'manifest.json').write_text(json.dumps({'config': config}))
        rows = []
        for p in config['new_tokens']:
            for rep in range(3):
                for route in ('CC', 'HC'):
                    rows.append({'P': p, 'route': route, 'repetition': rep, 'numerical_passed': True, 'boundary_min_cosine': 1.0, 'boundary_max_nrmse': 0.0, 'phase': 'calibration', 'passed': True, 'response_check': {'passed': True},
                                 'generated_ids': [1, 2], 'D': 2, 'output': '{"action":"wait"}'})
        for p in config['new_tokens']:
            for g in config['gap_ms']:
                for s in config['strategies']:
                    for route in ('CC', 'HC', 'router'):
                        for rep in range(3):
                            actual = 'CC' if route == 'router' else route
                            response = 100 if actual == 'CC' else 10
                            rows.append({'id': len(rows), 'phase': 'measurement', 'P': p, 'gap_requested_ms': g,
                                'strategy': s, 'requested_route': route, 'route': actual, 'repetition': rep,
                                'ready_at_return': ready and s != 'after_return', 'response_after_return_ms': response,
                                'tool_to_response_ms': g + response, 'actual_gap_ms': g, 'prepare_ms': 20,
                                'drain_after_response_ms': 0, 'passed': True, 'response_check': {'passed': True},
                                'generated_ids': [1, 2], 'D': 2, 'output': '{"action":"wait"}'})
        self.save(directory, rows)
        return rows

    def save(self, directory, rows):
        (directory / 'checked-rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))

    def test_pending_fast_hc_is_excluded_from_strict_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); self.fixture(d)
            result = module.summarize(d)
            for c in result['conditions']:
                self.assertEqual(c['joint_oracle']['route'], 'CC')
                self.assertEqual(c['joint_oracle_allow_wait']['route'], 'HC')
                self.assertEqual(c['policies'][2]['routing_oracle_after_return_ms'], 100)

    def test_ready_fast_hc_is_admissible(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); self.fixture(d, ready=True)
            result = module.summarize(d)
            self.assertEqual(result['conditions'][0]['joint_oracle']['route'], 'HC')
            self.assertEqual(result['conditions'][0]['policies'][2]['routing_oracle_after_return_ms'], 10)

    def test_one_bad_hc_calibration_excludes_whole_condition(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); rows = self.fixture(d, ready=True)
            bad = next(r for r in rows if r['phase'] == 'calibration' and r['P'] == 16 and r['route'] == 'HC')
            bad.update(numerical_passed=False, boundary_min_cosine=0.998)
            self.save(d, rows)
            result = module.summarize(d)
            self.assertFalse(result['HC_numerically_qualified'][16])
            self.assertFalse(result['oracle_gain_identifiable'])
            self.assertIsNone(result['conditions'][0]['router_vs_joint_oracle_ms'])
            self.assertTrue(result['HC_numerically_qualified'][64])
            self.assertEqual(result['conditions'][0]['joint_oracle_allow_wait']['route'], 'CC')
            self.assertEqual(result['conditions'][4]['joint_oracle_allow_wait']['route'], 'HC')

    def test_exploration_99_accepts_observed_hc_without_dropping_error_metric(self):
        observed = {'boundary_min_cosine': 0.998556, 'boundary_max_nrmse': 0.053982}
        self.assertTrue(module.numeric_accepted(observed, {'min_cosine': 0.99, 'max_nrmse': None}))
        self.assertFalse(module.numeric_accepted(observed, {'min_cosine': 0.999, 'max_nrmse': 0.05}))
        self.assertFalse(module.numeric_accepted({**observed, 'boundary_min_cosine': 0.989}, {'min_cosine': 0.99, 'max_nrmse': None}))

    def test_generalized_history_and_input_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); rows = self.fixture(d, ready=True)
            cfg = json.loads((d / 'manifest.json').read_text())
            cfg['config'].update(history_tokens=2048, new_tokens=[24], gap_ms=[0, 200])
            (d / 'manifest.json').write_text(json.dumps(cfg))
            rows = [r for r in rows if r['P'] == 16 and (r['phase'] == 'calibration' or r['gap_requested_ms'] in [0, 200])]
            for r in rows: r.update(P=24, H=2048)
            self.save(d, rows)
            result = module.summarize(d)
            self.assertEqual(result['H'], 2048)
            self.assertEqual(result['trials'], 60)
            self.assertEqual(len(result['conditions']), 2)

    def test_bad_hc_output_remains_diagnostic_and_never_enters_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); rows = self.fixture(d, ready=True)
            for r in rows:
                if r['P'] == 16 and r['route'] == 'HC':
                    r.update(passed=False, response_check={'passed': False}, output='{"action":"cWait"}', generated_ids=[1, 3, 2], D=3)
                    if r['phase'] == 'calibration': r.update(numerical_passed=False, boundary_min_cosine=0.98)
            self.save(d, rows)
            result = module.summarize(d)
            self.assertFalse(result['HC_numerically_qualified'][16])
            self.assertTrue(result['HC_numerically_qualified'][64])
            self.assertGreater(result['diagnostic_failures'], 0)
            self.assertTrue(result['accepted_response_and_state_passed'])
            self.assertEqual(result['conditions'][0]['joint_oracle_allow_wait']['route'], 'CC')

    def test_chosen_trajectory_is_included_and_unmatched_state_is_not_a_fake_oracle(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); rows = self.fixture(d, ready=True)
            for r in rows:
                if r['phase'] == 'measurement' and r['strategy'] == 'always' and r['requested_route'] == 'router':
                    r.update(route='HC', response_after_return_ms=9, tool_to_response_ms=r['gap_requested_ms']+9)
            self.save(d, rows)
            result = module.summarize(d)
            policy = result['conditions'][0]['policies'][1]
            self.assertEqual(policy['routing_oracle_after_return_ms'], 9)
            self.assertTrue(all(r['source_is_actual_router'] for r in policy['routing_oracle_replays']))
            next(r for r in rows if r['phase'] == 'measurement' and r['P'] == 16 and r['gap_requested_ms'] == 0 and r['strategy'] == 'always' and r['requested_route'] == 'CC')['ready_at_return'] = False
            self.save(d, rows)
            result = module.summarize(d)
            self.assertIsNone(result['conditions'][0]['policies'][1]['routing_oracle_after_return_ms'])

    def test_actual_publication_must_precede_final_trial_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); rows = self.fixture(d); events = []
            for i, r in enumerate(rows):
                t = i * 10
                r.update(id=i, http_issued_ms=t, http_returned_ms=t+1,
                         response_compute_completed_ms=t+2, response_completed_ms=t+4)
                events += [{'event': 'response_published', 'turn': i, 'monotonic_ms': t+3,
                            'output': r['output'], 'eog': True},
                           {'event': 'trial_completed', 'turn': i, 'monotonic_ms': t+5}]
            self.save(d, rows)
            event_file = d / 'native-events.jsonl'
            event_file.write_text(''.join(json.dumps(e) + '\n' for e in events))
            self.assertTrue(module.summarize(d)['response_publication_lifecycle_checked'])
            events[0]['monotonic_ms'] = 100
            event_file.write_text(''.join(json.dumps(e) + '\n' for e in events))
            with self.assertRaises(ValueError): module.summarize(d)

    def test_failed_or_incomplete_data_cannot_be_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp); rows = self.fixture(d); rows[-1]['passed'] = False; self.save(d, rows)
            with self.assertRaises(ValueError): module.summarize(d)
            self.save(d, rows[:-1])
            with self.assertRaises(ValueError): module.summarize(d)


if __name__ == '__main__': unittest.main()
