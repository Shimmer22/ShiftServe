"""保证后端、阶段和检查点不能混入不完整测量。"""
import copy
import sys
import unittest
import tempfile
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.profiling.pd_sweep import validate, summarize


class StageSweepValidation(unittest.TestCase):
    def setUp(self):
        self.request = dict(prefix_ids=[1]*32, input_ids=[2]*8, generated_ids=[3]*2,
                            decode_points=[1, 2], backend_order=['htp', 'cpu'])
        self.response = dict(mode='stage_sweep_v1', cache=dict(prefix_tokens=32),
                            logit_comparison=dict(passed=True, cosine=1.0, nrmse=0.0), rows=[])
        for backend in ['htp', 'cpu']:
            self.response['rows'].append(dict(backend=backend, H=32, P=8, D=2,
                state_restore_verified=True, finite_logits=True, prefill_ms=1,
                decode_checkpoints=[dict(D=1, decode_ms=2), dict(D=2, decode_ms=4)]))

    def test_complete_and_prefill_only(self):
        validate(self.request, self.response)
        request, response = copy.deepcopy(self.request), copy.deepcopy(self.response)
        request.update(generated_ids=[], decode_points=[])
        for row in response['rows']:
            row.update(D=0, decode_checkpoints=[])
        validate(request, response)

    def test_reject_missing_point_backend_or_wrong_shape(self):
        for mutation in ['missing', 'backend', 'shape', 'numerical', 'time']:
            response = copy.deepcopy(self.response)
            if mutation == 'missing': response['rows'][0]['decode_checkpoints'].pop()
            if mutation == 'backend': response['rows'][0]['backend'] = 'cpu'
            if mutation == 'shape': response['rows'][0]['P'] = 7
            if mutation == 'numerical': response['logit_comparison']['passed'] = False
            if mutation == 'time': response['rows'][0]['prefill_ms'] = float('nan')
            with self.assertRaises(AssertionError):
                validate(self.request, response)

    def test_numerical_difference_reported_without_filtering_timing(self):
        response = copy.deepcopy(self.response)
        response['logit_comparison'].update(passed=False, cosine=0.9888, nrmse=0.15)
        validate(self.request, response)

    def test_prefill_only_summary_keeps_numerically_flagged_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'measurements').mkdir()
            request, response = copy.deepcopy(self.request), copy.deepcopy(self.response)
            request.update(generated_ids=[], decode_points=[])
            response['cache']['source'] = 'memory'
            response['logit_comparison'].update(passed=False, cosine=.988, nrmse=.15)
            for row in response['rows']:
                row.update(D=0, decode_checkpoints=[], restore_ms=1)
            data = dict(request=request, response=response, round=0, stage='prefill')
            (out / 'measurements/r00-prefill-p008.json').write_text(json.dumps(data))
            summarize(out, dict(rounds=1, P=[8], D=[], H=32, scope='test'))
            summary = json.loads((out / 'summary.json').read_text())
            self.assertEqual(summary['stage_rows'], 2)
            self.assertEqual(summary['paired_requests'], 1)


if __name__ == '__main__':
    unittest.main()
