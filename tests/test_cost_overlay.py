"""离线候选覆盖 + 实际执行路径的在线修正，无手机。"""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling import CostKey, Sample
from a2o.runtime.self_calibrating import SelfCalibratingRuntime
from a2o.scheduling.offline_cost import OfflineCost, OfflineCostTable
from a2o.scheduling.online_cost import OnlineCostModel

KEY = CostKey('fixture-config', 'hybrid', 'HC', 256, 32, 4, 'prefill')


class OverlayTests(unittest.TestCase):
    def runtime(self, **options):
        baseline = OfflineCostTable([
            OfflineCost(256, 32, 4, 'CC', 48, 0),
            OfflineCost(256, 32, 4, 'HC', 38, 40)], signature=KEY.signature)
        return SelfCalibratingRuntime(offline=baseline, costs=OnlineCostModel(**options))

    def observe_hc(self, runtime, repetitions=3):
        for _ in range(repetitions):
            for phase, duration in [('entry', 40), ('prefill', 40), ('switch', 20), ('decode', 10)]:
                runtime.profiler.record(Sample(replace(KEY, phase=phase), duration))
            runtime.flush()

    def test_offline_initial_choice_without_online_data(self):
        runtime = self.runtime()
        decision = runtime.choose(KEY, 'ready')
        self.assertEqual(decision['backend'], 'htp')
        self.assertEqual(decision['CC']['source'], 'offline')
        self.assertEqual(decision['HC']['source'], 'offline')
        self.assertEqual(decision['HC']['future_ms'], 38)

    def test_only_observed_path_is_corrected(self):
        runtime = self.runtime()
        self.observe_hc(runtime, 2)
        self.assertEqual(runtime.choose(KEY, 'ready')['backend'], 'htp')
        self.observe_hc(runtime, 1)
        decision = runtime.choose(KEY, 'ready')
        self.assertEqual(decision['backend'], 'cpu')
        self.assertEqual(decision['HC']['future_ms'], 70)
        self.assertEqual(decision['HC']['source'], 'online_corrected')
        self.assertEqual(decision['CC']['future_ms'], 48)
        self.assertEqual(decision['CC']['source'], 'offline')
        self.assertEqual(runtime.path_cost(KEY, 100), 70)
        self.assertEqual(runtime.offline.lookup(KEY).future_ms, 38)

    def test_expiration_restores_offline_costs(self):
        now = [0.]
        runtime = self.runtime(ttl_s=1, clock=lambda: now[0])
        self.observe_hc(runtime)
        now[0] = 2.
        decision = runtime.choose(KEY, 'ready')
        self.assertEqual(decision['backend'], 'htp')
        self.assertEqual(decision['HC']['source'], 'offline')

    def test_candidate_readiness_and_coverage_are_preserved(self):
        runtime = self.runtime()
        decision = runtime.choose(KEY, 'pending')
        self.assertEqual(decision['backend'], 'cpu')
        self.assertEqual(decision['reason'], 'candidate_pending')
        self.assertEqual(runtime.choose(replace(KEY, new_tokens=34), 'ready')['reason'], 'coverage_missing')
        self.assertIsNone(runtime.path_prediction(replace(KEY, signature='other')))

    def test_partial_online_stages_do_not_double_count_future(self):
        runtime = self.runtime(min_samples=1)
        runtime.costs.update(Sample(KEY, 999))
        prediction = runtime.path_prediction(KEY)
        self.assertEqual(prediction['future_ms'], 38)
        self.assertEqual(prediction['future_source'], 'offline')

    def test_existing_router_table_format_and_explicit_fixture(self):
        path = ROOT / 'scenarios/self-calibrating-router.tsv'
        with self.assertRaises(ValueError):
            OfflineCostTable.load(path, signature=KEY.signature)
        table = OfflineCostTable.load(path, signature=KEY.signature, allow_fixture=True)
        self.assertEqual(table.lookup(KEY).future_ms, 38)
        self.assertEqual((table.context, table.batch, table.queue, table.opbatch), (2048, 128, 1, 1))


if __name__ == '__main__':
    unittest.main()
