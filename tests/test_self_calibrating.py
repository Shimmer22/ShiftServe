"""无手机测试：虚拟时钟覆盖预算、抽样、成本漂移与有界状态。"""
from dataclasses import replace
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.profiling import CostKey, Sample, SamplingProfiler
from a2o.runtime.self_calibrating import SelfCalibratingRuntime
from a2o.scheduling.online_cost import OnlineCostModel
from a2o.warmup import RequestWarmup, WarmupConfig, WarmupController


class Clock:
    def __init__(self):
        self.ns = 0
        self.reads = 0

    def tick(self):
        self.reads += 1
        return self.ns

    def advance(self, ms):
        self.ns += round(ms * 1e6)

    def seconds(self):
        return self.ns / 1e9


KEY = CostKey('fixture-config', 'hybrid', 'HC', 256, 32, 4, 'prefill')


class WarmupTests(unittest.TestCase):
    def test_cold_start_then_stable(self):
        clock = Clock()
        steps = iter([9, 5, 3, 3, 3])
        result = WarmupController(WarmupConfig(enabled=True), clock_ns=clock.tick).run(lambda: clock.advance(next(steps)))
        self.assertEqual((result.reason, result.steps, result.elapsed_ms), ('stable', 5, 23))

    def test_budget_predicts_next_step_and_records_nonpreemptible_overrun(self):
        clock = Clock()
        controller = WarmupController(WarmupConfig(enabled=True, budget_ms=10), clock_ns=clock.tick)
        result = controller.run(lambda: clock.advance(6))
        self.assertEqual((result.reason, result.steps), ('budget', 1))
        result = controller.run(lambda: clock.advance(12))
        self.assertEqual(result.overrun_ms, 2)

    def test_disabled_zero_budget_and_failure(self):
        def fail():
            raise RuntimeError('backend unavailable')
        self.assertEqual(WarmupController(WarmupConfig(enabled=False)).run(fail).steps, 0)
        self.assertEqual(WarmupController(WarmupConfig(enabled=True, budget_ms=0)).run(fail).steps, 0)
        self.assertEqual(WarmupController().run(fail).reason, 'disabled')
        with self.assertRaisesRegex(RuntimeError, 'backend unavailable'):
            WarmupController(WarmupConfig(enabled=True)).run(fail)

    def test_unstable_stops_at_limit(self):
        clock = Clock()
        steps = iter([1, 3, 1, 3])
        result = WarmupController(WarmupConfig(enabled=True, max_steps=4), clock_ns=clock.tick).run(
            lambda: clock.advance(next(steps)))
        self.assertEqual(result.reason, 'max_steps')


class ProfilingTests(unittest.TestCase):
    def test_cold_samples_are_consecutive_then_sparse(self):
        clock = Clock()
        profiler = SamplingProfiler(every=8, cold_samples=3, clock_ns=clock.tick)
        sampled = []
        for request in range(1, 12):
            with profiler.span(KEY):
                clock.advance(2)
            if profiler.drain():
                sampled.append(request)
        self.assertEqual(sampled, [1, 2, 3, 11])
        self.assertEqual(clock.reads, 8)

    def test_failed_calls_do_not_complete_cold_collection(self):
        profiler = SamplingProfiler(every=16)
        for _ in range(3):
            with self.assertRaises(RuntimeError):
                with profiler.span(KEY):
                    raise RuntimeError('failed')
        profiler.drain()
        for _ in range(3):
            with profiler.span(KEY):
                pass
        self.assertEqual(len(profiler.drain()), 3)

    def test_disabled_and_unsampled_read_no_clock(self):
        clock = Clock()
        off = SamplingProfiler(enabled=False, clock_ns=clock.tick)
        with off.span(KEY):
            pass
        self.assertEqual(clock.reads, 0)
        profiler = SamplingProfiler(every=4, cold_samples=0, clock_ns=clock.tick)
        for _ in range(5):
            with profiler.span(KEY):
                clock.advance(2)
        self.assertEqual(clock.reads, 4)
        self.assertEqual([s.duration_ms for s in profiler.drain()], [2, 2])

    def test_phase_sampling_does_not_alias(self):
        profiler = SamplingProfiler(every=4, cold_samples=0)
        for _ in range(5):
            for phase in ('entry', 'prefill', 'switch', 'decode'):
                with profiler.span(replace(KEY, phase=phase)):
                    pass
        self.assertEqual(len(profiler.drain()), 8)

    def test_bounded_queue_counters_and_failed_sample(self):
        profiler = SamplingProfiler(every=1, capacity=2)
        for h in range(4):
            with profiler.span(replace(KEY, history_tokens=h)):
                pass
        self.assertEqual(profiler.dropped, 2)
        self.assertEqual(len(profiler._seen), 2)
        with self.assertRaises(RuntimeError):
            with profiler.span(KEY):
                raise RuntimeError('failed')
        self.assertTrue(profiler.drain()[-1].failed)
        self.assertEqual(profiler.drain(), [])


class OnlineCostTests(unittest.TestCase):
    def test_natural_request_warmup_across_distinct_shapes(self):
        clock = Clock()
        runtime = SelfCalibratingRuntime(
            profiler=SamplingProfiler(clock_ns=clock.tick),
            costs=OnlineCostModel(clock=clock.seconds))
        self.assertEqual(runtime.state(KEY), 'cold')
        executions = []
        for i, p in enumerate((32, 34, 36)):
            key = replace(KEY, history_tokens=256 + 16 * i, new_tokens=p, decode_tokens=4 + i)
            with runtime.span(key):
                executions.append(key)
                clock.advance(p * 2)
            runtime.flush()
            self.assertEqual(runtime.state(key), 'ready' if i == 2 else 'collecting')
        self.assertEqual(len(executions), 3)  # 只有三次用户请求，无额外推理。
        prediction = runtime.costs.prediction(replace(KEY, new_tokens=38))
        self.assertEqual(prediction['source'], 'shape_bucket')
        self.assertEqual(prediction['duration_ms'], 76)
        self.assertEqual(runtime.costs.sample_count(KEY), 3)

    def test_bucket_boundaries_and_phase_specific_dimensions(self):
        model = OnlineCostModel(min_samples=1)
        model.update(Sample(replace(KEY, new_tokens=31), 31))
        self.assertIsNone(model.predict(KEY))  # P31/P32 不合桶。
        model.update(Sample(KEY, 32))
        self.assertEqual(model.predict(replace(KEY, decode_tokens=99)), 32)
        self.assertIsNone(model.predict(replace(KEY, signature='other-runtime')))
        self.assertIsNone(model.predict(replace(KEY, path='CC')))
        self.assertIsNone(model.predict(replace(KEY, mode='htp', path='HH')))
        self.assertIsNone(model.predict(replace(KEY, history_tokens=384)))
        entry = replace(KEY, phase='entry')
        model.update(Sample(entry, 40))
        self.assertEqual(model.predict(replace(entry, new_tokens=100, decode_tokens=100)), 40)
        self.assertIsNone(model.predict(replace(entry, history_tokens=0)))

    def test_expired_bucket_returns_to_dense_sampling(self):
        clock = Clock()
        runtime = SelfCalibratingRuntime(
            profiler=SamplingProfiler(every=100, clock_ns=clock.tick),
            costs=OnlineCostModel(ttl_s=1, clock=clock.seconds))
        for _ in range(3):
            with runtime.span(KEY):
                clock.advance(1)
            runtime.flush()
        self.assertEqual(runtime.state(KEY), 'ready')
        clock.advance(1001)
        self.assertEqual(runtime.state(KEY), 'cold')
        for _ in range(3):
            with runtime.span(KEY):
                clock.advance(1)
            self.assertEqual(runtime.flush(), 1)
        self.assertEqual(runtime.state(KEY), 'ready')

    def test_default_runtime_never_invokes_initialization(self):
        runtime = SelfCalibratingRuntime()
        self.assertIsInstance(runtime.warmup, RequestWarmup)
        self.assertFalse(hasattr(runtime.warmup, 'run'))

    def test_ema_adapts_and_ignores_warmup_and_failure(self):
        model = OnlineCostModel(alpha=0.5, min_samples=2)
        for sample in [Sample(KEY, 900, warmup=True), Sample(KEY, 900, failed=True),
                       Sample(KEY, float('nan')), Sample(KEY, -1)]:
            self.assertFalse(model.update(sample))
        model.update(Sample(KEY, 10))
        self.assertIsNone(model.predict(KEY))
        model.update(Sample(KEY, 20))
        self.assertEqual(model.predict(KEY), 15)
        model.update(Sample(KEY, 25))
        self.assertEqual(model.predict(KEY), 20)

    def test_exact_keys_stale_reset_and_eviction(self):
        clock = Clock()
        model = OnlineCostModel(min_samples=2, max_keys=2, ttl_s=1, shape_reuse=False, clock=clock.seconds)
        for _ in range(2):
            model.update(Sample(KEY, 10))
        for key in [replace(KEY, new_tokens=64), replace(KEY, signature='other'),
                    replace(KEY, path='CC'), replace(KEY, decode_tokens=8)]:
            self.assertIsNone(model.predict(key))
        clock.advance(1001)
        self.assertIsNone(model.predict(KEY))
        model.update(Sample(KEY, 20))
        self.assertIsNone(model.predict(KEY))
        model.update(Sample(replace(KEY, new_tokens=64), 1))
        model.update(Sample(replace(KEY, new_tokens=128), 1))
        self.assertEqual(len(model.snapshot()), 2)
        self.assertIsNone(model.predict(KEY))

    def test_gap_hides_entry_only_and_requires_all_phases(self):
        runtime = SelfCalibratingRuntime(costs=OnlineCostModel(min_samples=1))
        self.assertIsNone(runtime.path_cost(KEY))
        for phase, ms in [('entry', 40), ('prefill', 20), ('switch', 10), ('decode', 8),
                          ('model_init', 100), ('vision_init', 30)]:
            runtime.profiler.record(Sample(replace(KEY, phase=phase), ms))
        self.assertEqual(runtime.flush(), 6)
        self.assertEqual(runtime.path_cost(KEY, 0), 78)
        self.assertEqual(runtime.path_cost(KEY, 1000), 38)
        with self.assertRaises(ValueError):
            runtime.path_cost(KEY, float('nan'))

    def test_independent_switches(self):
        model = OnlineCostModel(enabled=False)
        self.assertFalse(model.update(Sample(KEY, 10)))
        self.assertIsNone(model.predict(KEY))
        with self.assertRaises(ValueError):
            replace(KEY, mode='auto')
        with self.assertRaises(ValueError):
            replace(KEY, mode='cpu')

    def test_demo_retains_synthetic_manifest_and_failure_output(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location('self_calibrating_demo',
                                                     ROOT / 'scripts/run/self_calibrating.py')
        demo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo)
        config = json.loads((ROOT / 'configs/self-calibrating-local.json').read_text())
        scenario = json.loads((ROOT / 'scenarios/self-calibrating-local.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'success'
            result = demo.run(config, scenario, out)
            self.assertEqual(result['initialization_warmup']['steps'], 0)
            self.assertEqual(result['first_ready_request'], 1)
            self.assertEqual(result['first_online_request'], 3)
            self.assertEqual(result['warmup_state'], 'ready')
            self.assertGreater(result['path_cost_ms'], 38)
            self.assertFalse(result['timing_valid_for_performance'])
            self.assertTrue((out / 'events.jsonl').is_file())
            out = Path(tmp) / 'failure'
            with self.assertRaises(ValueError):
                demo.run({**config, 'mode': 'auto'}, scenario, out)
            self.assertTrue((out / 'manifest.json').is_file())
            self.assertTrue((out / 'failure.json').is_file())


if __name__ == '__main__':
    unittest.main()
