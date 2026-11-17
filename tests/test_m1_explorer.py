"""Unit tests for M1Explorer online Bayesian linear routing."""
import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a2o.scheduling.m1_explorer import M1Explorer


def _synth_true_costs(h: int, p: int) -> dict[str, float]:
    h_k = h / 1000.0
    cpu_restore = 45.0 + 0.09 * h
    cpu_prefill = 80.0 + (55.0 + 9.0 * h_k) * p
    htp_prefill = 350.0 + (24.0 + 8.5 * h_k) * p
    htp_to_cpu = 40.0 + 0.25 * (h + p)
    return {
        "cc_restore_ms": cpu_restore,
        "hc_prepare_ms": 50.0 + 0.12 * h,
        "cpu_prefill_ms": cpu_prefill,
        "htp_prefill_ms": htp_prefill,
        "htp_to_cpu_ms": htp_to_cpu,
        "cc_total": cpu_restore + cpu_prefill,
        "hc_total": htp_prefill + htp_to_cpu,
    }


class M1ExplorerTest(unittest.TestCase):
    def test_tiny_probe_bootstrap_and_hp_convergence(self) -> None:
        tiny_probes = [
            {"H": 128, "P": 32, **_synth_true_costs(128, 32)},
            {"H": 128, "P": 128, **_synth_true_costs(128, 128)},
            {"H": 1024, "P": 32, **_synth_true_costs(1024, 32)},
            {"H": 1024, "P": 128, **_synth_true_costs(1024, 128)},
        ]
        exp = M1Explorer(beta=1.25)
        exp.bootstrap_probes(tiny_probes)

        # Extrapolated long-H prediction should have higher uncertainty than short-H
        short_pred = exp.predict_paths(512, 128)
        long_pred = exp.predict_paths(8000, 128)
        self.assertGreater(long_pred["hc_std_ms"], short_pred["hc_std_ms"])

        # Feed observations from realistic long-H tasks and check prediction error drops
        for h, p in [(4800, 100), (6000, 200), (7500, 150), (9000, 300)]:
            gt = _synth_true_costs(h, p)
            exp.observe(h=h, p=p, route="cpu", restore_ms=gt["cc_restore_ms"], prefill_ms=gt["cpu_prefill_ms"])
            exp.observe(h=h, p=p, route="htp", prefill_ms=gt["htp_prefill_ms"], handoff_ms=gt["htp_to_cpu_ms"])

        after = exp.predict_paths(8500, 250)
        gt_eval = _synth_true_costs(8500, 250)
        self.assertLess(abs(after["cc_mean_ms"] - gt_eval["cc_total"]) / gt_eval["cc_total"], 0.08)
        self.assertLess(abs(after["hc_mean_ms"] - gt_eval["hc_total"]) / gt_eval["hc_total"], 0.08)

        exported = exp.export_linear_hp_v1(reference_h=8000)
        self.assertEqual(exported["kind"], "linear_hp_v1")
        self.assertGreater(exported["htp_prefill"]["slope_ms_per_token"], 80.0)

    def test_exploration_escapes_optimism_trap(self) -> None:
        # Poison HTP with an overly pessimistic early sample so greedy locks onto CPU,
        # while M1Explorer's LCB explores HTP at high H/P and discovers HTP is faster.
        exp = M1Explorer(beta=2.0, max_explore_penalty_ms=20000.0, exploration_enabled=True)
        greedy = M1Explorer(beta=0.0, exploration_enabled=False)

        for router in (exp, greedy):
            router.bootstrap_probes([
                {"H": 128, "P": 64, "cc_restore_ms": 50.0, "cpu_prefill_ms": 4000.0, "htp_prefill_ms": 4500.0, "htp_to_cpu_ms": 60.0},
                {"H": 128, "P": 128, "cc_restore_ms": 50.0, "cpu_prefill_ms": 8000.0, "htp_prefill_ms": 9000.0, "htp_to_cpu_ms": 80.0},
                {"H": 1024, "P": 64, "cc_restore_ms": 130.0, "cpu_prefill_ms": 4500.0, "htp_prefill_ms": 5000.0, "htp_to_cpu_ms": 300.0},
            ])
            # Observe CPU a few times at H=6000, P=256 so CPU variance shrinks while HTP stays uncertain
            for _ in range(4):
                router.observe(h=6000, p=256, route="cpu", restore_ms=600.0, prefill_ms=25000.0)

        self.assertEqual(greedy.select_route(6000, 256, htp_ready=True).route, "cpu")

        saw_explore = False
        for _ in range(6):
            dec = exp.select_route(6000, 256, htp_ready=True)
            if dec.explored:
                saw_explore = True
            if dec.route == "htp":
                exp.observe(h=6000, p=256, route="htp", prefill_ms=14000.0, handoff_ms=1500.0)
            else:
                exp.observe(h=6000, p=256, route="cpu", restore_ms=600.0, prefill_ms=25000.0)
        self.assertTrue(saw_explore)
        self.assertEqual(exp.select_route(6000, 256, htp_ready=True).route, "htp")

    def test_dynamic_outlier_isolation_vs_regime_shift(self) -> None:
        exp = M1Explorer()
        exp.bootstrap_probes([
            {"H": 128, "P": 32, **_synth_true_costs(128, 32)},
            {"H": 128, "P": 128, **_synth_true_costs(128, 128)},
            {"H": 1024, "P": 32, **_synth_true_costs(1024, 32)},
            {"H": 1024, "P": 128, **_synth_true_costs(1024, 128)},
        ])
        before = exp.predict_paths(4096, 32)["cc_mean_ms"]
        # Single isolated 16.8s cold-start spike should be isolated without corrupting CPU estimate
        trusted = exp.cpu_prefill.update(exp._prefill_feat(4096, 24), 16810.0)
        self.assertFalse(trusted)
        after_spike = exp.predict_paths(4096, 32)["cc_mean_ms"]
        self.assertAlmostEqual(before, after_spike, delta=1.0)

        # Normal observation resets outlier streak
        self.assertTrue(exp.cpu_prefill.update(exp._prefill_feat(4096, 32), _synth_true_costs(4096, 32)["cpu_prefill_ms"]))

        # Consecutive sustained regime shift (e.g. thermal throttling) should be trusted on 2nd hit
        self.assertFalse(exp.cpu_prefill.update(exp._prefill_feat(4096, 64), 18000.0))
        self.assertTrue(exp.cpu_prefill.update(exp._prefill_feat(4096, 64), 18500.0))
        after_shift = exp.predict_paths(4096, 64)["cc_mean_ms"]
        self.assertGreater(after_shift, 8000.0)


if __name__ == "__main__":
    unittest.main()
