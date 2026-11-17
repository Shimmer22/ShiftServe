"""Unit tests for Mechanism 2 KV-Reduce joint cost table fitting and C++ parity."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from a2o.scheduling.kv_reduce_cost import fit_kv_cost_model, plan_joint_route_kv, plan_kv_reduce


def _synthetic_curve_summary() -> dict:
    return {
        "curve_dir": "synthetic",
        "h_sweep": [
            {"H": 512, "P": 256, "cpu": {"restore_ms": 70.0, "prefill_ms": 3200.0},
             "hc_nocb": {"decode_n_kv": 768, "decode_ms_per_tok": 46.0, "handoff_ms": 180.0, "prefill_ms": 5000.0}},
            {"H": 2048, "P": 256, "cpu": {"restore_ms": 155.0, "prefill_ms": 3200.0},
             "hc_nocb": {"decode_n_kv": 2304, "decode_ms_per_tok": 56.0, "handoff_ms": 330.0, "prefill_ms": 5100.0},
             "hc_prior": {"evict_ms": 1.0}, "cb_eval_tax_ms": 1100.0},
            {"H": 4096, "P": 256, "cpu": {"restore_ms": 260.0, "prefill_ms": 3210.0},
             "hc_nocb": {"decode_n_kv": 4352, "decode_ms_per_tok": 70.0, "handoff_ms": 530.0, "prefill_ms": 5200.0},
             "hc_prior": {"evict_ms": 1.5}, "cb_eval_tax_ms": 1200.0},
            {"H": 8192, "P": 256, "cpu": {"restore_ms": 480.0, "prefill_ms": 3250.0},
             "hc_nocb": {"decode_n_kv": 8448, "decode_ms_per_tok": 96.0, "handoff_ms": 930.0, "prefill_ms": 5400.0},
             "hc_prior": {"evict_ms": 2.6}, "cb_eval_tax_ms": 1300.0},
        ],
        "k_sweep": [
            {"decode_n_kv": 1536, "decode_ms_per_tok": 52.0, "handoff_ms": 260.0},
            {"decode_n_kv": 2048, "decode_ms_per_tok": 55.0, "handoff_ms": 310.0},
            {"decode_n_kv": 3072, "decode_ms_per_tok": 61.0, "handoff_ms": 410.0},
        ],
        "p_sweep": [
            {"P": 64, "cb_eval_tax_ms": 720.0, "cpu_prefill_ms": 800.0, "htp_nocb_prefill_ms": 1500.0},
            {"P": 512, "cb_eval_tax_ms": 1250.0, "cpu_prefill_ms": 6400.0, "htp_nocb_prefill_ms": 10200.0},
        ],
        "cpu_probes": [
            {"meta": {"kv_reduce": 0}, "rows": [{"h_target": 4096, "turn1_prompt_ms": 540.0}, {"h_target": 8192, "turn1_prompt_ms": 950.0}]},
            {"meta": {"kv_reduce": 1}, "rows": [{"h_target": 4096, "turn1_prompt_ms": 1150.0}, {"h_target": 8192, "turn1_prompt_ms": 1560.0}]},
        ],
    }


class KvReduceCostTest(unittest.TestCase):
    def test_deadzone_rejection_quality_floor_and_adaptive_k(self):
        model = fit_kv_cost_model(_synthetic_curve_summary())
        self.assertEqual(model["kind"], "kv_reduce_cost_v1")

        short = plan_kv_reduce(model, h=1024, p=256, expected_d=96, is_htp=False)
        self.assertFalse(short["admit"])
        self.assertEqual(short["reason"], "below_quality_or_evict_floor")

        low_d = plan_kv_reduce(model, h=4096, p=256, expected_d=16, is_htp=True)
        self.assertFalse(low_d["admit"])
        self.assertEqual(low_d["reason"], "negative_net_gain")

        mid = plan_kv_reduce(model, h=4096, p=256, expected_d=96, is_htp=False)
        self.assertTrue(mid["admit"])
        self.assertEqual(mid["keep_tokens"], 1536)

        long_8k = plan_kv_reduce(model, h=8192, p=256, expected_d=96, is_htp=True)
        self.assertTrue(long_8k["admit"])
        self.assertEqual(long_8k["k_floor"], 2304)
        self.assertEqual(long_8k["keep_tokens"], 2304)

    def test_cpp_and_python_exact_parity(self):
        cfg = json.loads((ROOT / "configs/qwen3-1_7b-kv-reduce-cost.json").read_text())
        py_4k = plan_kv_reduce(cfg, h=4096, p=256, expected_d=96, is_htp=False)
        py_8k = plan_kv_reduce(cfg, h=8192, p=256, expected_d=96, is_htp=True)

        compiler = shutil.which("clang++") or shutil.which("g++")
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as folder:
            binary = Path(folder) / "kv_reduce_cost_test"
            subprocess.run(
                [compiler, "-std=c++17", "-I" + str(ROOT / "src/a2o/native"),
                 str(ROOT / "tests/kv_reduce_cost_test.cpp"), "-o", str(binary)],
                check=True,
            )
            out = subprocess.check_output([str(binary)], text=True, timeout=10).strip()
        k4, g4, k8, g8, jr1, jt1, jr2, jt2 = out.split(",")
        self.assertEqual(int(k4), py_4k["keep_tokens"])
        self.assertEqual(int(k8), py_8k["keep_tokens"])
        self.assertAlmostEqual(float(g4), py_4k["expected_gain_ms"], places=6)
        self.assertAlmostEqual(float(g8), py_8k["expected_gain_ms"], places=6)

        rm = {
            "kind": "linear_hp_v1",
            "routing_p_min": 32,
            "h_min": 128,
            "h_max": 1536,
            "p_min": 1,
            "p_max": 384,
            "cpu_restore": {"intercept_ms": 50.0, "slope_ms_per_token": 0.05},
            "cpu_prefill": {"intercept_ms": 20.0, "slope_ms_per_token": 12.0},
            "htp_prefill": {"intercept_ms": 80.0, "slope_ms_per_token": 8.0},
            "htp_to_cpu": {"intercept_ms": 102.90720255782043, "slope_ms_per_token": 0.09779201752845793},
            "cpu_to_htp": {"intercept_ms": 100.0, "slope_ms_per_token": 0.1},
        }
        py_jp1 = plan_joint_route_kv(rm, cfg, 6144, 256, 96, htp_allowed=True, kv_enabled=True)
        py_jp2 = plan_joint_route_kv(rm, cfg, 6144, 16, 96, htp_allowed=True, kv_enabled=True)
        self.assertEqual(jr1, py_jp1["route"])
        self.assertAlmostEqual(float(jt1), py_jp1["selected_total_ms"], places=6)
        self.assertEqual(jr2, py_jp2["route"])
        self.assertAlmostEqual(float(jt2), py_jp2["selected_total_ms"], places=6)
        self.assertTrue(py_jp1["extrapolated"])

        # Logical H_anchor + pinned_tokens prevents k_floor collapse across compact turns.
        compact_guard = plan_kv_reduce(cfg, h=1536, p=512, expected_d=96, is_htp=False, h_logical=8192)
        self.assertFalse(compact_guard["admit"])
        self.assertEqual(compact_guard["k_floor"], 2304)
        self.assertEqual(compact_guard["reason"], "below_quality_or_evict_floor")

        pinned_guard = plan_kv_reduce(cfg, h=4096, p=512, expected_d=96, is_htp=False, h_logical=4096, pinned_tokens=3800)
        self.assertEqual(pinned_guard["k_floor"], 3800)

        # Default handoff_saving_enabled=False avoids counting unverified handoff savings.
        htp_plan = plan_kv_reduce(cfg, h=4096, p=512, expected_d=96, is_htp=True)
        self.assertEqual(htp_plan["handoff_saving_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
