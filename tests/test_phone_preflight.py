import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock

from a2o.runtime.phone_preflight import PhonePreflightError, require_phone_preflight


def frame(*background):
    rows = [
        "10 90 bash bash",
        "11 90 top top",
        "12 240 service /runs/eval/service",
        *background,
    ]
    return f"Tasks: {len(rows)} total, 1 running\n PID %CPU NAME [ARGS]\n" + "\n".join(rows) + "\n"


class PhonePreflightTest(unittest.TestCase):
    def check_phone(self, output, frames):
        data = "A2O_PHONE_PREFLIGHT 10 11 12\n" + "".join(frames)
        transport = Mock(kind="ssh", identity="test-phone", remote_root="/runs")
        transport.call.return_value = subprocess.CompletedProcess([], 0, data.encode(), b"")
        return require_phone_preflight(transport, output, attached=True, expected_owner="eval")

    def test_exactly_15_passes_and_verified_service_is_excluded(self):
        with tempfile.TemporaryDirectory() as path:
            report = json.loads(self.check_phone(path, [
                frame("30 90 daemon daemon"),
                frame("30 7.2 daemon daemon", "31 7.8 worker worker"),
                frame("31 2.1 worker worker"),
            ]).read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["background_peak_cpu_pct"], 15)
            self.assertEqual(report["locked_service_pid"], 12)

    def test_above_limit_in_either_interval_is_fatal_and_recorded(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(PhonePreflightError, "15.1% > 15%"):
                self.check_phone(path, [
                    frame(), frame("30 9.1 daemon daemon", "31 6.0 worker worker"), frame(),
                ])
            report_path = next(Path(path).glob("*.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "failed")
            self.assertTrue((report_path.parent / report["raw_log"]).is_file())

    def test_idle_tr_in_initial_frame_is_fatal(self):
        for args in ["tr tr -d x", "toybox /system/bin/toybox tr -d x"]:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as path:
                with self.assertRaisesRegex(PhonePreflightError, "残留 tr PID 30"):
                    self.check_phone(path, [frame(f"30 0 {args}"), frame(), frame()])

    def test_missing_interval_fails_closed(self):
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(PhonePreflightError, "missing sampler identity or CPU interval"):
                self.check_phone(path, [frame(), frame()])

    def test_thermal_cooldown_retries_until_cool_and_unthrottled(self):
        hot = (
            "A2O_PHONE_PREFLIGHT 10 11 12\n"
            "A2O_PHONE_TELEMETRY quiet=38200 batt=36000 p2=2707200 p5=2707200 p7=2246400 mem_avail_kb=10000000 swap_free_kb=12000000\n"
            + "".join([frame(), frame(), frame()])
        )
        cool = (
            "A2O_PHONE_PREFLIGHT 10 11 12\n"
            "A2O_PHONE_TELEMETRY quiet=35100 batt=34000 p2=2707200 p5=2707200 p7=2745600 mem_avail_kb=10000000 swap_free_kb=12000000\n"
            + "".join([frame(), frame(), frame()])
        )
        transport = Mock(kind="ssh", identity="test-phone", remote_root="/runs")
        transport.call.side_effect = [
            subprocess.CompletedProcess([], 0, hot.encode(), b""),
            subprocess.CompletedProcess([], 0, cool.encode(), b""),
        ]
        with tempfile.TemporaryDirectory() as path:
            report_path = require_phone_preflight(
                transport, path, attached=True, expected_owner="eval",
                max_quiet_c=37.0, min_prime_khz=2400000, min_mem_avail_kib=4000000,
                cooldown_timeout_s=10, cooldown_poll_s=0,
            )
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["attempts"], 2)
            self.assertAlmostEqual(report["telemetry"]["quiet_c"], 35.1)

    def test_thermal_throttle_without_cooldown_fails(self):
        throttled = (
            "A2O_PHONE_PREFLIGHT 10 11 12\n"
            "A2O_PHONE_TELEMETRY quiet=39500 batt=38000 p2=2100000 p5=2100000 p7=1900000 mem_avail_kb=10000000 swap_free_kb=12000000\n"
            + "".join([frame(), frame(), frame()])
        )
        transport = Mock(kind="ssh", identity="test-phone", remote_root="/runs")
        transport.call.return_value = subprocess.CompletedProcess([], 0, throttled.encode(), b"")
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaisesRegex(PhonePreflightError, "机身温度 39.5°C > 37.0°C"):
                require_phone_preflight(
                    transport, path, attached=True, expected_owner="eval",
                    max_quiet_c=37.0, min_prime_khz=2400000, cooldown_timeout_s=0,
                )

    def test_transient_ssh_drop_retries_and_passes(self):
        from unittest.mock import patch
        data = "A2O_PHONE_PREFLIGHT 10 11 12\n" + "".join([frame(), frame(), frame()])
        transport = Mock(kind="ssh", identity="test-phone", remote_root="/runs")
        transport.call.side_effect = [
            subprocess.CalledProcessError(255, ["ssh"], stderr=b"kex_exchange_identification: Connection closed by remote host"),
            subprocess.CompletedProcess([], 0, data.encode(), b""),
        ]
        with patch("a2o.runtime.network_retry.time.sleep") as mock_sleep, tempfile.TemporaryDirectory() as path:
            report_path = require_phone_preflight(transport, path, attached=True, expected_owner="eval")
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "passed")
            mock_sleep.assert_called_once_with(30)


if __name__ == "__main__":
    unittest.main()
