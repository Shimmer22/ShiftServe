"""本轮仅新增五项测试；编译一次，以合成成本和可阻塞 ABI 假后端运行。"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class NativeCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('clang++') or shutil.which('g++')
        if compiler is None:
            raise RuntimeError('native tests require a C++ compiler')
        cls.tmp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.tmp.cleanup)
        cls.binaries = {}
        for name in ('native_online_cost', 'execution_lifecycle'):
            binary = str(Path(cls.tmp.name) / name)
            subprocess.run([compiler, '-std=c++17', '-pthread', '-I' + str(ROOT / 'src/a2o/native'),
                            str(ROOT / 'tests' / (name + '_test.cpp')), '-o', binary]
                           + (['-ldl'] if sys.platform.startswith('linux') else []), check=True)
            cls.binaries[name] = binary

    def run_case(self, binary, case):
        subprocess.run([self.binaries[binary], case], check=True, timeout=15)

    def test_offline_baseline_and_observed_path_correction(self):
        self.run_case('native_online_cost', 'routing')

    def test_fixed_decode_horizon_and_dense_to_sparse_sampling(self):
        self.run_case('native_online_cost', 'horizon')

    def test_expiration_capacity_and_execution_identity(self):
        self.run_case('native_online_cost', 'state')

    def test_cancel_inside_api_cross_round_and_close(self):
        self.run_case('execution_lifecycle', 'cancel')

    def test_budget_failure_propagates_and_releases_plan(self):
        self.run_case('execution_lifecycle', 'failure')


if __name__ == '__main__':
    unittest.main()
