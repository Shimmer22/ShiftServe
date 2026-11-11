from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReplayCacheTest(unittest.TestCase):
    def test_native_cache_lifecycle(self):
        compiler = shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler)
        vendor = ROOT.parent / 'shared/runtime/llama.cpp/vendor'
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / 'cache-test'
            subprocess.run([compiler, '-std=c++17', '-I' + str(vendor),
                            '-I' + str(ROOT / 'src/a2o/native'),
                            str(ROOT / 'tests/replay_cache_test.cpp'), '-o', str(binary)], check=True)
            subprocess.run([str(binary), str(Path(tmp) / 'cache')], check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
