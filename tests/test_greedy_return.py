"""贪心返回决策与异步准备生命周期的原生验证。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class GreedyReturnTest(unittest.TestCase):
    def test_native_decision_timeout_and_error(self):
        compiler=shutil.which('clang++') or shutil.which('g++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as folder:
            binary=Path(folder)/'test'
            subprocess.run([compiler,'-std=c++17','-pthread','-I'+str(ROOT/'src/a2o/native'),
                            str(ROOT/'tests/greedy_return_test.cpp'),'-o',str(binary)],check=True)
            subprocess.run([str(binary)],check=True,timeout=10)


if __name__=='__main__':
    unittest.main()
