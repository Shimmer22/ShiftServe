"""对实际缓存读取实现做带内存边界检查的原生测试。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

class NativePrepackTest(unittest.TestCase):
    def test_actual_kv_dma_and_tail_layout(self):
        compiler = shutil.which('clang++')
        self.assertIsNotNone(compiler)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / 'a2o-kv-prepack.h').symlink_to(ROOT / 'src/a2o/native/state/htp-kv-prepack.h')
            for name in ('pack','main','load'):
                (Path(tmp) / f'a2o-kv-prepack-{name}.inc').symlink_to(ROOT / f'src/a2o/native/state/htp-kv-prepack-{name}.inc')
            (Path(tmp) / 'a2o-kv-prepack-workspace.inc').write_text('// 原生资源获取由测试 mock；实际分块与 DMA 顺序仍运行生产源码。\n')
            binary = str(Path(tmp) / 'kv-prepack-test')
            subprocess.run([compiler, '-std=c++17', '-O1', '-g', '-fsanitize=address,undefined',
                            '-I' + str(ROOT / 'src/a2o/native'), '-I' + tmp,
                            str(ROOT / 'tests/kv_prepack_load_test.cpp'), '-o', binary], check=True)
            subprocess.run([binary], check=True)

if __name__ == '__main__':
    unittest.main()
