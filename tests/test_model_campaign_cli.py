"""直接执行入口，覆盖编译字符串检查未捕获的源码读取/编码错误。"""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ModelCampaignCLI(unittest.TestCase):
    def test_actual_cli_startup_without_device_access(self):
        for script in ['scripts/run/route_phone.py', 'scripts/run/install_model.py',
                       'scripts/run/model_migration_campaign.py', 'scripts/collect/model_cost_campaign.py']:
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT/script), '--help'],
                                        cwd=ROOT, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)


if __name__ == '__main__':
    unittest.main()
