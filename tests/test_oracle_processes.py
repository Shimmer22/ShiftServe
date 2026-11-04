from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.validation.oracle import active_model_processes


class OracleProcessTest(unittest.TestCase):
    def test_android_basename_and_full_path(self):
        table = ('PID ARGS\n14126 agent --oracle-profile 1\n'
                 '14127 /data/local/tmp/a2o_test/agent --oracle-profile 1\n'
                 '14128 llama-server --model model.gguf\n')
        self.assertEqual(len(active_model_processes(table)), 3)

    def test_ignore_service_and_inspection_shell(self):
        table = ('PID ARGS\n1943 vendor.xiaomi.hardware.otrpagent@2.0-service\n'
                 '14129 sh -c ps -A -o PID,ARGS\n14130 grep agent\n')
        self.assertEqual(active_model_processes(table), [])


if __name__ == '__main__':
    unittest.main()
