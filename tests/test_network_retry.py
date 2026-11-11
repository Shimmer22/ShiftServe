"""断网等待不能重试模型/权限错误，也不能因短暂断网结束调度。"""
import subprocess
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.runtime.network_retry import retry_network


class NetworkRetryTests(unittest.TestCase):
    @patch('a2o.runtime.network_retry.time.sleep')
    def test_reconnect_until_success(self, sleep):
        operation = Mock(side_effect=[
            subprocess.CalledProcessError(255, ['ssh'], stderr=b'Network is unreachable'),
            subprocess.TimeoutExpired(['scp'], 30),
            urllib.error.URLError(ConnectionResetError('Connection reset by peer')),
            'resumed'])
        self.assertEqual(retry_network(operation, report=lambda _: None), 'resumed')
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(operation.call_count, 4)

    @patch('a2o.runtime.network_retry.time.sleep')
    def test_real_failure_is_not_retried(self, sleep):
        for exc in (RuntimeError('device backend unavailable'),
                    subprocess.CalledProcessError(255, ['ssh'], stderr=b'Permission denied'),
                    urllib.error.HTTPError('http://phone', 500, 'backend failed', {}, None)):
            operation = Mock(side_effect=exc)
            with self.assertRaises(type(exc)):
                retry_network(operation)
            self.assertEqual(operation.call_count, 1)
        sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
