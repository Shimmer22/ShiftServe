"""传输恢复只重试网络故障，不能吞掉校验失败。"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from a2o.runtime.model_install import retry


class ModelInstallRetry(unittest.TestCase):
    def test_lost_connection_retries(self):
        calls = []
        def operation():
            calls.append(1)
            if len(calls)==1:
                raise subprocess.CalledProcessError(255,['ssh'],stderr=b'Connection reset by peer')
            return 'verified'
        with tempfile.TemporaryDirectory() as temp, patch('a2o.runtime.model_install.time.sleep'):
            self.assertEqual(retry(operation,Path(temp)), 'verified')
            self.assertEqual(len(calls),2)

    def test_checksum_error_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp, patch('a2o.runtime.model_install.time.sleep') as sleep:
            with self.assertRaisesRegex(ValueError,'checksum'):
                retry(lambda: (_ for _ in ()).throw(ValueError('checksum')),Path(temp))
            sleep.assert_not_called()

    def test_network_retries_are_bounded(self):
        calls = []
        def operation():
            calls.append(1)
            raise ConnectionResetError('reset')
        with tempfile.TemporaryDirectory() as temp, patch('a2o.runtime.model_install.time.sleep'):
            with self.assertRaises(ConnectionResetError):
                retry(operation,Path(temp),attempts=3)
        self.assertEqual(len(calls),3)


if __name__ == '__main__':
    unittest.main()
