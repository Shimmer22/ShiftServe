"""SSH失败不能回退ADB，挂起探针不通过时不能取得模型执行锁。"""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

from a2o.runtime.device_transport import DeviceTransport


class DeviceTransportTest(unittest.TestCase):
    def make_transport(self, directory):
        p = Path(directory)
        config = p / 'ssh.json'
        config.write_text(json.dumps(dict(host='example.invalid', port=8022, user='test',
            identity_file='~/.ssh/test', known_hosts='known_hosts', remote_root='/private/runs',
            clock_probe='/private/clock')))
        return DeviceTransport(p, 'ssh', config=config)

    def test_forward_failure_is_explicit_and_never_calls_adb(self):
        with tempfile.TemporaryDirectory() as d:
            transport = self.make_transport(d)
            failed = Mock()
            failed.poll.return_value = 255
            failed.stderr.read.return_value = b'port forwarding denied'
            with patch('a2o.runtime.device_transport.subprocess.Popen', return_value=failed) as popen, \
                    patch('a2o.runtime.device_transport.time.sleep'):
                with self.assertRaisesRegex(RuntimeError, 'port forwarding denied'):
                    transport.call('reverse', 'tcp:1234', 'tcp:5678')
            command = popen.call_args.args[0]
            self.assertEqual(command[0], 'ssh')
            self.assertIn('ExitOnForwardFailure=yes', command)

    def test_suspension_rejects_model_lock(self):
        with tempfile.TemporaryDirectory() as d:
            transport = self.make_transport(d)
            transport.call = Mock()
            data = '\n'.join(json.dumps(dict(monotonic_s=m, boottime_s=b))
                             for m, b in [(0, 0), (5, 15)])
            with patch('a2o.runtime.device_transport.subprocess.run', side_effect=[
                    subprocess.CompletedProcess([], 0, b'', b''),
                    subprocess.CompletedProcess([], 0, data.encode(), b'')]):
                with self.assertRaisesRegex(RuntimeError, 'clock precondition failed'):
                    transport.prepare_ssh(Path(d))
            self.assertIsNone(transport.lock_owner)
            self.assertEqual(transport.call.call_count, 1)

    def test_android_shell_does_not_use_termux_ps(self):
        with tempfile.TemporaryDirectory() as d:
            transport = self.make_transport(d)
            command = transport.shell_command('ps -A -o PID,ARGS')
            self.assertIn('StrictHostKeyChecking=yes', command)
            self.assertTrue(command[-1].startswith('export PATH=/system/bin:'))


if __name__ == '__main__':
    unittest.main()
