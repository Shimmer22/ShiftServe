"""仅重试连接故障；计算、权限及协议错误保留为失败。"""
from pathlib import Path
import subprocess
import time
import urllib.error


def is_network_error(exc):
    if isinstance(exc, (ConnectionError, subprocess.TimeoutExpired, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        return False
    if isinstance(exc, urllib.error.URLError):
        return is_network_error(exc.reason)
    message = str(exc)
    if isinstance(exc, subprocess.CalledProcessError):
        err = exc.stderr or b''
        out = exc.output or b''
        err_text = err.decode('utf-8', errors='replace') if isinstance(err, bytes) else str(err)
        out_text = out.decode('utf-8', errors='replace') if isinstance(out, bytes) else str(out)
        if exc.returncode == 255 and exc.cmd and Path(str(exc.cmd[0])).name == 'ssh' and not err_text.strip() and not out_text.strip():
            return True
        message += ' ' + err_text + ' ' + out_text
    message = message.lower()
    return any(word in message for word in (
        'network is unreachable', 'no route to host', 'connection timed out',
        'operation timed out', 'connection reset', 'broken pipe',
        'connection closed', 'connection refused', 'could not resolve hostname',
        'temporary failure in name resolution', 'nodename nor servname provided',
        'network is down', 'software caused connection abort',
        'kex_exchange_identification', 'ssh_exchange_identification',
        'client_loop', 'host is down'))


def retry_network(operation, report=print, delay=30):
    """无次数上限；调用者须保证操作可重试或能核对已执行结果。"""
    while True:
        try:
            return operation()
        except Exception as exc:
            if not is_network_error(exc):
                raise
            report('网络暂不可达，30秒后重试：' + str(exc))
            time.sleep(delay)
