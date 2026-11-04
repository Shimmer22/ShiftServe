"""显式白名单只读 Tool，不执行模型生成动作。"""
import subprocess
from urllib.request import urlopen
from urllib.parse import urlparse


def execute_tool(spec, device):
    kind = spec['type']
    if kind == 'device_display':
        # 仅查询显示尺寸；不设置分辨率，不执行模型生成的命令。
        proc = subprocess.run(['adb', '-s', device, 'shell', 'wm', 'size'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30, check=True)
        if not proc.stdout.strip():
            raise RuntimeError('display query returned no observation')
        return proc.stdout
    if kind == 'http_get':
        url = spec['url']
        if urlparse(url).scheme not in {'http', 'https'}:
            raise ValueError('HTTP tool requires http/https URL')
        with urlopen(url, timeout=30) as response:
            raw = response.read(65537)
            if len(raw) > 65536:
                raise ValueError('HTTP tool result exceeds 64 KiB; refusing truncation')
            return raw
    raise ValueError('unknown read-only tool: ' + str(kind))
