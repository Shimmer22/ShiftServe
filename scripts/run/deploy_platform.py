#!/usr/bin/env python3
"""从 shared 增量部署实验资产；逐文件校验，不操作手机设置或用户数据。"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.environment import PKG, default_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=default_device())
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'results' / (stamp + '-platform-deployment')
    out.mkdir(parents=True)
    report = {'device': args.device, 'passed': False, 'files': []}
    adb = ['adb', '-s', args.device]
    with (out / 'deployment.log').open('w') as log:
        def call(command, *, check=True, timeout=60):
            proc = subprocess.run(adb + command, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, timeout=timeout)
            log.write(proc.stdout); log.flush()
            if check:
                proc.check_returncode()
            return proc

        try:
            call(['get-state'])
            report['identity'] = call(['shell', 'getprop ro.product.model; getprop ro.soc.model; '
                                      'getprop ro.build.fingerprint; getprop ro.serialno']).stdout
            call(['shell', 'mkdir -p /data/local/tmp/models /data/local/tmp/llama.cpp/lib'])
            files = sorted((PKG / 'lib').glob('*.so'))
            files += [ROOT.parent / 'shared/models' / name for name in
                      ('Qwen2.5-VL-3B-Instruct-Q4_0.gguf', 'mmproj-model-f16.gguf')]
            for path in files:
                digest = hashlib.sha256()
                with path.open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        digest.update(chunk)
                expected = digest.hexdigest()
                target = '/data/local/tmp/' + ('models/' if path.suffix == '.gguf' else 'llama.cpp/lib/') + path.name
                probe = call(['shell', 'sha256sum', target], check=False)
                actual = probe.stdout.split()[0] if probe.returncode == 0 else None
                reused = actual == expected
                if not reused:
                    print('部署 ' + path.name, flush=True)
                    call(['push', str(path), target], timeout=600)
                    actual = call(['shell', 'sha256sum', target], timeout=120).stdout.split()[0]
                report['files'].append({'source': str(path.resolve()), 'destination': target,
                    'sha256': expected, 'remote_sha256': actual, 'reused': reused, 'passed': actual == expected})
                (out / 'deployment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
                if actual != expected:
                    raise RuntimeError('文件哈希不一致：' + target)
            report['passed'] = True
        except Exception as exc:
            report['failure'] = str(exc)
            raise
        finally:
            (out / 'deployment.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
            print('部署记录：' + str(out), flush=True)


if __name__ == '__main__':
    main()
