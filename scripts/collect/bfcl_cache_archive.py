#!/usr/bin/env python3
"""在手机和 wins 私有目录间流式备份/恢复冻结状态，不在 Mac 留大文件副本。"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.device_transport import DeviceTransport


def stream(source, destination):
    # tar 直接经过管道；stderr 用文件，避免大输出阻塞管道。
    import tempfile
    with tempfile.TemporaryFile() as errors:
        producer = subprocess.Popen(source, stdout=subprocess.PIPE, stderr=errors)
        try:
            consumer = subprocess.Popen(destination, stdin=producer.stdout, stdout=subprocess.DEVNULL, stderr=errors)
            producer.stdout.close()
            consumer_code = consumer.wait()
            producer_code = producer.wait()
        finally:
            if producer.poll() is None:
                producer.terminate()
                producer.wait()
        if producer_code or consumer_code:
            errors.seek(0)
            raise RuntimeError(errors.read().decode(errors='replace'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['backup', 'restore'])
    parser.add_argument('--phone-directory', required=True)
    parser.add_argument('--phone-owner', required=True)
    parser.add_argument('--host-directory', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    if not args.phone_directory.startswith(transport.remote_root + '/') or not args.host_directory.startswith('/home/wins/ai-workspace/cache/a2o-frozen-replay/'):
        raise ValueError('缓存只能位于项目私有缓存目录')
    q = shlex.quote
    guard = 'test "$(cat ' + q(transport.remote_root + '/model.lock/owner') + ')" = ' + q(args.phone_owner)
    host = ['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', 'wins']
    stage = args.host_directory + '.incoming-' + uuid.uuid4().hex
    if args.action == 'backup':
        source = guard + ' && cd ' + q(args.phone_directory) + ' && sha256sum *.json *.state > SHA256SUMS && tar -cf - .'
        destination = 'mkdir -p ' + q(stage) + ' && tar -C ' + q(stage) + ' -xf - && cd ' + q(stage) + ' && sha256sum -c SHA256SUMS >/dev/null'
        stream(transport.shell_command(source), host + [destination])
        # 校验完成才发布；已有同名快照摘要不同则拒绝覆盖，保留 incoming。
        publish = '''import hashlib,json,os,sys
from pathlib import Path
stage,target=map(Path,sys.argv[1:]);target.mkdir(parents=True,exist_ok=True)
files=list(stage.iterdir())
for path in files:
 if path.name=='SHA256SUMS':continue
 existing=target/path.name
 if existing.exists() and hashlib.sha256(existing.read_bytes()).digest()!=hashlib.sha256(path.read_bytes()).digest():
  raise RuntimeError('existing cache differs: '+str(existing))
for path in files:
 if path.name!='SHA256SUMS':os.replace(path,target/path.name)
(stage/'SHA256SUMS').unlink();stage.rmdir()
entries=[p for p in sorted(target.iterdir()) if p.suffix in ('.json','.state')]
(target/'SHA256SUMS.tmp').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\\n' for p in entries))
os.replace(target/'SHA256SUMS.tmp',target/'SHA256SUMS')
print(json.dumps(dict(files=len(entries),state_files=sum(p.suffix=='.state' for p in entries),bytes=sum(p.stat().st_size for p in entries))))
'''
        result = subprocess.run(host + [shlex.join(['python3', '-c', publish, stage, args.host_directory])], check=True, capture_output=True, text=True)
        report = json.loads(result.stdout)
    else:
        source = 'cd ' + q(args.host_directory) + ' && sha256sum -c SHA256SUMS >/dev/null && tar -cf - .'
        phone_stage = args.phone_directory + '.incoming-' + uuid.uuid4().hex
        destination = (guard + ' && mkdir -p ' + q(phone_stage) + ' && tar -C ' + q(phone_stage) +
                       ' -xf - && cd ' + q(phone_stage) + ' && sha256sum -c SHA256SUMS >/dev/null && test ! -e ' +
                       q(args.phone_directory) + ' && mv ' + q(phone_stage) + ' ' + q(args.phone_directory))
        stream(host + [source], transport.shell_command(destination))
        report = dict(restored=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(action=args.action, host='wins', host_directory=args.host_directory,
         phone_directory=args.phone_directory, verified_sha256=True, completed_unix=time.time(), **report), indent=2) + '\n')


if __name__ == '__main__':
    main()
