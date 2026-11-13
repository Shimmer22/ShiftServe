#!/usr/bin/env python3
"""持锁安装已校验的共享GGUF；临时文件验证后原子发布，不替换旧模型。"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import hashlib
import shutil
import time

ROOT = Path(__file__).resolve().parents[3]
from a2o.profiling.model_cost_campaign import digest, save
from a2o.runtime.device_transport import DeviceTransport
from a2o.validation.oracle import active_model_processes
from a2o.runtime.network_retry import is_network_error


def retry(operation, out, attempts=8):
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            if not is_network_error(exc) or attempt+1==attempts:
                raise
            save(out / 'network-retry.json', dict(attempt=attempt+1, error=repr(exc), updated_unix=time.time()))
            time.sleep(10)


def receive_guard(transport, partial, receiver):
    """只清理本owner为本partial启动的有限长度接收器。"""
    q = shlex.quote
    owner = 'test "$(cat '+q(transport.remote_root+'/model.lock/owner')+')" = '+q(transport.lock_owner)
    cleanup = ('if [ -f '+q(receiver)+' ]; then p=$(cat '+q(receiver)+'); '
               'if kill -0 "$p" 2>/dev/null && tr "\\000" " " < /proc/"$p"/cmdline | '
               '/system/bin/grep -F '+q('of='+partial)+' >/dev/null; then kill "$p"; fi; fi')
    transport.call('shell', owner+' && { '+cleanup+'; }')


def chunk_transfer(transport, source_path, partial, out):
    q = shlex.quote
    receiver = partial+'.receiver.pid'
    total = source_path.stat().st_size
    size = int(transport.call('shell', 'if [ -f '+q(partial)+' ]; then stat -c %s '+q(partial)+'; else echo 0; fi').stdout)
    assert 0 <= size <= total, 'partial长度异常'
    offset = max(0, size-16*1024*1024)
    receive_guard(transport, partial, receiver)
    transport.call('shell', '/system/bin/toybox truncate -s '+str(offset)+' '+q(partial))
    save(out/'resume.json', dict(previous_bytes=size, resume_offset=offset, total_bytes=total))
    chunk_size = 32*1024*1024
    with source_path.open('rb') as source:
        source.seek(offset)
        while offset<total:
            data = source.read(min(chunk_size,total-offset))
            expected = hashlib.sha256(data).hexdigest()

            def transfer_one():
                receive_guard(transport, partial, receiver)
                command = ('echo $$ > '+q(receiver)+'; exec /system/bin/toybox dd '
                           'of='+q(partial)+' bs=1048576 count='+str(len(data))+
                           ' iflag=count_bytes seek='+str(offset)+' oflag=seek_bytes conv=notrunc status=none')
                run = subprocess.run(transport.shell_command(command), input=data,
                                     capture_output=True, timeout=180)
                with (out/'transfer.log').open('ab') as log:
                    log.write(run.stdout+run.stderr)
                if run.returncode:
                    raise subprocess.CalledProcessError(run.returncode, run.args, run.stdout, run.stderr)
                check = ('/system/bin/toybox dd if='+q(partial)+' bs=1048576 skip='+str(offset)+
                         ' count='+str(len(data))+' iflag=skip_bytes,count_bytes status=none | sha256sum')
                actual = transport.call('shell',check).stdout.decode().split()[0]
                if actual!=expected:
                    raise RuntimeError('分块SHA-256不匹配，停止并保留partial')

            retry(transfer_one,out)
            offset += len(data)
            save(out/'progress.json', dict(phase='transferring', verified_bytes=offset,total_bytes=total,
                                          chunk_sha256=expected,updated_unix=time.time()))
            print('verified_bytes',offset,'/',total,flush=True)
    receive_guard(transport,partial,receiver)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/model-default.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true', help='同一owner断点续传，重传末尾16MiB后校验完整文件')
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prior = list(out.glob('*.json'))
    if prior:
        archive = out/'attempts'/str(time.time_ns())
        archive.mkdir(parents=True)
        for path in prior:
            shutil.copy2(path,archive/path.name)
    actual = digest(args.source)
    assert actual == cfg['model_sha256'], '本地GGUF校验失败'
    transport = DeviceTransport(ROOT, 'ssh', config=ROOT / 'configs/termux-ssh.json')
    original_call = transport.call
    transport.call = lambda *parts: retry(lambda: original_call(*parts),out)
    q = shlex.quote
    ps = transport.call('shell', 'ps -A -o PID,ARGS').stdout.decode()
    (out / 'processes-before.txt').write_text(ps)
    assert not active_model_processes(ps), '设备有模型进程，拒绝安装并发'
    primary_error = False
    try:
        retry(lambda: transport.prepare_ssh(out, resume_lock=args.resume),out,attempts=3)
        remote = cfg['model_path']
        exists = transport.call('shell', 'if [ -f ' + q(remote) + ' ]; then sha256sum ' + q(remote) + '; else echo absent; fi').stdout.decode().split()[0]
        if exists != 'absent':
            assert exists == actual, '同名模型校验不同，拒绝覆盖'
        else:
            partial = remote + '.partial'
            if args.resume:
                chunk_transfer(transport,args.source,partial,out)
            else:
                command = ['scp', '-P', str(transport.config['port']), *transport.options,
                           str(args.source.resolve()), transport.target + ':' + partial]
                with (out / 'transfer.log').open('a') as log:
                    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=3600, check=True)
            copied = transport.call('shell', 'sha256sum ' + q(partial)).stdout.decode().split()[0]
            assert copied == actual, '手机GGUF校验失败，保留partial'
            transport.call('shell', 'test ! -e ' + q(remote) + ' && mv ' + q(partial) + ' ' + q(remote))
        save(out / 'completion.json', dict(installed=True, local=str(args.source.resolve()), remote=remote, sha256=actual))
        print(json.dumps(dict(installed=True, model=remote, sha256=actual)), flush=True)
    except BaseException as exc:
        primary_error = True
        save(out / 'failure.json', dict(error=repr(exc)))
        raise
    finally:
        cleanup={}
        try:
            if transport.lock_owner:
                receive_guard(transport,cfg['model_path']+'.partial',cfg['model_path']+'.partial.receiver.pid')
                transport.release_lock()
            cleanup['lock_released']=True
        except Exception as exc:
            cleanup.update(lock_released=False,error=repr(exc))
        try:
            retry(lambda:subprocess.run(transport.prefix+['termux-wake-unlock'],capture_output=True,timeout=30,check=True),out,attempts=3)
            cleanup['wake_unlocked']=True
        except Exception as exc:
            cleanup.update(wake_unlocked=False,wake_error=repr(exc))
        save(out/'cleanup.json',cleanup)
        if not primary_error and not (cleanup.get('lock_released') and cleanup.get('wake_unlocked')):
            raise RuntimeError('模型已校验，但安装清理未完成；禁止启动后续实验：'+json.dumps(cleanup))


if __name__ == '__main__':
    main()
