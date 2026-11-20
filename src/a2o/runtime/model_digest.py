"""复用同一设备上未变更模型的启动校验结果。"""

import hashlib
import json
from pathlib import Path
import shlex
import tempfile
import time


def verify_model_digest(shell, device, model, expected, *, force=False, cache_dir=None):
    """shell 执行设备命令并返回 stdout。完整复核后才写入缓存。"""
    begin = time.monotonic()
    path = shlex.quote(model)
    stat_command = f"stat -L -c '%d:%i:%s:%y:%z' -- {path}"
    boot_command = "cat /proc/sys/kernel/random/boot_id 2>/dev/null || printf 'unavailable\\n'"
    before = shell(f"{boot_command}; {stat_command}").splitlines()
    cacheable = len(before) == 2 and before[0] != "unavailable"
    directory = cache_dir if cache_dir is not None else Path.home() / ".cache" / "a2o" / "model-digests"
    key = hashlib.sha256(json.dumps([device, model]).encode()).hexdigest()
    cache = directory / (key + ".json")
    record = dict(device=device, path=model, identity=before, sha256=expected)
    hit = False
    if cacheable and not force:
        try:
            hit = json.loads(cache.read_text()) == record
        except (OSError, ValueError):
            pass
    if not hit:
        actual = shell(f"sha256sum -- {path}").split()[0]
        if actual != expected:
            raise RuntimeError("手机模型 SHA-256 不一致，拒绝启动")
        if shell(f"{boot_command}; {stat_command}").splitlines() != before:
            raise RuntimeError("模型在 SHA-256 校验期间变更，拒绝启动")
        if cacheable:
            try:
                directory.mkdir(mode=0o700, parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(dir=directory) as staging:
                    pending = Path(staging) / "verified.json"
                    pending.write_text(json.dumps(record))
                    pending.replace(cache)
            except OSError:
                # 本地缓存不可写时，本次模型仍已经过完整校验。
                pass
    return dict(sha256=expected, source="cached_stat" if hit else "full_sha256",
                elapsed_ms=(time.monotonic() - begin) * 1000)
