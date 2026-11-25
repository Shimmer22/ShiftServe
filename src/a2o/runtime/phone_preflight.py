"""为手机实验检查 Termux 可见进程，留存原始采样和门禁结论。"""
from __future__ import annotations

from decimal import Decimal
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from urllib.parse import urlsplit

from .network_retry import retry_network

CPU_LIMIT = Decimal("15")
INTERVALS = 2
TASKS = re.compile(r"^Tasks:\s+(\d+) total,")
HEADER = re.compile(r"^\s*PID\s+%CPU\s+NAME\s+\[?ARGS\]?\s*$")
PROCESS = re.compile(r"^\s*(\d+)\s+(\d+(?:\.\d+)?)\s+(\S+)(?:\s+(.*))?$")
MARKER = "A2O_PHONE_PREFLIGHT"
TELEMETRY_MARKER = "A2O_PHONE_TELEMETRY"
REPORT_ENV = "A2O_PHONE_PREFLIGHT_REPORT"


def _parse_telemetry(line):
    raw = {}
    for token in line.split()[1:]:
        if "=" in token:
            k, v = token.split("=", 1)
            raw[k] = v

    def to_temp(val):
        if val is None or val == "na":
            return None
        num = float(val)
        return num / 1000.0 if num > 200.0 else num

    def to_int(val):
        if val is None or val == "na":
            return None
        return int(val)

    return dict(
        quiet_c=to_temp(raw.get("quiet")),
        battery_c=to_temp(raw.get("batt")),
        policy2_smax_khz=to_int(raw.get("p2")),
        policy5_smax_khz=to_int(raw.get("p5")),
        policy7_smax_khz=to_int(raw.get("p7")),
        mem_avail_kib=to_int(raw.get("mem_avail_kb")),
        swap_free_kib=to_int(raw.get("swap_free_kb")),
    )


class PhonePreflightError(RuntimeError):
    """脏环境或无法检查时拒绝开始测量。"""


def _is_tr(name, args):
    if name.rsplit("/", 1)[-1] == "tr":
        return True
    words = args.split(maxsplit=2)
    if not words:
        return False
    executable = words[0].rsplit("/", 1)[-1]
    return executable == "tr" or (
        executable in {"toybox", "busybox"} and len(words) > 1 and words[1] == "tr"
    )


def _parse_snapshot(raw):
    frames = []
    frame = None
    markers = []
    telemetry = None
    reading_rows = False
    for line in raw.splitlines():
        if line.startswith(MARKER + " "):
            markers.append(tuple(map(int, line.split()[1:])))
            continue
        if line.startswith(TELEMETRY_MARKER + " "):
            telemetry = _parse_telemetry(line)
            continue
        if match := TASKS.match(line):
            frame = (int(match[1]), [])
            frames.append(frame)
            reading_rows = False
        elif HEADER.fullmatch(line):
            if frame is None or reading_rows:
                raise ValueError("unexpected process table header")
            reading_rows = True
        elif reading_rows and line.strip():
            match = PROCESS.fullmatch(line)
            if not match:
                raise ValueError("invalid process row")
            pid, cpu, name, args = match.groups()
            frame[1].append((int(pid), Decimal(cpu), name, args or ""))
    if len(markers) != 1 or len(markers[0]) != 3 or len(frames) != INTERVALS + 1:
        raise ValueError("missing sampler identity or CPU interval")
    shell, sampler, service = markers[0]
    if shell <= 0 or sampler <= 0 or shell == sampler or service < 0 or service in {shell, sampler}:
        raise ValueError("invalid sampler identity")
    if any(len(rows) != count or len({row[0] for row in rows}) != count for count, rows in frames):
        raise ValueError("incomplete process table")

    def item(row):
        pid, cpu, name, args = row
        return dict(pid=pid, cpu_pct=float(cpu), name=name, args=args)

    tr = {row[0]: item(row) for _, rows in frames for row in rows if _is_tr(row[2], row[3])}
    exclusions = {shell, sampler, service}
    samples = []
    # 第一帧没有完整采样窗口，只用来检查 tr。
    for number, (_, rows) in enumerate(frames[1:], 1):
        if sampler not in {r[0] for r in rows} or (service and service not in {r[0] for r in rows}):
            raise ValueError("sampler or locked service missing from process table")
        background = sorted((r for r in rows if r[0] not in exclusions), key=lambda r: -r[1])
        total = sum((r[1] for r in background), Decimal(0))
        samples.append((total, dict(interval=number, total_cpu_pct=float(total),
                                    processes=[item(r) for r in background])))
    peak = max(total for total, _ in samples)
    result = dict(background_peak_cpu_pct=float(peak), residual_tr=list(tr.values()),
                  locked_service_pid=service or None, sampler_pids=[shell, sampler],
                  samples=[summary for _, summary in samples])
    if telemetry is not None:
        result["telemetry"] = telemetry
    return result, peak


def _sample_command(transport, attached, expected_owner):
    service_check = "service=0; "
    if attached:
        lock_root = shlex.quote(transport.remote_root)
        owner_check = (f'test "$owner" = {shlex.quote(expected_owner)}; ' if expected_owner else "")
        service_check = (
            f'root={lock_root}; IFS= read -r owner < "$root/model.lock/owner"; '
            + owner_check
            + 'case "$owner" in ""|.|..|*[!A-Za-z0-9._-]*) exit 2;; esac; '
            'if [ "$owner" = qwen35-phone-server ]; then '
            'bin="${root%/runs}/qwen35-server/llama-server"; else bin="$root/$owner/service"; fi; '
            'IFS= read -r service < "${bin%/*}/pid"; '
            'case "$service" in ""|*[!0-9]*) exit 2;; esac; '
            'test "$(/system/bin/readlink "/proc/$service/exe")" = "$bin"; '
        )
    return (
        "set -e; " + service_check
        + f"/system/bin/top -b -n {INTERVALS + 1} -d 1 -o PID,%CPU,NAME,ARGS & sampler=$!; "
        + f'printf "{MARKER} %s %s %s\\n" "$$" "$sampler" "$service"; wait "$sampler"; '
        + 'qt=$(cat /sys/class/thermal/thermal_zone64/temp 2>/dev/null || echo na); '
        + 'bt=$(cat /sys/class/thermal/thermal_zone75/temp 2>/dev/null || echo na); '
        + 'f2=$(cat /sys/devices/system/cpu/cpufreq/policy2/scaling_max_freq 2>/dev/null || echo na); '
        + 'f5=$(cat /sys/devices/system/cpu/cpufreq/policy5/scaling_max_freq 2>/dev/null || echo na); '
        + 'f7=$(cat /sys/devices/system/cpu/cpufreq/policy7/scaling_max_freq 2>/dev/null || echo na); '
        + 'ma=na; sf=na; if [ -r /proc/meminfo ]; then '
        + 'while IFS=" :" read -r k v _; do case "$k" in MemAvailable) ma="$v";; SwapFree) sf="$v";; esac; done < /proc/meminfo; fi; '
        + f'printf "{TELEMETRY_MARKER} quiet=%s batt=%s p2=%s p5=%s p7=%s mem_avail_kb=%s swap_free_kb=%s\\n" '
        + '"$qt" "$bt" "$f2" "$f5" "$f7" "$ma" "$sf"; '
        + ('test "$(/system/bin/readlink "/proc/$service/exe")" = "$bin"; ' if attached else "")
        + ":"
    )


def _check_environment(measured, *, max_quiet_c=None, min_prime_khz=None, min_mem_avail_kib=None):
    telemetry = measured.get("telemetry") or {}
    thermal_reasons = []
    other_reasons = []
    if max_quiet_c is not None:
        quiet_c = telemetry.get("quiet_c")
        if quiet_c is None:
            other_reasons.append("缺少机身温度采样")
        elif quiet_c > max_quiet_c:
            thermal_reasons.append(f"机身温度 {quiet_c:.1f}°C > {max_quiet_c:.1f}°C")
    if min_prime_khz is not None:
        clocks = [telemetry.get(k) for k in ("policy2_smax_khz", "policy5_smax_khz", "policy7_smax_khz")]
        if any(c is None for c in clocks):
            other_reasons.append("缺少 CPU scaling_max_freq 采样")
        elif any(c < min_prime_khz for c in clocks):
            thermal_reasons.append(
                f"CPU 上限降频 p2/p5/p7={clocks[0]}/{clocks[1]}/{clocks[2]} kHz < {min_prime_khz}"
            )
    if min_mem_avail_kib is not None:
        mem = telemetry.get("mem_avail_kib")
        if mem is None:
            other_reasons.append("缺少 MemAvailable 采样")
        elif mem < min_mem_avail_kib:
            other_reasons.append(f"MemAvailable {mem} KiB < {min_mem_avail_kib} KiB")
    return thermal_reasons, other_reasons


def require_phone_preflight(transport, output, *, attached=False, expected_owner=None, call=None,
                            max_quiet_c=None, min_prime_khz=None, min_mem_avail_kib=None,
                            cooldown_timeout_s=0, cooldown_poll_s=15):
    """前两个完整区间中，任一区间的非测试进程 CPU 超过 15% 或处于降频/过热状态即拒绝。"""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"phone-preflight-{time.time_ns()}"
    raw_path = stem.with_suffix(".txt")
    report_path = stem.with_suffix(".json")
    report = dict(status="failed", device=transport.identity, cpu_limit_pct=float(CPU_LIMIT),
                  cpu_unit="one logical core = 100%", raw_log=raw_path.name,
                  scope="Termux-visible processes" if transport.kind == "ssh" else "ADB-visible processes")
    if max_quiet_c is not None:
        report["max_quiet_c"] = float(max_quiet_c)
    if min_prime_khz is not None:
        report["min_prime_khz"] = int(min_prime_khz)
    if min_mem_avail_kib is not None:
        report["min_mem_avail_kib"] = int(min_mem_avail_kib)
    started = time.monotonic()
    deadline = started + max(0, cooldown_timeout_s)
    attempts = 0
    thermal_history = []
    try:
        while True:
            attempts += 1
            cmd = _sample_command(transport, attached, expected_owner)
            proc = retry_network(lambda: (call or transport.call)("shell", cmd))
            raw_path.write_bytes(proc.stdout + proc.stderr)
            measured, peak = _parse_snapshot(proc.stdout.decode())
            report.update(measured)
            report["attempts"] = attempts
            report["cooldown_wait_s"] = round(time.monotonic() - started, 2)
            tel = measured.get("telemetry")
            if tel:
                thermal_history.append(dict(attempt=attempts, elapsed_s=report["cooldown_wait_s"], bg_cpu_pct=float(peak), **tel))
                report["thermal_history"] = thermal_history
                print(
                    f"PREFLIGHT #{attempts} bg_cpu={float(peak):.1f}% quiet={tel.get('quiet_c')}°C "
                    f"p2/p5/p7={tel.get('policy2_smax_khz')}/{tel.get('policy5_smax_khz')}/{tel.get('policy7_smax_khz')}kHz "
                    f"mem={tel.get('mem_avail_kib')}KiB",
                    flush=True,
                )
            reasons = []
            if measured["residual_tr"]:
                reasons.append("残留 tr PID " + ", ".join(str(row["pid"]) for row in measured["residual_tr"]))
            if peak > CPU_LIMIT:
                reasons.append(f"非测试进程 CPU 合计 {peak:.1f}% > {CPU_LIMIT}%")
            thermal_reasons, env_reasons = _check_environment(
                measured, max_quiet_c=max_quiet_c, min_prime_khz=min_prime_khz,
                min_mem_avail_kib=min_mem_avail_kib,
            )
            reasons.extend(env_reasons)
            if not reasons and thermal_reasons and time.monotonic() < deadline:
                print(f"手机散热等待中（{'; '.join(thermal_reasons)}），{cooldown_poll_s}s 后重检", flush=True)
                time.sleep(cooldown_poll_s)
                continue
            reasons.extend(thermal_reasons)
            if reasons:
                raise PhonePreflightError("；".join(reasons))
            report["status"] = "passed"
            break
    except (OSError, subprocess.SubprocessError, ValueError, PhonePreflightError) as exc:
        if not raw_path.exists():
            raw_path.write_bytes((getattr(exc, "stdout", None) or b"") + (getattr(exc, "stderr", None) or b""))
        report["reason"] = str(exc)
        raise PhonePreflightError(f"手机 Pre-flight 中止：{exc}；记录 {report_path}") from exc
    finally:
        report["checked_unix_ns"] = time.time_ns()
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report_path


def require_attached_phone_preflight(root, output, *, expected_owner=None, **kwargs):
    from .device_transport import DeviceTransport

    root = Path(root)
    transport = DeviceTransport(root, "ssh", config=root / "configs/termux-ssh.json")
    return require_phone_preflight(transport, output, attached=True, expected_owner=expected_owner, **kwargs)


def docker_preflight_args(root, output, *, expected_owner=None):
    """Docker 内没有手机 SSH；在启动 Docker 命令的控制端执行实测。"""
    report = require_attached_phone_preflight(root, output, expected_owner=expected_owner)
    summary = json.loads(report.read_text())
    for sample in summary["samples"]:
        del sample["processes"]
    payload = json.dumps(summary, ensure_ascii=False, separators=(",", ":"))
    return ["--env", f"{REPORT_ENV}={payload}"]


def require_http_phone_preflight(root, endpoint, output):
    """兼容使用手机 HTTP 的评测和同入口的纯主机评测。"""
    root, output = Path(root), Path(output)
    config = json.loads((root / "configs/termux-ssh.json").read_text())
    if urlsplit(endpoint).hostname != config["host"]:
        return
    if REPORT_ENV not in os.environ:
        require_attached_phone_preflight(root, output)
        return
    path = output / "phone-preflight-controller.json"
    payload = os.environ[REPORT_ENV]
    path.write_text(payload + "\n")
    try:
        report = json.loads(payload)
        age = time.time_ns() - report["checked_unix_ns"]
        expected_device = f"{config['user']}@{config['host']}:{config['port']}"
        valid = (
            report["status"] == "passed"
            and report["device"] == expected_device
            and report["scope"] == "Termux-visible processes"
            and report["cpu_limit_pct"] == float(CPU_LIMIT)
            and 0 <= report["background_peak_cpu_pct"] <= float(CPU_LIMIT)
            and report["locked_service_pid"] > 0
            and not report["residual_tr"]
            and len(report["samples"]) == INTERVALS
            and -5_000_000_000 <= age <= 60_000_000_000
        )
    except (ValueError, TypeError, KeyError) as exc:
        raise PhonePreflightError(f"手机 Pre-flight 控制端记录无效：{path}") from exc
    if not valid:
        raise PhonePreflightError(f"手机 Pre-flight 控制端记录未通过或已过期：{path}")
