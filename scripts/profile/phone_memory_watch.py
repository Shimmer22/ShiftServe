#!/usr/bin/env python3
"""Sample a running phone campaign's memory and page faults over Termux SSH."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from a2o.runtime.device_transport import DeviceTransport

TURN_LOG = re.compile(r"^(\S+) turn (\d+): (submitting|cpu|htp)\b")
REMOTE_SAMPLE = r'''
while :; do
    owner=
    { IFS= read -r owner < "$root/model.lock/owner"; } 2>/dev/null
    case "$owner" in
        "$run_name"-*)
            pid=
            { IFS= read -r pid < "$root/$owner/pid"; } 2>/dev/null
            case "$pid" in
                ''|*[!0-9]*) echo idle ;;
                *) awk -v owner="$owner" -v pid="$pid" -v statfile="/proc/$pid/stat" '
                    FILENAME == statfile {
                        sub(/^.*\) /, "")
                        printf "%s\t%s\tstart_ticks=%s\tminflt=%s\tmajflt=%s\tuser_ticks=%s\tsystem_ticks=%s",
                               owner, pid, $20, $8, $10, $12, $13
                        seen=1
                        next
                    }
                    seen && $1 ~ /^(Vm(Size|HWM|RSS|Swap)|Rss(Anon|File|Shmem)|MemAvailable|Swap(Total|Free|Cached)):/ {
                        sub(/:$/, "", $1)
                        printf "\t%s_kib=%s", $1, $2
                    }
                    seen && $1 ~ /^(read_bytes|write_bytes):$/ {
                        sub(/:$/, "", $1)
                        printf "\t%s=%s", $1, $2
                    }
                    END { if (seen) print ""; else print "idle" }
                ' "/proc/$pid/stat" "/proc/$pid/status" /proc/meminfo "/proc/$pid/io" 2>/dev/null ;;
            esac ;;
        *) echo idle ;;
    esac
    sleep "$interval"
done
'''


def campaign_status(run: Path) -> str:
    return json.loads((run / "status.json").read_text())["status"]


def write_row(stream, row: dict) -> None:
    row.update(host_unix_ns=time.time_ns(), host_monotonic_ns=time.monotonic_ns())
    stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def watch(run: Path, interval: float, append: bool) -> None:
    t = DeviceTransport(ROOT, "ssh", config=ROOT / "configs/termux-ssh.json")
    command = t.shell_command(f"root={shlex.quote(t.remote_root)}; run_name={shlex.quote(run.name)}; "
                              f"interval={interval}; " + REMOTE_SAMPLE)
    with (run / "memory-samples.jsonl").open("a" if append else "x", buffering=1) as samples, \
            run.with_suffix(".log").open() as timeline:
        write_row(samples, {"event": "watch_start", "interval_s": interval})
        turn = None
        while campaign_status(run) == "running":
            with subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True) as reader:
                try:
                    for line in reader.stdout:
                        for event in timeline:
                            match = TURN_LOG.match(event)
                            if match:
                                condition, number, action = match.groups()
                                turn = {"condition": condition, "turn": int(number),
                                        "turn_status": "submitted" if action == "submitting" else "returned"}
                        if campaign_status(run) != "running":
                            break
                        parts = line.rstrip().split("\t")
                        if len(parts) < 3:
                            continue
                        owner, pid, *fields = parts
                        values = {name: int(value) for name, value in (field.split("=", 1) for field in fields)
                                  if value.isdigit()}
                        state = turn if turn and owner == f"{run.name}-{turn['condition']}" else {}
                        write_row(samples, {"event": "sample", "owner": owner, "pid": int(pid), **values, **state})
                finally:
                    if reader.poll() is None:
                        reader.terminate()
            if campaign_status(run) == "running":
                write_row(samples, {"event": "ssh_reconnect", "exit_code": reader.returncode})
                time.sleep(interval)
        write_row(samples, {"event": "watch_end", "campaign_status": campaign_status(run)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--append", action="store_true", help="Preserve earlier samples when resuming a campaign")
    args = parser.parse_args()
    if not 0.5 <= args.interval <= 60:
        parser.error("interval must be 0.5–60 seconds")
    watch(args.run.resolve(), args.interval, args.append)


if __name__ == "__main__":
    main()
