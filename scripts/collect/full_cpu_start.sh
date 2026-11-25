#!/bin/sh
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
PYTHONPATH="$root/src${PYTHONPATH:+:$PYTHONPATH}" exec python3 - "$root" <<'PY'
import os
from pathlib import Path
import shlex
import sys

from a2o.runtime.phone_preflight import docker_preflight_args

root = Path(sys.argv[1])
gate = docker_preflight_args(root, root / "results/20261107-toolsandbox-full/preflight")
project = "/workspace/projects/a2o-gap-profiling"
command = ["docker", "exec", *gate, "a2o-toolsandbox-py311",
           "/workspace/tools/a2o-toolsandbox-py311-env/bin/python",
           f"{project}/scripts/collect/toolsandbox.py", "--config",
           f"{project}/configs/toolsandbox-full-cpu.json", "--output",
           "/workspace/outputs/a2o-gap-profiling/20261107-full-cpu"]
os.execvp("ssh", ["ssh", "wins", shlex.join(command)])
PY
