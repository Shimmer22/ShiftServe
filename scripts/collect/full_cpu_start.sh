#!/bin/sh
set -eu
exec ssh wins 'docker exec a2o-toolsandbox-py311 /workspace/tools/a2o-toolsandbox-py311-env/bin/python /workspace/projects/a2o-gap-profiling/scripts/collect/toolsandbox.py --config /workspace/projects/a2o-gap-profiling/configs/toolsandbox-full-cpu.json --output /workspace/outputs/a2o-gap-profiling/20261107-full-cpu'
