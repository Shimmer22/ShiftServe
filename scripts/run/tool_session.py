#!/usr/bin/env python3
"""实际只读 Tool 驱动的持久手机会话；不自动执行模型动作。"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.environment import default_device
from a2o.runtime.tool_session import run_tool_session

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--scenario', type=Path, required=True)
    p.add_argument('--device', default=default_device())
    p.add_argument('--control', action='store_true')
    p.add_argument('--mechanism-test', action='store_true')
    a = p.parse_args()
    result = run_tool_session(a.config, a.scenario, a.device, control=a.control, mechanism=a.mechanism_test)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['passed']:
        raise SystemExit(2)
