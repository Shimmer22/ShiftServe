#!/usr/bin/env python3
"""在手机测试页显式执行一次受限点击，验收真实观察反馈与历史记忆。"""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.environment import default_device
from a2o.runtime.gui_loop import run_gui_loop

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/single-loop.json')
    args = parser.parse_args()
    result = run_gui_loop(args.config, args.device)
    if not result['passed']: raise SystemExit(2)
