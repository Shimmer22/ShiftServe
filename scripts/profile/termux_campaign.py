#!/usr/bin/env python3
"""串行重跑原报告的854条记录，不生成新报告。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.profiling.termux_campaign import main

if __name__ == '__main__':
    main()
