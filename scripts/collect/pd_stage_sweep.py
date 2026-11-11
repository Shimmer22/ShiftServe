#!/usr/bin/env python3
"""独立于聊天运行的 CPU/HTP P/D 扫描入口。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.profiling.pd_sweep import main
if __name__ == '__main__':
    main()
