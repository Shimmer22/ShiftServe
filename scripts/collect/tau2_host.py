#!/usr/bin/env python3
"""固定官方 τ²-bench 的主机并发采集入口。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.workloads.tau2 import main

if __name__ == '__main__':
    main()
