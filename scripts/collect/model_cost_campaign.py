#!/usr/bin/env python3
"""新模型四路径准入与CPU/HTP关键成本表，支持独立后台采集。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.profiling.model_cost_campaign import main

if __name__ == '__main__':
    main()
