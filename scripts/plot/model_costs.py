#!/usr/bin/env python3
"""验收原始成本采集并导出图表。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from a2o.profiling.cost_report import main

if __name__=='__main__':
    main()
