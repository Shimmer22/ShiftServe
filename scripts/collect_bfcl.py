#!/usr/bin/env python3
"""BFCL 文本多轮采集入口；只执行官方内存沙箱的方法，不执行宿主动作。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.bfcl_collect import main

if __name__ == '__main__':
    main()
