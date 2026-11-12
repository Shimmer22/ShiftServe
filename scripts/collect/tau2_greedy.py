#!/usr/bin/env python3
"""always-prepare + 贪心继续/放弃的手机接续实验。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from a2o.profiling.tau2_greedy import main

if __name__=='__main__':
    main()
