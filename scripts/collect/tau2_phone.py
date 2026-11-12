#!/usr/bin/env python3
"""τ² 冻结边界的 always-prepare 手机 pilot。"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from a2o.profiling.tau2_phone import main

if __name__=='__main__':
    main()
