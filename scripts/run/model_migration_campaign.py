#!/usr/bin/env python3
"""串行执行模型安装、基础准入和关键成本采集。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from a2o.runtime.model_migration import main

if __name__ == "__main__":
    main()
