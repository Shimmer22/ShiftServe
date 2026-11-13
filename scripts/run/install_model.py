#!/usr/bin/env python3
"""持锁校验并断点安装共享模型。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from a2o.runtime.model_install import main

if __name__ == "__main__":
    main()
