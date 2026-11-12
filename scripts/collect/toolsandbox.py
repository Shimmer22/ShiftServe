#!/usr/bin/env python3
"""ToolSandbox 手机服务端测评入口，在 wins 的专用环境执行。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from a2o.workloads.toolsandbox import main
if __name__ == '__main__':
    main()
