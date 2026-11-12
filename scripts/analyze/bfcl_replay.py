"""运行 vLLM 轨迹注意力采样。"""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from a2o.profiling.bfcl_replay import main

if __name__=='__main__':
    main()
