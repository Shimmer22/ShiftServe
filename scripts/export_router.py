#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.router_table import export_table

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--decode-evaluations', type=int, default=4)
    a = p.parse_args()
    print(json.dumps(export_table(a.run, a.output, a.decode_evaluations), ensure_ascii=False, indent=2))
