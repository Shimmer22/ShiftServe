#!/usr/bin/env python3
"""只读 Android 时钟对照，保存原始输出，不改变手机设置。"""
import argparse
import json
from pathlib import Path
import subprocess
import time
ROOT=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
ndk=sorted((Path.home()/'Library/Android/sdk/ndk').iterdir(),key=lambda p:tuple(map(int,p.name.split('.'))))[-1]
cxx=next((ndk/'toolchains/llvm/prebuilt').glob('*/bin/aarch64-linux-android28-clang++'))
binary=ROOT/'build/clock-probe';subprocess.run([str(cxx),'-std=c++17',str(ROOT/'src/a2o/native/profiling/clock_probe.cpp'),'-o',str(binary)],check=True)
adb=['adb','-s',a.device];remote='/data/local/tmp/a2o_clock_probe_'+str(time.time_ns())
a.output.mkdir(parents=True,exist_ok=True)
try:
 subprocess.run(adb+['push',str(binary),remote],check=True,capture_output=True)
 subprocess.run(adb+['shell','chmod','755',remote],check=True,capture_output=True)
 start=time.monotonic();r=subprocess.run(adb+['shell',remote],check=True,capture_output=True,text=True,timeout=30);elapsed=time.monotonic()-start
 (a.output/'device-clocks.jsonl').write_text(r.stdout)
 v=[json.loads(x) for x in r.stdout.splitlines()]
 (a.output/'clock-summary.json').write_text(json.dumps(dict(host_elapsed_s=elapsed,device_delta_s={k:v[1][k]-v[0][k] for k in v[0]}),indent=2)+'\n')
 print((a.output/'clock-summary.json').read_text())
finally:
 subprocess.run(adb+['shell','rm','-f',remote],check=True,capture_output=True)
