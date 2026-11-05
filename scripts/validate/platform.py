#!/usr/bin/env python3
"""串行验证新平台的响应、历史、状态迁移与真实图像；不作性能标定。"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.environment import default_device
from a2o.runtime.runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=default_device())
    parser.add_argument('--stage', choices=('all', 'text', 'vision'), default='all')
    args = parser.parse_args()
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d-%H%M%S')
    out = ROOT / 'results' / (stamp + '-platform-validation')
    out.mkdir(parents=True)
    adb = ['adb', '-s', args.device]
    report = {'device': args.device, 'stage': args.stage, 'passed': False, 'runs': [],
              'timing_valid_for_performance': False,
              'scope': '共享运行时迁移验收；保存的真机截图；不执行生成动作；旧标定不用于性能结论'}

    def save():
        (out / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')

    save()
    try:
        report['identity'] = subprocess.check_output(adb + ['shell',
            'getprop ro.product.model; getprop ro.soc.model; getprop ro.build.fingerprint; getprop ro.serialno'], text=True)
        processes = subprocess.check_output(adb + ['shell', 'ps -A -o PID,NAME,ARGS'], text=True)
        (out / 'processes-before.txt').write_text(processes)
        if any('llama-server' in line or '/agent ' in line or 'agent_runtime' in line for line in processes.splitlines()):
            raise RuntimeError('发现模型进程，请先结束已有实验；不争抢设备')
        jobs = []
        if args.stage in ('all', 'text'):
            jobs += [('cpu', 'ready', False)]
            jobs += [(mode, 'history', mode != 'hybrid') for mode in ('cpu', 'npu', 'hybrid')]
        if args.stage in ('all', 'vision'):
            jobs += [(mode, 'screenshot-history', False) for mode in ('cpu', 'npu', 'hybrid')]
        for mode, scenario, roundtrip in jobs:
            name = mode + '-' + scenario
            config = {'mode': mode, 'kv_policy': 'none', 'pre_move': False,
                      'cpu_repack': False, 'a2o_ms': 0, 'format': 'chat',
                      'keep_history': 48, 'batch_size': 128, 'context_size': 2048,
                      'htp_queue': 1, 'htp_opbatch': 1, 'roundtrip': roundtrip,
                      'vision_backend': 'cpu', 'vision_flash': 'off',
                      'timeout_s': 300, 'timing_valid_for_performance': False}
            path = out / (name + '.json')
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n')
            row = {'condition': name, 'config': str(path.relative_to(ROOT)), 'passed': False}
            report['runs'].append(row); save()
            print('START ' + name, flush=True)
            result = run(path, ROOT / 'scenarios' / (scenario + '.json'), args.device, 'migration-' + name)
            row.update(result_directory=result['result_directory'],
                       response_check=result['response_check'], passed=result['response_check']['passed'])
            save()
            if not row['passed']:
                raise RuntimeError('响应验收失败：' + name)
        report['passed'] = all(row['passed'] for row in report['runs'])
    except Exception as exc:
        report['failure'] = str(exc)
        raise
    finally:
        save()
        print('验证记录：' + str(out), flush=True)


if __name__ == '__main__':
    main()
