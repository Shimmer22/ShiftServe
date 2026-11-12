#!/usr/bin/env python3
"""汇总 Full 重复轮的质量与耗时波动；失败不从完成率中删除。"""
import argparse
import json
from pathlib import Path
import statistics


def distribution(values):
    values = [float(value) for value in values if value is not None]
    if not values:
        return dict(n=0, mean=None, stdev=None, cv=None, min=None, max=None)
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    return dict(n=len(values), mean=mean, stdev=sd,
                cv=sd / mean if sd is not None and mean != 0 else None,
                min=min(values), max=max(values))


def summarize(base):
    folders = [base] + sorted(base.glob('round-[0-9][0-9]'))
    records = []
    for index, folder in enumerate(folders):
        round_id = 1 if folder == base else int(folder.name.split('-')[1])
        for condition in ('cpu', 'htp', 'route'):
            summary_path = folder / condition / 'summary.json'
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            for result in summary['results']:
                evaluation = result.get('evaluation') or {}
                records.append(dict(round=round_id, condition=condition, name=result['name'],
                    trace_complete=result['trace_complete'], full_score=result['full_score'],
                    similarity=evaluation.get('similarity'),
                    e2e_ms=result.get('agent_e2e_ms') if result['trace_complete'] else None,
                    inference_calls=result['inference_calls'], tool_calls=result['tool_calls']))
    per_case = []
    for condition, name in sorted({(r['condition'], r['name']) for r in records}):
        rows = [r for r in records if r['condition'] == condition and r['name'] == name]
        per_case.append(dict(condition=condition, name=name, attempts=len(rows),
            complete=sum(r['trace_complete'] for r in rows),
            full_score=sum(r['full_score'] for r in rows),
            e2e_ms_completed_only=distribution([r['e2e_ms'] for r in rows]),
            similarity=distribution([r['similarity'] for r in rows]),
            inference_calls=distribution([r['inference_calls'] for r in rows]),
            tool_calls=distribution([r['tool_calls'] for r in rows])))
    per_round = []
    for round_id, condition in sorted({(r['round'], r['condition']) for r in records}):
        rows = [r for r in records if r['round'] == round_id and r['condition'] == condition]
        per_round.append(dict(round=round_id, condition=condition, attempts=len(rows),
            complete=sum(r['trace_complete'] for r in rows),
            failures=sum(not r['trace_complete'] for r in rows),
            full_score=sum(r['full_score'] for r in rows),
            similarity=distribution([r['similarity'] for r in rows])))
    return dict(scope='端到端总波动，含工具时间/UUID及轨迹变化；不能解释为纯硬件噪声',
        timing='仅完整轨迹有E2E；每题同时报告全部attempt/complete，禁止把成功子集当无损加速',
        dispersion='样本标准差；不足两轮时stdev/CV为空；三轮仅为初步估计',
        per_case=per_case, per_round=per_round, records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    args = parser.parse_args()
    summary = summarize(args.base)
    output = args.base / 'repeats'
    output.mkdir(exist_ok=True)
    temporary = output / 'noise-summary.tmp'
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(output / 'noise-summary.json')
    print(json.dumps(summary['per_round'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
