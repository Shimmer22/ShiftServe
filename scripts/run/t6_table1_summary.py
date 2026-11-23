"""按配对轨迹汇总 Table 1，不用部分结果填论文主表。"""

import csv
import math

TABLE1_ARMS = {
    'full_cpu': 'Full-CPU',
    'full_htp': 'Full-HTP',
    'shared_static_hc': 'Shared-Static-HC',
    'shared_m1': 'ShiftRouter',
    'shifttrim': 'ShiftTrim',
    'shiftserve_joint': 'ShiftServe',
}


def write_table1(output, cases, rows, memory):
    by_arm = {arm: {row['id']: row for row in rows if row['arm'] == arm} for arm in TABLE1_ARMS}
    table = []
    for arm, name in TABLE1_ARMS.items():
        traces = by_arm[arm]
        ledger = memory[arm]
        row = dict(
            condition=name, completed_cases=len(traces), planned_cases=len(cases),
            warm_rounds=None, post_return_s=None, speedup_vs_full_cpu=None,
            speedup_vs_static_hc=None, prep_wait_s=None, prefill_s=None,
            prior_compact_s=None, handoff_s=None, decode_s=None, decode_ms_per_token=None,
            remaining_s=None, peak_pss_plus_dma_mib=ledger['peak_pss_plus_dma_mib'],
            peak_vmswap_mib=ledger['peak_vmswap_mib'],
            missing_pss_plus_dma_samples=ledger['missing_pss_plus_dma_samples'],
        )
        if set(traces) == set(cases):
            n = sum(trace['warm_rounds'] for trace in traces.values())
            total = math.fsum(trace['post_return_ms'] for trace in traces.values())
            def phase(field):
                return math.fsum(trace['post_return_phases_ms'][field] for trace in traces.values())

            row.update(warm_rounds=n, post_return_s=total / (1000 * n),
                       prep_wait_s=phase('prep_wait') / (1000 * n),
                       prefill_s=(phase('prefill_including_prior') - phase('prior')) / (1000 * n),
                       prior_compact_s=(phase('prior') + phase('compact')) / (1000 * n),
                       handoff_s=phase('handoff') / (1000 * n),
                       decode_s=phase('decode') / (1000 * n),
                       decode_ms_per_token=phase('decode') / sum(t['warm_decode_tokens'] for t in traces.values()),
                       remaining_s=math.fsum(phase(p) for p in ('observation', 'restore', 'snapshot', 'other')) / (1000 * n))
            for baseline, key in (('full_cpu', 'speedup_vs_full_cpu'),
                                  ('shared_static_hc', 'speedup_vs_static_hc')):
                if set(by_arm[baseline]) == set(cases):
                    row[key] = math.fsum(by_arm[baseline][case]['post_return_ms'] for case in cases) / total
        table.append(row)
    pending = output / 'table1.tmp'
    with pending.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    pending.replace(output / 'table1.csv')
