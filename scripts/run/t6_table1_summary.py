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
TABLE1_SEVEN_ARMS = {
    'full_cpu': 'Full-CPU (Independent)',
    'shared_cc': 'Full-CPU (Shared)',
    'full_htp': 'Full-HTP',
    'shared_static_hc': 'Shared-Static-HC',
    'shared_m1': 'ShiftRouter',
    'shifttrim': 'ShiftTrim',
    'shiftserve_joint': 'ShiftServe',
}
TABLE1_EIGHT_ARMS = {
    'full_cpu': 'Full-CPU (Independent)',
    'shared_cc': 'Full-CPU (Shared)',
    'full_htp': 'Full-HTP',
    'dual_static_hc': 'Vanilla-Static-HC',
    'shared_static_hc': 'Shared-Static-HC',
    'shared_m1': 'ShiftRouter',
    'shifttrim': 'ShiftTrim',
    'shiftserve_joint': 'ShiftServe',
}


def write_table1(output, cases, rows, memory, arms=None):
    arm_labels = arms or TABLE1_ARMS
    by_arm = {arm: {row['id']: row for row in rows if row['arm'] == arm} for arm in arm_labels}
    table = []
    for arm, name in arm_labels.items():
        traces = by_arm[arm]
        ledger = memory[arm]
        row = dict(
            condition=name, completed_cases=len(traces), planned_cases=len(cases),
            warm_rounds=None, post_return_s=None, ledger_s=None, post_return_excl_ledger_s=None,
            speedup_vs_full_cpu=None, speedup_vs_static_hc=None,
            speedup_excl_ledger_vs_full_cpu=None, speedup_excl_ledger_vs_static_hc=None,
            **({'speedup_vs_vanilla_static_hc': None, 'speedup_excl_ledger_vs_vanilla_static_hc': None}
               if 'dual_static_hc' in arm_labels else {}),
            prep_wait_s=None, prefill_s=None,
            prior_compact_s=None, handoff_s=None, decode_s=None, decode_ms_per_token=None,
            remaining_s=None, remaining_excl_ledger_s=None, peak_pss_plus_dma_mib=ledger['peak_pss_plus_dma_mib'],
            peak_vmswap_mib=ledger['peak_vmswap_mib'],
            missing_pss_plus_dma_samples=ledger['missing_pss_plus_dma_samples'],
        )
        if set(traces) == set(cases):
            n = sum(trace['warm_rounds'] for trace in traces.values())
            total = math.fsum(trace['post_return_ms'] for trace in traces.values())
            ledger_total = math.fsum(trace.get('post_return_ledger_ms', 0.0) for trace in traces.values())
            net_total = total - ledger_total
            def phase(field):
                return math.fsum(trace['post_return_phases_ms'][field] for trace in traces.values())

            remaining_s = math.fsum(phase(p) for p in ('observation', 'restore', 'snapshot', 'other')) / (1000 * n)
            ledger_s = ledger_total / (1000 * n)
            row.update(warm_rounds=n, post_return_s=total / (1000 * n),
                       ledger_s=ledger_s,
                       post_return_excl_ledger_s=net_total / (1000 * n),
                       prep_wait_s=phase('prep_wait') / (1000 * n),
                       prefill_s=(phase('prefill_including_prior') - phase('prior')) / (1000 * n),
                       prior_compact_s=(phase('prior') + phase('compact')) / (1000 * n),
                       handoff_s=phase('handoff') / (1000 * n),
                       decode_s=phase('decode') / (1000 * n),
                       decode_ms_per_token=phase('decode') / sum(t['warm_decode_tokens'] for t in traces.values()),
                       remaining_s=remaining_s,
                       remaining_excl_ledger_s=remaining_s - ledger_s)
            for baseline, key, net_key in (
                ('full_cpu', 'speedup_vs_full_cpu', 'speedup_excl_ledger_vs_full_cpu'),
                ('shared_static_hc', 'speedup_vs_static_hc', 'speedup_excl_ledger_vs_static_hc'),
                *(([('dual_static_hc', 'speedup_vs_vanilla_static_hc', 'speedup_excl_ledger_vs_vanilla_static_hc')])
                  if 'dual_static_hc' in arm_labels else ()),
            ):
                if baseline in by_arm and set(by_arm[baseline]) == set(cases):
                    base_gross = math.fsum(by_arm[baseline][case]['post_return_ms'] for case in cases)
                    base_net = base_gross - math.fsum(by_arm[baseline][case].get('post_return_ledger_ms', 0.0) for case in cases)
                    row[key] = base_gross / total
                    row[net_key] = base_net / net_total
        table.append(row)
    pending = output / 'table1.tmp'
    with pending.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(table)
    pending.replace(output / 'table1.csv')
