#!/usr/bin/env python3
"""离线验收已提交的完整扫描、原始归档、清理与拟合输入，无需连接手机。"""
import collections
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/20261111-prefill-p24-128-geometric-cpu6-t10'
EVIDENCE = OUT / 'evidence'


def main():
    points = list(range(24, 65)) + [66, 68, 72, 80, 96, 128]
    manifest = json.loads((EVIDENCE / 'phone-records-manifest.json').read_text())
    archive = EVIDENCE / manifest['archive']
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
    expected_files = {r['path']: r for r in manifest['files']}
    raw = {}
    fixture = json.loads((EVIDENCE / 'token-source.response.json').read_text())['choices'][0]
    with tarfile.open(archive, 'r:gz') as tar:
        assert set(tar.getnames()) == set(expected_files)
        for member in tar.getmembers():
            data = tar.extractfile(member).read()
            entry = expected_files[member.name]
            assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
            if '/measurements/r' not in member.name or '.raw.' in member.name:
                continue
            d = json.loads(data)
            request, response = d['request'], d['response']
            assert request['prefix_ids'] == fixture['prompt_token_ids'][:4096]
            p = len(request['input_ids'])
            assert request['input_ids'] == fixture['prompt_token_ids'][4096:4096+p]
            assert not request['generated_ids'] and not request['decode_points']
            assert response['mode'] == 'stage_sweep_v1'
            assert [r['backend'] for r in response['rows']] == request['backend_order']
            run = member.name.split('/')[0]
            for row in response['rows']:
                assert (row['H'], row['P'], row['D']) == (4096, p, 0)
                assert row['state_restore_verified'] and row['finite_logits']
                expected_threads = 6 if row['backend'] == 'cpu' else 4
                assert row['n_threads'] == row['n_threads_batch'] == expected_threads
                key = (run, d['round'], row['backend'], p)
                assert key not in raw
                raw[key] = row
    assert len(raw) == 940
    with (OUT / 'stage-times.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 940
    counts, rounds = collections.Counter(), collections.defaultdict(set)
    for row in rows:
        key = (row['source_run'], int(row['round']), row['backend'], int(row['P']))
        original = raw[key]
        for field in ['prefill_ms', 'restore_ms']:
            assert float(row[field]) == original[field]
        assert float(row['elapsed_ms']) == original['prefill_ms']
        pair = (row['backend'], int(row['P']))
        counts[pair] += 1;rounds[pair].add(int(row['round']))
    assert {p for backend, p in counts} == set(points)
    assert len(counts) == 94 and set(counts.values()) == {10}
    assert all(r == set(range(10)) for r in rounds.values())
    cleaned = OUT / 'figures-cleaned'
    with (cleaned / 'cleaned-stage-times.csv').open() as stream:
        kept = list(csv.DictReader(stream))
    with (cleaned / 'excluded-stage-times.csv').open() as stream:
        removed = list(csv.DictReader(stream))
    assert len(kept) == 931 and len(removed) == 9
    key_of = lambda r: (r['source_run'], r['round'], r['backend'], r['P'])
    assert not {key_of(r) for r in kept} & {key_of(r) for r in removed}
    assert {key_of(r) for r in rows} == {key_of(r) for r in kept + removed}
    for backend, p in counts:
        values = sorted(float(r['elapsed_ms'])/1000 for r in rows if r['backend'] == backend and int(r['P']) == p)
        def quantile(q):
            pos = (len(values)-1)*q;i = int(pos)
            return values[i] + (values[min(i+1, len(values)-1)]-values[i])*(pos-i)
        q1, q3 = quantile(.25), quantile(.75)
        threshold = q3 + 1.5*(q3-q1)
        assert sorted(float(r['elapsed_ms'])/1000 for r in kept if r['backend'] == backend and int(r['P']) == p) == [v for v in values if v <= threshold]
    for point in json.loads((cleaned / 'curve-data.json').read_text()):
        values = [float(r['elapsed_ms'])/1000 for r in kept if r['backend'] == point['backend'] and int(r['P']) == point['P']]
        assert len(values) == point['n']
        assert abs(statistics.mean(values)-point['mean_s']) < 1e-10
    print('通过：493份归档文件校验、470个原始请求、940条耗时、各点10次、CPU6/HTP4线程、9条上尾剔除及均值。')


if __name__ == '__main__':
    main()
