import gzip
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/collect"))

import host_androidcontrol_worker as ac_w
import host_droidtask_campaign as dt_c
from a2o.profiling.host_accuracy import run_metadata


def _varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _pb_bytes(fn: int, payload: bytes) -> bytes:
    return _varint((fn << 3) | 2) + _varint(len(payload)) + payload


def _pb_int_list(vals: list[int]) -> bytes:
    packed = b"".join(_varint(v) for v in vals)
    return _pb_bytes(3, _pb_bytes(1, packed))


def _pb_bytes_list(vals: list[bytes]) -> bytes:
    return _pb_bytes(1, b"".join(_pb_bytes(1, v) for v in vals))


def _write_fake_shard(path: Path, eps: list[tuple[int, str]]):
    tree_pb = b""
    action_pb = json.dumps({"action_type": "navigate_home"}).encode()
    with gzip.open(path, "wb") as g:
        for eid, goal in eps:
            entries = [
                _pb_bytes(1, _pb_bytes(1, b"episode_id") + _pb_bytes(2, _pb_int_list([eid]))),
                _pb_bytes(1, _pb_bytes(1, b"goal") + _pb_bytes(2, _pb_bytes_list([goal.encode()]))),
                _pb_bytes(1, _pb_bytes(1, b"actions") + _pb_bytes(2, _pb_bytes_list([action_pb] * 4))),
                _pb_bytes(1, _pb_bytes(1, b"accessibility_trees") + _pb_bytes(2, _pb_bytes_list([tree_pb] * 4))),
                _pb_bytes(1, _pb_bytes(1, b"step_instructions") + _pb_bytes(2, _pb_bytes_list([b"go home"] * 4))),
            ]
            example = _pb_bytes(1, b"".join(entries))
            g.write(struct.pack("<Q", len(example)) + b"\x00\x00\x00\x00" + example + b"\x00\x00\x00\x00")


def test_load_episodes_multi_shard_and_seeded_shuffle(tmp_path: Path):
    s1 = tmp_path / "s1.tfrecord.gz"
    s2 = tmp_path / "s2.tfrecord.gz"
    splits = tmp_path / "splits.json"
    _write_fake_shard(s1, [(10, "g10"), (2, "g2")])
    _write_fake_shard(s2, [(5, "g5"), (2, "g2_dup"), (8, "g8")])
    splits.write_text(json.dumps({"test": [2, 5, 8, 10]}))

    a = [e["episode_id"] for e in ac_w.load_episodes([s1, s2], splits, min_steps=4, sample_seed=42)]
    b = [e["episode_id"] for e in ac_w.load_episodes([s2, s1], splits, min_steps=4, sample_seed=42)]
    assert a == b
    assert sorted(a) == [2, 5, 8, 10]


def test_droidtask_build_summary_compaction_active(tmp_path: Path):
    cfg = {
        "name": "test",
        "conditions": [
            {"name": "full_kv", "kv_reduce": 0, "kv_keep_ratio": -1.0},
            {"name": "m2_prior_mid", "kv_reduce": 1, "kv_keep_ratio": 0.5},
        ],
    }
    runtime = {"binary": "/bin/true", "binary_sha256": "x", "build_id": "b", "kv_layer_stride": 4, "kv_min_evict_tokens": 256}
    meta = {c["name"]: run_metadata(dt_c.PROTOCOL_VERSION, cfg, c["name"], 2, [], runtime) for c in cfg["conditions"]}

    for cname, r0_compact, r0_acc, r1_acc in [("full_kv", 0, 1, 0), ("m2_prior_mid", 2, 1, 1)]:
        cdir = tmp_path / "conditions" / cname
        cdir.mkdir(parents=True)
        res = [
            {
                "id": "t_active", "total_turns": 4, "complete_turns": 4, "valid_turns": r0_acc * 4, "id_valid_turns": 4,
                "all_turns_valid": bool(r0_acc),
                "kv_stats": {"compact_turns": r0_compact, "prior_turns": r0_compact, "recent_turns": 0, "reuse_turns": 3,
                             "before_tokens": 1000, "logical_tokens": 2000, "kept_tokens": 1000 if r0_compact else 2000, "evicted_tokens": 1000 if r0_compact else 0},
            },
            {
                "id": "t_short", "total_turns": 2, "complete_turns": 2, "valid_turns": r1_acc * 2, "id_valid_turns": 2,
                "all_turns_valid": bool(r1_acc),
                "kv_stats": {"compact_turns": 0, "prior_turns": 0, "recent_turns": 0, "reuse_turns": 1,
                             "before_tokens": 500, "logical_tokens": 500, "kept_tokens": 500, "evicted_tokens": 0},
            },
        ]
        summary = {
            **meta[cname], "total_tasks": 2, "total_turns": 6, "complete_turns": 6,
            "valid_turns": r0_acc * 4 + r1_acc * 2, "id_valid_turns": 6,
            "turn_accuracy": (r0_acc * 4 + r1_acc * 2) / 6, "id_turn_accuracy": 1.0,
            "all_turns_valid_tasks": r0_acc + r1_acc, "results": res,
        }
        (cdir / "worker-summary.json").write_text(json.dumps(summary))

    dt_c.build_summary(tmp_path, cfg, meta)
    out = json.loads((tmp_path / "summary.json").read_text())
    assert out["compaction_active_task_count"] == 1
    assert out["compaction_active_task_ids"] == ["t_active"]
    assert out["conditions"][0]["total_tasks"] == 2
    assert out["compaction_active_conditions"][0]["total_tasks"] == 1
    assert out["compaction_active_conditions"][1]["evicted_tokens"] == 1000
