#!/usr/bin/env python3
"""Parse llama_state_seq_get_data blob → per-token K features + segment labels.

Qwen2VL M-RoPE: magic(0xaf143cd8)+seq_id, then n_stream, cell_count,
per-cell meta (pos, n_seq_id, cell_ext{x,y}, seq_ids...), then K/V rows.
"""
from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_BF16 = 30
IO_MAGIC = 0xAF143CD8


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.off = 0

    def remaining(self) -> int:
        return len(self.data) - self.off

    def read(self, n: int) -> bytes:
        if n < 0 or self.off + n > len(self.data):
            raise EOFError(f"need {n} bytes at {self.off}, have {self.remaining()}")
        b = self.data[self.off : self.off + n]
        self.off += n
        return b

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]


def k_row_to_f32(raw: bytes, k_type: int, n_embd: int) -> np.ndarray:
    if k_type == GGML_TYPE_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if k_type == GGML_TYPE_F32:
        return np.frombuffer(raw, dtype=np.float32).copy()
    if k_type == GGML_TYPE_BF16:
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).copy()
    u8 = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
    out = np.zeros(n_embd, dtype=np.float32)
    n = min(u8.size, n_embd)
    out[:n] = u8[:n]
    return out


def project_feat(x: np.ndarray, feat_dim: int, seed: int = 0) -> np.ndarray:
    """Deterministic random projection to feat_dim (for policy compatibility)."""
    if x.shape[-1] == feat_dim:
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = np.clip(x, -1e3, 1e3)
    rng = np.random.RandomState(seed)
    w = rng.normal(0.0, 1.0 / np.sqrt(max(x.shape[-1], 1)), size=(x.shape[-1], feat_dim)).astype(np.float32)
    y = np.matmul(x, w, dtype=np.float32)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    n = np.linalg.norm(y, axis=-1, keepdims=True) + 1e-8
    return (y / n).astype(np.float32)


@dataclass
class ParseResult:
    n_cells: int
    n_layer: int
    k_type: int
    k_size_row: int
    v_trans: int
    bytes_k: int
    bytes_v: int
    positions: list
    notes: list


def parse_seq_blob(
    path: Path,
    n_pos_per_embd: int = 4,
    feat_dim: Optional[int] = 64,
    cell_ext_bytes: int = 8,
    mean_layers: bool = True,
) -> tuple[ParseResult, np.ndarray]:
    data = Path(path).read_bytes()
    r = Reader(data)
    notes: list[str] = []

    magic = r.u32()
    if magic == IO_MAGIC:
        seq_id = r.i32()
        notes.append(f"io_magic_ok seq_id={seq_id}")
    else:
        r.off = 0
        notes.append(f"no_io_magic got={magic:#x}; treating as raw payload")

    n_stream = r.u32()
    if n_stream != 1:
        notes.append(f"n_stream={n_stream} (expected 1)")

    positions: list[int] = []
    per_layer_k: list[list[np.ndarray]] = []
    k_type_global = -1
    n_layer = 0
    v_trans = 0
    bytes_k = 0
    bytes_v = 0
    cell_count = 0
    k_size_row_g = 0

    for _s in range(n_stream):
        cell_count = r.u32()
        if cell_count == 0:
            continue
        for _ in range(cell_count):
            pos = r.i32()
            n_seq_id = r.u32()
            if n_pos_per_embd > 1:
                r.read(cell_ext_bytes)
            for __ in range(n_seq_id):
                r.i32()
            positions.append(pos)
        if n_pos_per_embd > 1:
            notes.append(f"cell_ext_bytes={cell_ext_bytes} n_pos_per_embd={n_pos_per_embd}")

        v_trans = r.u32()
        n_layer = r.u32()
        for li in range(n_layer):
            k_type = r.i32()
            k_size_row = r.u64()
            if k_type_global < 0:
                k_type_global = k_type
                k_size_row_g = k_size_row
            if k_type == GGML_TYPE_F16:
                n_embd = k_size_row // 2
            elif k_type == GGML_TYPE_F32:
                n_embd = k_size_row // 4
            elif k_type == GGML_TYPE_BF16:
                n_embd = k_size_row // 2
            else:
                n_embd = 256
            layer_rows = []
            for _ci in range(cell_count):
                raw = r.read(k_size_row)
                bytes_k += k_size_row
                layer_rows.append(k_row_to_f32(raw, k_type, n_embd))
            per_layer_k.append(layer_rows)

        if not v_trans:
            for _li in range(n_layer):
                _vt = r.i32()
                v_size_row = r.u64()
                chunk = cell_count * v_size_row
                r.read(chunk)
                bytes_v += chunk
        else:
            for _li in range(n_layer):
                _vt = r.i32()
                v_size_el = r.u32()
                n_embd_v = r.u32()
                chunk = n_embd_v * cell_count * v_size_el
                r.read(chunk)
                bytes_v += chunk

    if r.remaining() != 0:
        notes.append(f"trailing_bytes={r.remaining()}")

    # (n_cells, n_embd) — mean over layers
    if not per_layer_k:
        raise RuntimeError("no K layers parsed")
    stacked = np.stack(
        [np.stack(rows, axis=0) for rows in per_layer_k], axis=0
    )  # (n_layer, n_cells, n_embd)
    if mean_layers:
        feats = stacked.mean(axis=0)
    else:
        feats = stacked[0]
    notes.append(f"feat_shape={list(feats.shape)} mean_layers={mean_layers}")

    if feat_dim is not None and feats.shape[-1] != feat_dim:
        feats = project_feat(feats, feat_dim, seed=42)
        notes.append(f"projected_to_feat_dim={feat_dim}")

    pr = ParseResult(
        n_cells=cell_count,
        n_layer=n_layer,
        k_type=k_type_global,
        k_size_row=k_size_row_g,
        v_trans=v_trans,
        bytes_k=bytes_k,
        bytes_v=bytes_v,
        positions=positions,
        notes=notes,
    )
    return pr, feats


def assign_segments(positions: list[int], segments: list[dict]) -> list[str]:
    """Map absolute positions to segment names via [pos0,pos1) or [start,end)."""
    labels = ["unknown"] * len(positions)
    for i, pos in enumerate(positions):
        for seg in segments:
            a = int(seg.get("pos0", seg.get("start", -1)))
            b = int(seg.get("pos1", seg.get("end", -1)))
            name = seg.get("seg_id", seg.get("name", "unknown"))
            if a <= pos < b:
                labels[i] = name
                break
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--meta", required=True, help="segments.json from dump tool")
    ap.add_argument("--out-npz", required=True)
    ap.add_argument("--out-json", default="")
    ap.add_argument("--feat-dim", type=int, default=64)
    ap.add_argument("--n-pos-per-embd", type=int, default=4)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    segments = meta["segments"] if "segments" in meta else meta

    pr, feats = parse_seq_blob(
        Path(args.seq),
        n_pos_per_embd=args.n_pos_per_embd,
        feat_dim=args.feat_dim,
    )
    labels = assign_segments(pr.positions, segments)

    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Build EXP-008 load_segments_from_npz compatible payload
    seg_ids = []
    kinds = []
    arrays = {
        "feats": feats,
        "positions": np.asarray(pr.positions, dtype=np.int32),
        "labels": np.asarray(labels),
    }
    for seg in segments:
        sid = str(seg.get("seg_id", seg.get("name")))
        kind = str(seg.get("kind", sid.split("_")[0] if "_" in sid else sid))
        a = int(seg.get("pos0", seg.get("start", 0)))
        b = int(seg.get("pos1", seg.get("end", 0)))
        # positions are absolute 0..n-1 matching row index when dump is contiguous
        mask = (np.asarray(pr.positions) >= a) & (np.asarray(pr.positions) < b)
        feat_seg = feats[mask]
        if feat_seg.shape[0] == 0:
            continue
        seg_ids.append(sid)
        kinds.append(kind)
        arrays[f"features_{sid}"] = feat_seg.astype(np.float32)
        arrays[f"n_tokens_{sid}"] = np.asarray(feat_seg.shape[0], dtype=np.int32)
    arrays["seg_ids"] = np.asarray(seg_ids, dtype=object)
    arrays["kinds"] = np.asarray(kinds, dtype=object)
    np.savez_compressed(out, **arrays)

    summary = {
        "parse": asdict(pr),
        "feat_shape": list(feats.shape),
        "label_counts": {k: int(sum(1 for x in labels if x == k)) for k in sorted(set(labels))},
        "bytes_per_token_est": (pr.bytes_k + pr.bytes_v) // max(pr.n_cells, 1),
        "out_npz": str(out),
        "seg_ids": seg_ids,
        "kinds": kinds,
        "synthetic": False,
    }
    out_json = Path(args.out_json) if args.out_json else out.with_suffix(".json")
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
