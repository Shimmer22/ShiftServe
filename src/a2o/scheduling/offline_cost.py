"""复用现有 Router TSV：离线候选覆盖不受在线采样稀疏影响。"""
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class OfflineCost:
    history_tokens: int
    new_tokens: int
    decode_tokens: int
    path: str
    future_ms: float
    entry_ms: float


class OfflineCostTable:
    def __init__(self, rows, *, signature, mode='hybrid', source='offline'):
        self.signature, self.mode, self.source = signature, mode, source
        self.rows = {(r.history_tokens, r.new_tokens, r.decode_tokens, r.path): r for r in rows}

    def lookup(self, key):
        if key.signature != self.signature or key.mode != self.mode:
            return None
        return self.rows.get((key.history_tokens, key.new_tokens, key.decode_tokens, key.path))

    @classmethod
    def load(cls, path, *, signature, mode='hybrid', allow_fixture=False):
        lines = Path(path).read_text().splitlines()
        header = lines[0].split() if lines else []
        if len(header) != 6 or header[0] != 'a2o-router-v1':
            raise ValueError('invalid router table header')
        if header[1] == 'mechanism-fixture' and not allow_fixture:
            raise ValueError('synthetic costs require explicit fixture permission')
        rows, seen = [], set()
        for line in lines[1:]:
            if not line.strip() or line.startswith('#'):
                continue
            h, p, d, route, future, entry, samples = line.split()
            row = OfflineCost(int(h), int(p), int(d), route, float(future), float(entry))
            key = (row.history_tokens, row.new_tokens, row.decode_tokens, route)
            if (key in seen or row.history_tokens < 0 or row.new_tokens <= 0
                    or row.decode_tokens <= 0 or route not in {'CC', 'HC'}
                    or int(samples) < (0 if header[1] == 'mechanism-fixture' else 3)
                    or any(not math.isfinite(v) or v < 0 for v in (row.future_ms, row.entry_ms))):
                raise ValueError('invalid or duplicate router cost row')
            seen.add(key)
            rows.append(row)
        result = cls(rows, signature=signature, mode=mode, source=header[1])
        result.context, result.batch, result.queue, result.opbatch = map(int, header[2:])
        return result
