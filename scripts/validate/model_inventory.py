#!/usr/bin/env python3
"""只读GGUF头与张量目录，核对模型结构及逐张量量化类型。"""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct


def inventory(path):
    with path.open('rb') as stream:
        def number(fmt):
            size = struct.calcsize('<' + fmt)
            return struct.unpack('<' + fmt, stream.read(size))[0]

        def string(keep=True):
            size = number('Q')
            if keep:
                return stream.read(size).decode('utf-8')
            stream.seek(size, 1)

        def value(kind, keep=True):
            formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
            if kind == 8:
                return string(keep)
            if kind == 9:
                element, length = number('I'), number('Q')
                for _ in range(length):
                    value(element, False)
                return {'elements': length, 'type': element}
            return number(formats[kind])

        assert stream.read(4) == b'GGUF'
        version = number('I')
        assert version in [2, 3]
        count, keys = number('Q'), number('Q')
        metadata = {}
        for _ in range(keys):
            key, kind = string(), number('I')
            selected = key.startswith(('general.', 'qwen3.', 'tokenizer.ggml.model'))
            item = value(kind, selected)
            if selected or key == 'tokenizer.ggml.tokens':
                metadata[key] = item
        types = {0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0',
                 10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K'}
        tensors = []
        for _ in range(count):
            name, dims = string(), number('I')
            shape = [number('Q') for _ in range(dims)]
            kind, offset = number('I'), number('Q')
            tensors.append(dict(name=name, shape=shape, type=types.get(kind, str(kind)), offset=offset))
        return dict(path=str(path.resolve()), gguf_version=version, tensor_count=count, metadata=metadata,
                    tensor_types=dict(Counter(t['type'] for t in tensors)),
                    output_head=[t for t in tensors if t['name'] == 'output.weight'], tensors=tensors)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.model)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ['tensor_count', 'tensor_types', 'output_head']}, indent=2))
