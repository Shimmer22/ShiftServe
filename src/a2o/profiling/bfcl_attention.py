"""BFCL 注意力试跑公共工具：原生 XML 调用解析与原子 JSON 写入。"""
import ast
import json
import re
from pathlib import Path


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')
    temp.replace(path)


def native_calls(content, docs, parse_calls):
    blocks = re.findall(r'<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>', content, re.S)
    if '<tool_call>' in content and not blocks:
        raise ValueError('malformed native tool call')
    calls = []
    for name, body in blocks:
        name = name.strip()
        if name not in docs:
            raise ValueError('unknown function: '+name)
        kwargs = {}
        for parameter, raw in re.findall(r'<parameter=([^>]+)>(.*?)</parameter>', body, re.S):
            parameter, raw = parameter.strip(), raw.strip()
            spec = docs[name]['parameters']['properties'][parameter]
            kind = spec.get('type', 'string')
            if kind in ('string','str'):
                kwargs[parameter] = raw
            else:
                try:
                    kwargs[parameter] = json.loads(raw)
                except json.JSONDecodeError:
                    kwargs[parameter] = ast.literal_eval(raw)
        canonical = name+'('+', '.join(k+'='+repr(v) for k,v in kwargs.items())+')'
        calls.extend(parse_calls(canonical, set(docs)))
    return calls
