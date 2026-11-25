#!/usr/bin/env python3
"""在 wins 执行，验证手机服务正常 JSON 响应、原始 token 和跨轮记忆。"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.runtime.phone_preflight import require_http_phone_preflight
from a2o.workloads.toolsandbox import post, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    require_http_phone_preflight(ROOT, cfg["endpoint"], args.output)
    messages = [{'role': 'system', 'content': 'Respond with JSON only. Follow the user instructions.'}]
    checks = []
    for index, question in enumerate([
        'Remember the session code CEDAR5837. Reply exactly {"status":"ready"}.',
        'What session code did I give you? Reply as {"session_code":"..."}.']):
        messages.append({'role': 'user', 'content': question})
        body = dict(model=cfg.get('model_id', 'qwen35-9b-q4_0'), messages=messages, temperature=0,
                    max_tokens=128, return_tokens=True, cache_prompt=True,
                    chat_template_kwargs={'enable_thinking': False})
        save(args.output / f'{index}.request.json', body)
        started = time.monotonic()
        response = post(cfg['endpoint'], '/v1/chat/completions', body, cfg['request_timeout_s'])
        save(args.output / f'{index}.response.json', response)
        message = response['choices'][0]['message']
        value = json.loads(message['content'])
        expected = {'status': 'ready'} if index == 0 else {'session_code': 'CEDAR5837'}
        native = response.get('__verbose', {})
        passed = (value == expected and response['choices'][0]['finish_reason'] == 'stop'
                  and bool(native.get('tokens')) and not native.get('truncated'))
        checks.append(dict(turn=index, passed=passed, value=value,
                           seconds=time.monotonic()-started, usage=response.get('usage'),
                           timings=response.get('timings')))
        save(args.output / 'validation.json', dict(passed=all(r['passed'] for r in checks), checks=checks))
        if not passed:
            raise RuntimeError('手机 JSON/记忆响应校验失败；原始输出已保存')
        messages.append({'role': 'assistant', 'content': message['content']})
    print(json.dumps(dict(passed=True, checks=checks), indent=2))


if __name__ == '__main__':
    main()
