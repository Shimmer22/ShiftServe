"""真实 ToolSandbox 状态机 + 确定性 HTTP 夹具，检查 token/时间线关联。"""
import json
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.workloads.toolsandbox import run_case, common_prefix


class MetricsTest(unittest.TestCase):
    def test_prefix(self):
        self.assertEqual(common_prefix([1, 2, 3], [1, 2, 4]), 2)
        self.assertEqual(common_prefix([], [1]), 0)

    @unittest.skipUnless(importlib.util.find_spec('tool_sandbox'), '需在 wins ToolSandbox 专用环境执行')
    def test_official_environment(self):
        from tool_sandbox.common.tool_discovery import ToolBackend
        from tool_sandbox.scenarios.single_tool_call_scenarios import named_single_tool_call_scenarios
        scenario = named_single_tool_call_scenarios(ToolBackend.DEFAULT)['get_wifi']
        calls = []

        def template(body):
            return json.dumps(body['messages'], sort_keys=True)

        def fake_post(base, route, body, timeout=1800):
            if route == '/tokenize':
                return {'tokens': list(body['content'].encode())}
            if route == '/apply-template':
                return {'prompt': template(body)}
            self.assertEqual(route, '/v1/chat/completions')
            calls.append(body)
            message = {'role': 'assistant', 'content': 'Wifi is turned on'}
            finish = 'stop'
            if len(calls) == 1:
                finish = 'tool_calls'
                message = {'role': 'assistant', 'content': None, 'tool_calls': [
                    {'id': '0-opaque/server:id', 'type': 'function',
                     'function': {'name': 'get_wifi_status', 'arguments': '{}'}}]}
            return {'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'fixture',
                    'choices': [{'index': 0, 'message': message, 'finish_reason': finish}],
                    'usage': {'prompt_tokens': len(template(body).encode()), 'completion_tokens': 3,
                              'total_tokens': len(template(body).encode()) + 3},
                    '__verbose': {'prompt': template(body), 'tokens': [1, 2, 3],
                                  'stop_type': 'eos', 'truncated': False}}

        cfg = {'endpoint': 'fixture', 'seed': 1, 'max_tokens': 100,
               'context_size': 100000, 'request_timeout_s': 1}
        with tempfile.TemporaryDirectory() as temporary, patch('a2o.workloads.toolsandbox.post', fake_post):
            result = run_case('get_wifi', scenario, cfg, Path(temporary))
            self.assertTrue(result['trace_complete'], result['error'])
            self.assertEqual(result['inference_calls'], 2)
            self.assertEqual(result['tool_calls'], 1)
            metrics = json.loads(Path(temporary, 'get_wifi/metrics.json').read_text())
            tool = metrics['tools'][0]
            self.assertEqual(tool['tool'], 'get_wifi_status')
            self.assertEqual(tool['tool_id'], 'call_0_0')
            self.assertEqual(tool['P_body'], len(tool['result'].encode()))
            self.assertEqual(tool['D_next'], 3)
            self.assertEqual(tool['H'], metrics['inference'][0]['prompt_tokens'] + 3)
            self.assertGreaterEqual(tool['G_ms'], tool['execution_ms'])
            self.assertGreaterEqual(tool['issue_to_next_request_ms'], tool['G_ms'])
            self.assertIsNotNone(result['evaluation'])
            self.assertGreaterEqual(result['agent_e2e_ms'],
                                    sum(step['host_request_ms'] for step in metrics['inference']))
            self.assertGreaterEqual(tool['issue_to_response_ms'], tool['issue_to_next_request_ms'])
            self.assertGreaterEqual(tool['result_to_response_ms'], tool['result_to_next_request_ms'])


if __name__ == '__main__':
    unittest.main()
