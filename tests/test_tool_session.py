import http.server
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from a2o.scheduling.router_table import export_table
from a2o.tools.adapters import execute_tool
from a2o.validation.tool_session import validate_tool_config, validate_lifecycle


class ToolSessionTest(unittest.TestCase):
    def test_calibration_export_filters_failed_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'router.tsv'
            metadata = export_table(ROOT / 'results/20261103-213658-342299-four-path-structured-grid', output)
            self.assertGreater(metadata['rows'], 0)
            rows = [line.split('\t') for line in output.read_text().splitlines()[1:]]
            self.assertTrue(all(r[3] in {'CC', 'HC'} and int(r[-1]) == 3 for r in rows))
            # 导入后的表不包含已知未通过的 HH/CH，也不推断新 H/P。
            self.assertTrue(all(int(r[0]) in {1024, 4096} and int(r[1]) in {16, 64} for r in rows))
            bad = Path(tmp) / 'bad'; bad.mkdir()
            manifest = json.loads((ROOT / 'results/20261103-213658-342299-four-path-structured-grid/manifest.json').read_text())
            (bad / 'manifest.json').write_text(json.dumps(manifest))
            raw = json.loads((ROOT / 'results/20261103-213658-342299-four-path-structured-grid/H1024.jsonl').read_text().splitlines()[0])
            records = [{**raw, 'repetition': i, 'numerical_passed': i != 1} for i in range(3)]
            (bad / 'H1024.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
            rejected = export_table(bad, Path(tmp) / 'bad.tsv')
            self.assertEqual(rejected['rows'], 0)
            self.assertEqual(rejected['rejected'][0]['valid'], 2)

    def test_real_http_tool_executes_and_rejects_unknown_tools(self):
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200); self.end_headers(); self.wfile.write(b'{"result":"ok"}')
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever); thread.start()
        try:
            raw = execute_tool({'type': 'http_get', 'url': f'http://127.0.0.1:{server.server_port}/observation'}, 'unused')
            self.assertEqual(json.loads(raw)['result'], 'ok')
        finally:
            server.shutdown(); server.server_close(); thread.join()
        for tool in [{'type': 'shell', 'command': 'anything'}, {'type': 'http_get', 'url': 'file:///etc/passwd'}]:
            with self.assertRaises(ValueError):
                execute_tool(tool, 'unused')

    def test_production_mode_rejects_mechanism_fixtures(self):
        config = json.loads((ROOT / 'configs/tool-session-smoke.json').read_text())
        scenario = json.loads((ROOT / 'scenarios/tool-session-memory.json').read_text())
        with self.assertRaises(ValueError):
            validate_tool_config(config, scenario, False)
        validate_tool_config(config, scenario, True)

    def test_missing_or_reordered_publish_fails(self):
        names = ['tool_started', 'observation_ready', 'route_selected', 'prefill_started',
                 'action_complete', 'cpu_state_committed', 'response_published']
        events = [{'event': name, 'turn': 0} for name in names]
        events.append({'event': 'session_closed', 'turn': 1, 'started_tasks': 0, 'reaped_tasks': 0})
        turns = [{'committed_version': 1, 'authority_checked': True}]
        self.assertTrue(validate_lifecycle(events, turns, False)['passed'])
        self.assertFalse(validate_lifecycle(events[:-2] + events[-1:], turns, False)['passed'])
        events[0], events[-2] = events[-2], events[0]
        self.assertFalse(validate_lifecycle(events, turns, False)['passed'])
