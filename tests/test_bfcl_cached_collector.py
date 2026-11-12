import http.server
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]


class CachedCollectorTest(unittest.TestCase):
    def test_rotation_resume_and_rejected_response_preservation(self):
        calls = []
        state = {'malformed': False, 'version': 2, 'reset_once': False}

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def send(self, value):
                body = json.dumps(value).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self.send(dict(condition='route', context_size=8192, replay_cache_version=state['version'], build_id='test'))

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/a2o/costs':
                    self.send({})
                    return
                calls.append(body)
                if state['reset_once']:
                    state['reset_once'] = False
                    self.close_connection = True
                    return
                rows = [dict(condition=c, H=body['H'], P=body['P'], D=body['D']) for c in body['condition_order']]
                if state['malformed']:
                    rows[0]['P'] += 1
                self.send(dict(cache=dict(prefix_tokens=body['H']), results=rows))

        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                base = Path(tmp)
                request = dict(task='a', step='t00-s00', prompt_ids=[1, 2, 3], generated_ids=[4], H=2, P=1, D=1)
                (base / 'requests.json').write_text(json.dumps([request]))
                (base / 'costs.json').write_text('{}')
                def run(output):
                    return subprocess.run([sys.executable, str(ROOT / 'scripts/collect/bfcl_phone_cached.py'),
                         '--requests', str(base / 'requests.json'), '--cost-model', str(base / 'costs.json'),
                         '--output', str(base / output), '--rounds', '3',
                         '--endpoint', 'http://127.0.0.1:' + str(server.server_port)], capture_output=True, text=True)
                result = run('good')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([call['condition_order'] for call in calls], [
                    ['cpu', 'htp', 'route'], ['htp', 'route', 'cpu'], ['route', 'cpu', 'htp']])
                self.assertEqual(run('good').returncode, 0)
                self.assertEqual(len(calls), 3, 'resume must not recompute completed cells')
                state['reset_once'] = True
                result = run('retry')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(len(calls), 7, 'one lost response must retry only its frozen cell')
                self.assertTrue((base / 'retry/network-events.jsonl').exists())
                state['malformed'] = True
                self.assertNotEqual(run('bad').returncode, 0)
                self.assertTrue((base / 'bad/r00-b000.raw.json').exists())
                self.assertFalse((base / 'bad/r00-b000.json').exists())
                state['version'] = 1
                self.assertNotEqual(run('legacy').returncode, 0)
                self.assertEqual(len(calls), 8, 'legacy service must fail before replay')
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
