"""旧回放汇总不能误读新 Tool 会话；所有输出限于临时目录。"""
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class AnalysisScopeTest(unittest.TestCase):
    def test_legacy_reports_exclude_tool_session_without_losing_replay(self):
        single = multi = tool = None
        for manifest in sorted((ROOT / 'results').glob('*/manifest.json')):
            result_path = manifest.parent / 'result.json'
            if not result_path.exists():
                continue
            m, r = json.loads(manifest.read_text()), json.loads(result_path.read_text())
            if 'response_check' in r and not r.get('session') and 'turns' not in r and 'scenario' in m:
                single = single or manifest.parent
            if (r.get('session') and 'response_check' in r
                    and not any(t.get('compared') for t in r['turns'])):
                multi = multi or manifest.parent
            if 'turns' in r and 'response_check' not in r and 'scenario' in m:
                tool = tool or manifest.parent
        self.assertTrue(single and multi and tool, 'recorded fixtures required')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scenarios').mkdir()
            shutil.copy(ROOT / 'scenarios/history.json', root / 'scenarios/history.json')
            for name, folder in [('single', single), ('multi', multi), ('tool', tool)]:
                target = root / 'results' / name
                target.mkdir(parents=True)
                for file in ['manifest.json', 'result.json']:
                    shutil.copy(folder / file, target / file)
            for script in ['results', 'multiturn']:
                namespace = runpy.run_path(str(ROOT / f'scripts/analyze/{script}.py'))
                entry = namespace['main']
                entry.__globals__['ROOT'] = root
                with patch.object(sys, 'argv', [script]):
                    entry()
            one = json.loads((root / 'results/validation-summary.json').read_text())
            many = json.loads((root / 'results/multiturn-summary.json').read_text())
            self.assertEqual([r['run'] for r in one['runs']], ['results/single'])
            self.assertEqual([r['run'] for r in many], ['results/multi'])
