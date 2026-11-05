"""旧回放汇总不能误读新 Tool 会话；所有输出限于临时目录。"""
import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from fixture_data import replay_fixtures

ROOT = Path(__file__).resolve().parents[1]


class AnalysisScopeTest(unittest.TestCase):
    def test_legacy_reports_exclude_tool_session_without_losing_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scenarios').mkdir()
            shutil.copy(ROOT / 'scenarios/history.json', root / 'scenarios/history.json')
            replay_fixtures(root)
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
