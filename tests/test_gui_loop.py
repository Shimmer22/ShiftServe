"""动作执行边界：错误响应不能变成手机点击。"""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.runtime.gui_loop import validate_tap


class ActionGuardTests(unittest.TestCase):
    def test_accepts_image_pixel_point_inside_button(self):
        self.assertEqual(validate_tap({'action': 'tap', 'x': 500, 'y': 500, 'memo': 'PINE-42'},
                                      (100, 400, 900, 1400), 1000, 2000, 'PINE-42'), (500, 500))

    def test_rejects_outside_edges_and_browser_chrome(self):
        for x, y in ((500, 50), (100, 500), (900, 500), (-1, 500), (1001, 500)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                validate_tap({'action': 'tap', 'x': x, 'y': y, 'memo': 'PINE-42'},
                             (100, 400, 900, 1400), 1000, 2000, 'PINE-42')

    def test_rejects_wrong_action_code_types_and_extra_commands(self):
        valid = {'action': 'tap', 'x': 500, 'y': 500, 'memo': 'PINE-42'}
        for change in ({'action': 'shell'}, {'memo': 'PINE-99'}, {'x': True}, {'y': 500.0}, {'command': 'input keyevent 3'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_tap({**valid, **change}, (100, 400, 900, 1400), 1000, 2000, 'PINE-42')
