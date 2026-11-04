import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from a2o.runner import check_response, validate


class ResponseContract(unittest.TestCase):
    def test_memory_and_action_step_are_checked(self):
        scenario = {'check': {'type': 'json_action', 'action': 'wait',
                              'equals': {'memo': 'PINE-47', 'step': 2}}}
        self.assertTrue(check_response({'output': '{"action":"wait","memo":"PINE-47","step":2}'}, scenario)['passed'])
        for text in ['{"action":"wait","memo":"UNKNOWN","step":2}',
                     '{"action":"wait","memo":"PINE-47","step":1}']:
            self.assertFalse(check_response({'output': text}, scenario)['passed'])

    def test_eog_is_not_a_valid_response(self):
        self.assertFalse(check_response({'output': ''}, {'check': {'type': 'nonempty'}})['passed'])

    def test_valid_json_requires_requested_action(self):
        scenario = {'check': {'type': 'json_action', 'action': 'wait', 'keys': ['screen', 'action']}}
        for text in ['[]', '{}', '{"action":"tap","screen":"Settings"}',
                     'Here is the action: {"screen":"Settings","action":"wait"}']:
            self.assertFalse(check_response({'output': text}, scenario)['passed'])
        self.assertTrue(check_response({'output': '{"screen":"Settings","action":"wait"}'}, scenario)['passed'])

    def test_single_json_fence_is_parsed_and_reported(self):
        scenario = {'check': {'type': 'json_action', 'action': 'wait'}}
        text = '```json\n{"screen":"Settings","action":"wait"}\n```'
        result = check_response({'output': text}, scenario)
        self.assertTrue(result['passed'])
        self.assertFalse(result['strict_format_passed'])
        self.assertEqual(result['response_format'], 'json_code_fence')
        for bad in [text+'\nExplanation', text.replace('"wait"','"tap"'), '```json\ninvalid\n```']:
            self.assertFalse(check_response({'output': bad}, scenario)['passed'])

    def test_impossible_ablation_is_rejected(self):
        for mode in ['cpu', 'npu']:
            with self.assertRaises(ValueError):
                validate({'mode': mode, 'kv_policy': 'none', 'pre_move': True})

    def test_vision_preparation_requires_known_phase(self):
        config = {"mode": "hybrid", "kv_policy": "none", "pre_move": False,
                  "vision_backend": "htp", "vision_prepare": "a2o"}
        validate(config)
        with self.assertRaises(ValueError):
            validate({**config, "vision_prepare": "during_unknown_frame"})

    def test_connection_spelling_and_format_are_separate(self):
        scenario = {'check': {'type': 'one_of', 'values': ['WIFI', 'WI-FI'], 'strict_value': 'WIFI'}}
        result = check_response({'output': 'WI-FI'}, scenario)
        self.assertTrue(result['passed'])
        self.assertFalse(result['strict_format_passed'])
        self.assertFalse(check_response({'output': 'BLUETOOTH'}, scenario)['passed'])

    def test_screenshot_requires_correct_screen_not_just_json(self):
        scenario = {'check': {'type': 'json_action', 'action': 'wait',
                             'keys': ['screen', 'action'], 'screen_contains': ['developer', '开发者']}}
        for screen in ['Developer options', '开发者选项']:
            text = '{"screen":"' + screen + '","action":"wait"}'
            self.assertTrue(check_response({'output': text}, scenario)['passed'])
        result = check_response({'output': '{"screen":"Home","action":"wait"}'}, scenario)
        self.assertTrue(result['json_action_passed'])
        self.assertFalse(result['screen_passed'])
        self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
