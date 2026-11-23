"""Unit tests for AndroidControl worker parsing and evaluation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/collect"))

from host_androidcontrol_worker import eval_step, gold_action_json


def test_eval_step_click_and_others():
    click_step = {
        "action": {"action_type": "click", "x": 116, "y": 219},
        "hit_ids": [3, 1],
    }
    assert eval_step(click_step, '{"action_type":"click","id":3}')["strict_valid"]
    r_parent = eval_step(click_step, '{"action_type":"click","id":1}')
    assert r_parent["valid"] and not r_parent["strict_valid"]
    assert not eval_step(click_step, '{"action_type":"click","id":9}')["valid"]
    assert gold_action_json(click_step) == '{"action_type": "click", "id": 3}'

    inp_step = {"action": {"action_type": "input_text", "text": "Hello World"}, "hit_ids": []}
    assert eval_step(inp_step, '{"action_type":"input_text","text":"hello world"}')["valid"]
    assert not eval_step(inp_step, '{"action_type":"input_text","text":"wrong"}')["valid"]

    app_step = {"action": {"action_type": "open_app", "app_name": "Etsy"}, "hit_ids": []}
    assert eval_step(app_step, '{"action_type":"open_app","app_name":"etsy"}')["valid"]

    scr_step = {"action": {"action_type": "scroll", "direction": "down"}, "hit_ids": []}
    assert eval_step(scr_step, '{"action_type":"scroll","direction":"down"}')["valid"]
