"""Unit tests for DroidTask worker prediction parser and evaluator."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/collect"))

from host_droidtask_worker import _eval_step, _load_tasks, _parse_prediction


def test_parse_with_think_block():
    raw = '<think>need to tap settings</think>\n{"Finished": "No", "id": 5, "action": "tap", "input_text": "N/A"}'
    obj, err = _parse_prediction(raw)
    assert err is None
    assert obj == {"Finished": "No", "id": 5, "action": "tap", "input_text": "N/A"}


def test_eval_step_variants():
    assert _eval_step([5, "null"], '{"Finished":"No","id":5,"action":"tap","input_text":"N/A"}')["valid"]
    assert _eval_step([5, "null"], '{"Finished":"No","id":5,"action":"click","input_text":"N/A"}')["valid"]
    assert not _eval_step([5, "null"], '{"Finished":"No","id":4,"action":"tap","input_text":"N/A"}')["valid"]
    assert _eval_step([2, "Adara"], '{"Finished":"No","id":2,"action":"input","input_text":"adara"}')["valid"]
    assert not _eval_step([2, "Adara"], '{"Finished":"No","id":2,"action":"tap","input_text":"adara"}')["valid"]
    assert _eval_step([2, "Adara"], '{"Finished":"No","id":2,"action":"tap","input_text":"adara"}')["id_valid"]
    assert _eval_step([-1, "null"], '{"Finished":"Yes","id":-1,"action":"N/A","input_text":"N/A"}')["valid"]


def test_load_tasks_staged():
    tasks = _load_tasks(ROOT / "scenarios/droidtask/profile_no_answer.json", 4000)
    assert len(tasks) >= 10
    assert tasks[0]["cum_chars"] >= tasks[-1]["cum_chars"] >= 4000
    tasks_all = _load_tasks(ROOT / "scenarios/droidtask/profile_no_answer.json", 0)
    assert len(tasks_all) == 158
