"""Unit tests for Stage 1 H/P/D analysis and phone prior admission simulation."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/analyze"))

from mobile_gui_hpd_and_freeze import quantile, simulate_phone_prior


def test_quantile_and_phone_prior():
    assert quantile([10, 20, 30, 40], 0.5) == 25.0
    case = {
        "pin": 240,
        "turns": [
            {"H": 0, "P": 710, "D": 14},
            {"H": 706, "P": 51, "D": 13},
            {"H": 757, "P": 1164, "D": 13},
            {"H": 1921, "P": 304, "D": 13},
            {"H": 2225, "P": 326, "D": 13},
        ],
    }
    hits50, fit50 = simulate_phone_prior(case, ratio=0.5)
    hits75, fit75 = simulate_phone_prior(case, ratio=0.75)
    assert fit50 and fit75
    assert hits50 == 3
    assert hits75 >= 1
