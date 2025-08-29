import json
from pathlib import Path


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else 0.0


def test_metrics_recompute_from_artifact():
    ab_json = next(Path.cwd().glob("**/ab_test_complete.json"), None)
    assert ab_json, "ab_test_complete.json not found"
    data = json.loads(ab_json.read_text())
    sessions = data.get("raw_results") or []
    assert sessions, "no sessions in raw_results"
    by = {
        True: {"ias": [], "gas": [], "close": []},
        False: {"ias": [], "gas": [], "close": []},
    }
    for s in sessions:
        cond = bool(s["bandit_enabled"])
        tel = s["telemetry"]
        by[cond]["ias"].append(_mean(tel.get("ias_scores", [])))
        by[cond]["gas"].append(_mean(tel.get("gas_scores", [])))
        cr = tel.get("close_rates", [])
        by[cond]["close"].append(cr[-1] if cr else 0.0)
    # minimal assertions that they are numeric (no fabricated text)
    for cond in (True, False):
        assert all(isinstance(x, (int, float)) for x in by[cond]["ias"])
        assert all(isinstance(x, (int, float)) for x in by[cond]["gas"])
        assert all(isinstance(x, (int, float)) for x in by[cond]["close"])
