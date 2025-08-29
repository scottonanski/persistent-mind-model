# tests/test_real_ab_flow.py
import json
import subprocess
import sys
from pathlib import Path


def run(cmd, cwd=None, env=None):
    r = subprocess.run(cmd, text=True, capture_output=True, cwd=cwd, env=env)
    assert r.returncode == 0, f"failed: {' '.join(map(str,cmd))}\nSTDERR:\n{r.stderr}"
    return r


def _runner_path():
    # EDIT THIS if your runner lives elsewhere
    p = Path("metrics/run_a_session.py")
    assert p.exists(), f"session runner not found at {p.resolve()}"
    return p


def _fresh_results(tmpdir: Path) -> Path:
    d = tmpdir / "ab_results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_runner_produces_real_logs(tmp_path: Path):
    results_dir = _fresh_results(tmp_path)
    runner = _runner_path()

    # run exactly one tiny real A/B test into an ISOLATED dir
    cmd = [
        sys.executable,
        "metrics/ab_test_bandit.py",
        "--sessions-per-condition",
        "1",
        "--turns-per-session",
        "5",
        "--runner",
        str(runner),
        "--results-dir",
        str(results_dir),
    ]
    run(cmd)

    ab_json = results_dir / "ab_test_complete.json"
    assert ab_json.exists(), f"missing {ab_json}"
    data = _load_json(ab_json)
    sessions = data.get("raw_results") or []
    assert len(sessions) >= 2, "need >=2 sessions (1 bandit, 1 baseline)"

    # HARD REQUIREMENTS: real executions only
    for s in sessions:
        assert s.get("returncode") == 0, f"session failed: {s.get('stderr')}"
        tel = s.get("telemetry") or {}
        for k in ("ias_scores", "gas_scores", "close_rates"):
            arr = tel.get(k, [])
            assert (
                isinstance(arr, list) and len(arr) > 0
            ), f"empty {k} in {s.get('session_id')}"


def test_sanity_metrics_recompute(tmp_path: Path):
    # Generate fresh artifacts for this test
    results_dir = _fresh_results(tmp_path)
    runner = _runner_path()

    # Run AB test to generate artifacts
    cmd = [
        sys.executable,
        "metrics/ab_test_bandit.py",
        "--sessions-per-condition",
        "1",
        "--turns-per-session",
        "5",
        "--runner",
        str(runner),
        "--results-dir",
        str(results_dir),
    ]
    run(cmd)

    ab_json = results_dir / "ab_test_complete.json"
    data = _load_json(ab_json)
    sessions = data.get("raw_results") or []
    assert sessions, "no sessions in raw_results"

    def _mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return sum(xs) / len(xs) if xs else 0.0

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

    # Numeric & finite checks
    for cond in (True, False):
        for name in ("ias", "gas", "close"):
            vals = by[cond][name]
            assert vals and all(
                isinstance(v, (int, float)) for v in vals
            ), f"bad {name} for cond={cond}"


def test_analyzers_do_not_import_sim():
    import importlib
    import inspect

    forbidden = "bandit_reward_reshaping"
    for m in ("metrics.pmm_sanity_metrics", "metrics.quick_phase3b_test"):
        mod = importlib.import_module(m)
        src = inspect.getsource(mod)
        assert forbidden not in src, f"{m} imports {forbidden}"
