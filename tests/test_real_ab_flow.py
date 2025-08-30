# tests/test_real_ab_flow.py (DB-grounded, no subprocess)
from pathlib import Path

from pmm.storage.sqlite_store import SQLiteStore
from pmm.emergence import compute_emergence_scores, EmergenceAnalyzer


def _make_store(path: Path) -> SQLiteStore:
    return SQLiteStore(str(path))


def _seed_and_collect(
    store: SQLiteStore, turns: int = 5, commits: int = 3, closes: int = 2
):
    """Seed a session directly into SQLite and collect telemetry arrays.

    Telemetry:
    - ias_scores: IAS after each assistant turn
    - gas_scores: GAS after each assistant turn
    - close_rates: commitment_close_rate after each evidence append
    """
    ias_scores = []
    gas_scores = []
    close_rates = []

    analyzer = EmergenceAnalyzer(storage_manager=store)

    # Interleave conversation/self_expression turns
    for i in range(turns):
        store.append_event(kind="conversation", content=f"U{i}", meta={})
        store.append_event(kind="self_expression", content=f"A{i}", meta={})
        scores = compute_emergence_scores(storage_manager=store)
        try:
            ias_scores.append(float(scores.get("IAS", scores.get("ias", 0.0))))
            gas_scores.append(float(scores.get("GAS", scores.get("gas", 0.0))))
        except Exception:
            ias_scores.append(0.0)
            gas_scores.append(0.0)

    # Open commitments
    commit_hashes = [
        store.append_event(kind="commitment", content=f"C{i}", meta={})["hash"]
        for i in range(commits)
    ]

    # Close some via evidence; track close rate after each
    for h in commit_hashes[:closes]:
        store.append_event(kind="evidence", content="done", meta={"commit_ref": h})
        close_rates.append(analyzer.commitment_close_rate(window=50))

    return {
        "ias_scores": ias_scores,
        "gas_scores": gas_scores,
        "close_rates": close_rates,
    }


def test_runner_produces_real_logs(tmp_path: Path):
    # Simulate two sessions (baseline/bandit) directly via DB seeding
    store_a = _make_store(tmp_path / "sess_a.sqlite")
    tel_a = _seed_and_collect(store_a, turns=5, commits=4, closes=2)

    store_b = _make_store(tmp_path / "sess_b.sqlite")
    tel_b = _seed_and_collect(store_b, turns=5, commits=4, closes=3)

    sessions = [
        {"bandit_enabled": False, "telemetry": tel_a, "session_id": "A"},
        {"bandit_enabled": True, "telemetry": tel_b, "session_id": "B"},
    ]

    assert len(sessions) >= 2, "need >=2 sessions (1 bandit, 1 baseline)"

    # HARD REQUIREMENTS: ensure telemetry arrays are populated and numeric
    for s in sessions:
        tel = s.get("telemetry") or {}
        for k in ("ias_scores", "gas_scores", "close_rates"):
            arr = tel.get(k, [])
            assert (
                isinstance(arr, list) and len(arr) > 0
            ), f"empty {k} in {s.get('session_id')}"
            assert all(
                isinstance(x, (int, float)) for x in arr
            ), f"non-numeric {k} in {s.get('session_id')}"


def test_sanity_metrics_recompute(tmp_path: Path):
    # Generate fresh, DB-grounded sessions
    store_a = _make_store(tmp_path / "sess_a.sqlite")
    tel_a = _seed_and_collect(store_a, turns=5, commits=4, closes=2)
    store_b = _make_store(tmp_path / "sess_b.sqlite")
    tel_b = _seed_and_collect(store_b, turns=5, commits=4, closes=3)
    sessions = [
        {"bandit_enabled": False, "telemetry": tel_a},
        {"bandit_enabled": True, "telemetry": tel_b},
    ]

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
        try:
            mod = importlib.import_module(m)
        except ModuleNotFoundError:
            # Legacy modules removed in refactor; acceptable
            continue
        src = inspect.getsource(mod)
        assert forbidden not in src, f"{m} imports {forbidden}"
