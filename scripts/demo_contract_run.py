#!/usr/bin/env python3
"""
Drive a short session to exercise the Next: contract validator and probe endpoints.

It avoids external network calls by injecting a stub LLM into reflect_once.
"""
import os
import sys
import time
import threading
import subprocess
from typing import Optional

# Ensure repo root is on sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pmm.self_model_manager import SelfModelManager  # noqa: E402
from pmm.reflection import reflect_once  # noqa: E402


class StubLLM:
    def __init__(self, replies):
        self.replies = list(replies)

    def chat(self, system: Optional[str] = None, user: Optional[str] = None) -> str:
        if self.replies:
            return self.replies.pop(0)
        # Fallback benign reflection that should pass
        return (
            "I noticed my pattern of concise replies improving.\n"
            "Next: Create 2 concrete examples within 5 minutes (refs: ev1, ev2)."
        )


def start_probe_async(port: int = 8002):
    def _run():
        # Prefer project venv if present
        py = os.path.join(ROOT, "venv", "bin", "python")
        cmd = [
            py if os.path.exists(py) else sys.executable,
            "-m",
            "uvicorn",
            "pmm.api.probe:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    # wait briefly for server to boot
    # give server up to ~3s to bind
    for _ in range(30):
        try:
            import urllib.request

            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as _:
                break
        except Exception:
            time.sleep(0.1)


def main():
    # Integration env per checklist
    os.environ.setdefault("PMM_AUTONOMY_AUTOSTART", "1")
    os.environ.setdefault("PMM_AUTONOMY_INTERVAL", "2")
    os.environ.setdefault("PMM_METRICS_MIN_INTERVAL", "20")
    os.environ.setdefault("PMM_METRICS_ACTIVE_INTERVAL", "8")
    os.environ.setdefault("PMM_METRICS_MIN_EVENTS", "3")
    os.environ.setdefault("PMM_METRICS_STATE_MIN_EVENTS", "5")
    os.environ.setdefault("PMM_REQUIRE_EVIDENCE_ARTIFACT", "1")
    os.environ.setdefault("PMM_ENFORCE_NEXT_CONTRACT", "1")
    os.environ.setdefault("PMM_TELEMETRY", "1")

    mgr = SelfModelManager("persistent_self_model.json")

    # Seed some interaction events to exceed state gate (>=5)
    for i in range(6):
        try:
            mgr.sqlite_store.append_event(
                kind="prompt", content=f"user message {i}", meta={"i": i}
            )
            mgr.sqlite_store.append_event(
                kind="response",
                content=f"assistant reply {i}",
                meta={"i": i},
            )
        except Exception:
            pass

    # Prepare 2 valid and 2 invalid reflections
    valid_1 = (
        "Observation: my summaries lacked concrete checks; I’ll fix that.\n"
        "Next: Create 3 scene summaries within 10 minutes (refs: ev1, ev2)."
    )
    valid_2 = (
        "I’m tuning novelty to avoid redundancy while keeping identity.\n"
        "Next: I will lower S0 novelty by 0.05 and rerun 2 prompts (refs: ev3)."
    )
    invalid_no_meas = (
        "Thinking about adjustments to scene quality.\n" "Next: Think about scenes."
    )
    invalid_non_action = (
        "I might consider improving things.\n"
        "Next: Maybe consider improving things sometime."
    )

    stub = StubLLM([valid_1, invalid_no_meas, valid_2, invalid_non_action])

    # Run four reflections with stubbed outputs
    for _ in range(4):
        try:
            reflect_once(
                mgr, llm=stub, active_model_config={"provider": "stub", "name": "stub"}
            )
        except Exception:
            pass

    # Start probe server and query endpoints
    port = 8002
    start_probe_async(port)

    import urllib.request
    import json

    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}") as resp:
            return json.loads(resp.read().decode("utf-8"))

    out = {
        "health": get("/health"),
        "traits": get("/traits"),
        "contract": get("/reflection/contract"),
        "bandit": get("/bandit/reflection"),
        "metrics_hourly": get("/metrics/hourly"),
    }

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
