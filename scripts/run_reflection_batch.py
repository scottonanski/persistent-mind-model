#!/usr/bin/env python3
"""
Generate a batch of reflections using the live adapter to exercise the
Next: contract and then print probe snapshots.
"""
import os
import sys
import time
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pmm.self_model_manager import SelfModelManager  # noqa: E402
from pmm.reflection import reflect_once  # noqa: E402


def run_reflections(n: int = 12):
    mgr = SelfModelManager("persistent_self_model.json")
    # Ensure enforcement + reroll on
    os.environ.setdefault("PMM_ENFORCE_NEXT_CONTRACT", "1")
    os.environ.setdefault("PMM_REFLECT_REROLL_ON_CONTRACT_FAIL", "1")
    os.environ.setdefault("PMM_REFLECT_REROLL_TEMP", "0.1")

    ok = 0
    fail = 0
    for i in range(n):
        try:
            ins = reflect_once(mgr, llm=None, active_model_config=None)
            meta = getattr(ins, "meta", {}) or {}
            if meta.get("accepted"):
                ok += 1
            else:
                fail += 1
        except Exception as e:
            print(f"reflect error: {type(e).__name__}: {e}")
            fail += 1
        time.sleep(0.3)
    return ok, fail


def ensure_probe(port: int = 8002):
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "pmm.api.probe:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=0.1,
        )
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        # spawn background
        subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "pmm.api.probe:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    time.sleep(1.0)


def main():
    ok, fail = run_reflections(12)
    print(f"Reflections: ok={ok} fail={fail}")
    ensure_probe(8002)
    import urllib.request
    import json

    def get(path):
        with urllib.request.urlopen(f"http://127.0.0.1:8002{path}") as resp:
            return json.loads(resp.read().decode("utf-8"))

    print(json.dumps(get("/reflection/contract?limit=12"), indent=2))
    print(json.dumps(get("/emergence"), indent=2))


if __name__ == "__main__":
    main()
