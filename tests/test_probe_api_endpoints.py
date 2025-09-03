import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

def _import_app():
    # accept multiple possible module paths
    for modpath in ("pmm.api.probe", "pmm.probe", "pmm.api"):
        try:
            mod = __import__(modpath, fromlist=["app"])
            if hasattr(mod, "app"):
                return mod.app
        except Exception:
            continue
    pytest.skip("FastAPI probe app not found (pmm.api.probe:app).")

def test_probe_endpoints_up():
    app = _import_app()
    client = TestClient(app)

    r = client.get("/endpoints")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, dict) and data, "/endpoints must list endpoints"

    # sanity on a couple of core endpoints if present
    for path in ("/identity", "/emergence", "/health"):
        resp = client.get(path)
        assert resp.status_code in (200, 404), f"{path} should exist or be intentionally absent"
        if resp.status_code == 200:
            assert resp.json() is not None
