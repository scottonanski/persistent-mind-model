from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime, timezone


@dataclass
class DevTask:
    task_id: str
    kind: str
    title: str
    ttl: float
    policy: Dict[str, Any]
    status: str = "open"  # open|closed
    created_at: str = ""
    closed_at: Optional[str] = None


class DevTaskManager:
    """Lightweight dev-task manager that persists via SQLite events.

    Event kinds written:
      - task_created
      - task_progress
      - task_closed
    """

    def __init__(self, sqlite_store):
        self.store = sqlite_store
        self._counter = 0

    def _new_id(self) -> str:
        self._counter += 1
        return f"dt{self._counter}"

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def open_task(
        self,
        kind: str,
        title: str,
        ttl_turns: Optional[int] = None,
        ttl_hours: Optional[float] = None,
        policy: Optional[Dict[str, Any]] = None,
    ) -> str:
        tid = self._new_id()
        ttl = ttl_turns if ttl_turns is not None else float(ttl_hours or 0)
        task = DevTask(
            task_id=tid,
            kind=str(kind),
            title=str(title),
            ttl=ttl,
            policy=policy or {},
            created_at=self._now(),
        )
        import json
        self.store.append_event(
            kind="task_created",
            content=json.dumps(asdict(task), ensure_ascii=False),
            meta={"task_id": tid},
        )
        return tid

    def update_task(self, task_id: str, *, note: Optional[str] = None, percent: Optional[float] = None) -> None:
        import json
        payload = {"note": note, "percent": percent}
        self.store.append_event(
            kind="task_progress",
            content=json.dumps(payload, ensure_ascii=False),
            meta={"task_id": task_id},
        )

    def close_task(self, task_id: str, *, reason: Optional[str] = None, evidence: Optional[str] = None) -> None:
        import json
        # 1) Emit an evidence:done row if any evidence string is provided
        if evidence:
            ev_content = {
                "type": "done",
                "summary": f"Task {task_id} closed",
                "artifact": evidence,
                "confidence": 0.7,
            }
            self.store.append_event(
                kind="evidence",
                content=json.dumps(ev_content, ensure_ascii=False),
                meta={"task_ref": task_id},
            )
        # 2) Append the task_closed row (unchanged)
        payload = {"reason": reason, "evidence": evidence}
        self.store.append_event(
            kind="task_closed",
            content=json.dumps(payload, ensure_ascii=False),
            meta={"task_id": task_id},
        )
