#!/usr/bin/env python3
"""
Split file manager: model vs. append-only logs for performance and clarity.
"""

import json
import threading
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Optional

from .model import PersistentMindModel, Event, Insight, Thought
from .validation import validate_model


class SplitModelManager:
    """Manages model state + append-only event/insight/thought logs."""

    def __init__(
        self,
        model_path: str,
        events_path: Optional[str] = None,
        insights_path: Optional[str] = None,
        thoughts_path: Optional[str] = None,
        recent_events_limit: int = 50,
        recent_insights_limit: int = 20,
        recent_thoughts_limit: int = 30,
    ):
        self.model_path = Path(model_path)
        self.events_path = Path(
            events_path or model_path.replace(".json", ".events.jsonl")
        )
        self.insights_path = Path(
            insights_path or model_path.replace(".json", ".insights.jsonl")
        )
        self.thoughts_path = Path(
            thoughts_path or model_path.replace(".json", ".thoughts.jsonl")
        )

        self.recent_events_limit = recent_events_limit
        self.recent_insights_limit = recent_insights_limit
        self.recent_thoughts_limit = recent_thoughts_limit

        self.lock = threading.RLock()
        self.model = self._load_or_create_model()

    def _load_or_create_model(self) -> PersistentMindModel:
        """Load model from file, creating new if doesn't exist."""
        if not self.model_path.exists():
            model = PersistentMindModel()
            self._save_model_only(model)
            return model

        try:
            with open(self.model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            validate_model(data)
            # TODO: Implement proper dataclass deserialization
            return PersistentMindModel()
        except Exception as e:
            raise ValueError(f"Failed to load model from {self.model_path}: {e}")

    def _save_model_only(self, model: PersistentMindModel) -> None:
        """Save just the model state (not logs)."""
        # TODO: Implement proper dataclass serialization
        model_dict = {"placeholder": "implement_serialization"}

        temp_path = self.model_path.with_suffix(".tmp")
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(model_dict, f, indent=2, ensure_ascii=False)
            temp_path.replace(self.model_path)
        except Exception as e:
            if temp_path.exists():
                temp_path.unlink()
            raise ValueError(f"Failed to save model: {e}")

    def _append_to_log(self, log_path: Path, entry: dict) -> None:
        """Append entry to JSONL log file."""
        with open(log_path, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    def add_event(
        self,
        summary: str,
        effects: Optional[List[dict]] = None,
        *,
        etype: str = "experience",
    ) -> Event:
        """Add event to log and keep recent summary in model."""
        with self.lock:
            ev_id = f"ev{self._get_next_event_id()}"
            ts = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

            # Create full event
            event = Event(
                id=ev_id,
                t=ts,
                type=etype,
                summary=summary,
                effects_hypothesis=effects or [],
            )

            # Append to log
            event_dict = {
                "id": ev_id,
                "t": ts,
                "type": etype,
                "summary": summary,
                "effects_hypothesis": effects or [],
            }
            self._append_to_log(self.events_path, event_dict)

            # Keep recent events in model (summarized)
            self.model.self_knowledge.autobiographical_events.append(event)
            if (
                len(self.model.self_knowledge.autobiographical_events)
                > self.recent_events_limit
            ):
                self.model.self_knowledge.autobiographical_events = (
                    self.model.self_knowledge.autobiographical_events[
                        -self.recent_events_limit :
                    ]
                )

            self._save_model_only(self.model)
            return event

    def add_insight(
        self,
        content: str,
        summary: Optional[str] = None,
        commitment: Optional[str] = None,
    ) -> Insight:
        """Add insight to log and keep recent summary in model."""
        with self.lock:
            in_id = f"in{self._get_next_insight_id()}"
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Extract commitment if not provided
            if not commitment:
                commitment = self._extract_commitment(content)

            # Create summary if not provided
            if not summary:
                summary = content[:100] + "..." if len(content) > 100 else content

            # Create full insight
            insight = Insight(id=in_id, t=ts, content=content)

            # Append to log with extended fields
            insight_dict = {
                "id": in_id,
                "t": ts,
                "content": content,
                "summary": summary,
                "commitment": commitment,
                "references": {},
            }
            self._append_to_log(self.insights_path, insight_dict)

            # Keep recent insights in model (summarized)
            insight.content = summary  # Store summary in model
            self.model.self_knowledge.insights.append(insight)
            if len(self.model.self_knowledge.insights) > self.recent_insights_limit:
                self.model.self_knowledge.insights = self.model.self_knowledge.insights[
                    -self.recent_insights_limit :
                ]

            # Track commitment if found
            if commitment:
                self._track_commitment(commitment, in_id)

            self._save_model_only(self.model)
            return insight

    def add_thought(self, content: str, trigger: str = "") -> Thought:
        """Add thought to log and keep recent in model."""
        with self.lock:
            th_id = f"th{self._get_next_thought_id()}"
            ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

            # Create thought
            thought = Thought(id=th_id, t=ts, content=content, trigger=trigger)

            # Append to log
            thought_dict = {
                "id": th_id,
                "t": ts,
                "content": content,
                "trigger": trigger,
            }
            self._append_to_log(self.thoughts_path, thought_dict)

            # Keep recent thoughts in model
            self.model.self_knowledge.thoughts.append(thought)
            if len(self.model.self_knowledge.thoughts) > self.recent_thoughts_limit:
                self.model.self_knowledge.thoughts = self.model.self_knowledge.thoughts[
                    -self.recent_thoughts_limit :
                ]

            self._save_model_only(self.model)
            return thought

    def _extract_commitment(self, content: str) -> Optional[str]:
        """Extract commitment from insight content."""
        lines = content.lower().split(".")
        for line in lines:
            line = line.strip()
            if any(
                starter in line
                for starter in ["i will", "next:", "i plan to", "i commit to"]
            ):
                return line.capitalize()
        return None

    def _track_commitment(self, commitment: str, insight_id: str) -> None:
        """Track commitment in future_scripts.goals."""
        _goal_id = f"goal_{len(self.model.narrative_identity.future_scripts.goals) + 1}"
        # TODO: Add Goal dataclass with status tracking
        # For now, just increment commitment counter
        if not hasattr(self.model.metrics, "commitments_open"):
            self.model.metrics.commitments_open = 0
        self.model.metrics.commitments_open += 1

    def _get_next_event_id(self) -> int:
        """Get next event ID by counting log lines."""
        if not self.events_path.exists():
            return 1
        with open(self.events_path, "r") as f:
            return sum(1 for _ in f) + 1

    def _get_next_insight_id(self) -> int:
        """Get next insight ID by counting log lines."""
        if not self.insights_path.exists():
            return 1
        with open(self.insights_path, "r") as f:
            return sum(1 for _ in f) + 1

    def _get_next_thought_id(self) -> int:
        """Get next thought ID by counting log lines."""
        if not self.thoughts_path.exists():
            return 1
        with open(self.thoughts_path, "r") as f:
            return sum(1 for _ in f) + 1

    def save_model(self) -> None:
        """Save current model state."""
        with self.lock:
            self._save_model_only(self.model)

    def get_recent_insights_full(self, limit: int = 5) -> List[dict]:
        """Get recent insights with full content from log."""
        if not self.insights_path.exists():
            return []

        insights = []
        with open(self.insights_path, "r") as f:
            for line in f:
                insights.append(json.loads(line.strip()))

        return insights[-limit:] if insights else []
