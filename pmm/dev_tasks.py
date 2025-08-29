#!/usr/bin/env python3
"""
Development task management for PMM.
Provides DevTaskManager class for tracking development tasks and progress.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class DevTask:
    """Represents a development task."""

    id: str
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, blocked
    priority: str = "medium"  # low, medium, high, critical
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


class DevTaskManager:
    """
    Manages development tasks for PMM system.

    This is a stub implementation to satisfy imports in chat.py.
    Can be expanded later for actual task management functionality.
    """

    def __init__(self):
        self.tasks: Dict[str, DevTask] = {}
        self._next_id = 1

    def create_task(
        self, title: str, description: str = "", priority: str = "medium"
    ) -> str:
        """Create a new development task."""
        task_id = f"task_{self._next_id:04d}"
        self._next_id += 1

        task = DevTask(
            id=task_id, title=title, description=description, priority=priority
        )

        self.tasks[task_id] = task
        return task_id

    def get_task(self, task_id: str) -> Optional[DevTask]:
        """Get a task by ID."""
        return self.tasks.get(task_id)

    def update_task_status(self, task_id: str, status: str) -> bool:
        """Update task status."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status
            self.tasks[task_id].updated_at = datetime.now()
            return True
        return False

    def list_tasks(self, status: Optional[str] = None) -> List[DevTask]:
        """List tasks, optionally filtered by status."""
        tasks = list(self.tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        return sorted(tasks, key=lambda t: t.created_at or datetime.min)

    def get_task_summary(self) -> Dict[str, Any]:
        """Get summary of all tasks."""
        total = len(self.tasks)
        by_status = {}
        by_priority = {}

        for task in self.tasks.values():
            by_status[task.status] = by_status.get(task.status, 0) + 1
            by_priority[task.priority] = by_priority.get(task.priority, 0) + 1

        return {
            "total_tasks": total,
            "by_status": by_status,
            "by_priority": by_priority,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export tasks to dictionary format."""
        return {
            "tasks": {
                task_id: {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status,
                    "priority": task.priority,
                    "created_at": (
                        task.created_at.isoformat() if task.created_at else None
                    ),
                    "updated_at": (
                        task.updated_at.isoformat() if task.updated_at else None
                    ),
                    "metadata": task.metadata,
                }
                for task_id, task in self.tasks.items()
            },
            "next_id": self._next_id,
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Import tasks from dictionary format."""
        self.tasks.clear()
        self._next_id = data.get("next_id", 1)

        for task_id, task_data in data.get("tasks", {}).items():
            task = DevTask(
                id=task_data["id"],
                title=task_data["title"],
                description=task_data["description"],
                status=task_data["status"],
                priority=task_data["priority"],
                metadata=task_data.get("metadata", {}),
            )

            # Parse datetime strings
            if task_data.get("created_at"):
                try:
                    task.created_at = datetime.fromisoformat(task_data["created_at"])
                except ValueError:
                    pass

            if task_data.get("updated_at"):
                try:
                    task.updated_at = datetime.fromisoformat(task_data["updated_at"])
                except ValueError:
                    pass

            self.tasks[task_id] = task
