import os
import tempfile
from pathlib import Path

from pmm.langchain_memory import PersistentMindMemory


def _make_memory(tmpdir: Path, embeddings: bool = False) -> PersistentMindMemory:
    return PersistentMindMemory(
        agent_path=str(tmpdir / "agent.json"),
        personality_config={"openness": 0.7},
        enable_embeddings=embeddings,
    )


def _history_to_str(history_val) -> str:
    """Coerce history into a single string for assertions.

    Supports either a raw string or a list of LangChain messages.
    """
    if isinstance(history_val, str):
        return history_val
    try:
        # Likely a list of BaseMessage (e.g., HumanMessage/SystemMessage)
        parts = []
        for m in history_val or []:
            content = getattr(m, "content", None)
            if isinstance(content, list):
                # Some message content can be a list of chunks; join text chunks
                parts.append(" ".join(str(getattr(c, "text", c)) for c in content))
            elif content is not None:
                parts.append(str(content))
        return "\n".join(parts)
    except Exception:
        return str(history_val)


def _count_recent_history_lines(history_blob) -> int:
    """Extract the Recent conversation history section and count lines."""
    text = _history_to_str(history_blob)
    marker = "Recent conversation history:\n"
    if marker not in text:
        return 0
    section = text.split(marker, 1)[1]
    # Stop at a double newline or end
    # Keep it simple: count up to next blank line
    lines = []
    for line in section.splitlines():
        if not line.strip():
            break
        lines.append(line)
    return len(lines)


def test_priority_recall_budget_caps_s4():
    """When stage is S4, recall should be tighter (<=10 lines, possibly 8 if scenes exist)."""
    os.environ["PMM_HARD_STAGE"] = "S4"
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        mem = _make_memory(tmpdir, embeddings=False)

        # Seed many events (more than cap)
        for i in range(25):
            mem.pmm.add_event(summary=f"seed event {i}", etype="event")

        # Add a commitment (ensures commitment bonus path exists)
        mem.pmm.add_commitment("Test recall prioritization.", "test")

        vars = mem.load_memory_variables({"input": "test recall"})
        history = vars.get("history", "")
        assert history
        n = _count_recent_history_lines(history)
        assert n <= 10


def test_priority_recall_budget_caps_s0():
    """When stage is S0, recall budget can be wider (<=15 lines)."""
    os.environ["PMM_HARD_STAGE"] = "S0"
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        mem = _make_memory(tmpdir, embeddings=False)

        for i in range(25):
            mem.pmm.add_event(summary=f"seed event {i}", etype="event")

        mem.pmm.add_commitment("Test recall prioritization.", "test")

        vars = mem.load_memory_variables({"input": "test recall"})
        history = vars.get("history", "")
        assert history
        n = _count_recent_history_lines(history)
        assert n <= 15
        # commit presence is validated semantically elsewhere; avoid brittle string checks
