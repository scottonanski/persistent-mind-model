# pmm/memory_keys.py
"""Memory namespacing utilities to prevent cross-model contamination."""


def agent_namespace(cfg: dict, install_id: str) -> str:
    """Generate agent-specific namespace for model-scoped memory."""
    provider = cfg.get("provider", "unknown")
    name = cfg.get("name", "unknown")
    epoch = cfg.get("epoch", 0)
    return f"{install_id}/agent/{provider}:{name}/epoch/{epoch}"


def global_namespace(install_id: str) -> str:
    """Generate global namespace for user facts shared across models."""
    return f"{install_id}/global"


def commitment_key(namespace: str, commitment_id: str) -> str:
    """Generate key for commitment storage."""
    return f"{namespace}/commitment/{commitment_id}"


def insight_key(namespace: str, insight_id: str) -> str:
    """Generate key for insight storage."""
    return f"{namespace}/insight/{insight_id}"


def user_fact_key(namespace: str, fact_type: str) -> str:
    """Generate key for user fact storage."""
    return f"{namespace}/user_fact/{fact_type}"


def agent_state_key(namespace: str, state_type: str) -> str:
    """Generate key for agent state storage."""
    return f"{namespace}/agent_state/{state_type}"
