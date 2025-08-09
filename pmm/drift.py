from __future__ import annotations
from datetime import date, datetime
from typing import Dict
from .model import PersistentMindModel, TraitScore, IdentityChange


def _days_since(yyyy_mm_dd: str) -> int:
    try:
        d = date.fromisoformat(yyyy_mm_dd)
        return (date.today() - d).days
    except Exception:
        return 10_000  # treat invalid as very old


def _get_trait(model: PersistentMindModel, target_field: str) -> TraitScore | None:
    # supports: "personality.traits.big5.X.score" or "personality.traits.hexaco.X.score"
    parts = target_field.split(".")
    if len(parts) < 5 or parts[0] != "personality" or parts[1] != "traits":
        return None
    group, trait, leaf = parts[2], parts[3], parts[-1]
    if leaf != "score":
        return None
    if group == "big5":
        return getattr(model.personality.traits.big5, trait, None)
    if group == "hexaco":
        return getattr(model.personality.traits.hexaco, trait, None)
    return None


def apply_effects(model: PersistentMindModel) -> Dict[str, float]:
    """Apply pending effects_hypothesis to traits.
    Returns net drift per trait-path (without .score).
    """
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg = model.drift_config
    net: Dict[str, float] = {}

    for ev in model.self_knowledge.autobiographical_events:
        if isinstance(ev.meta, dict) and ev.meta.get("processed"):
            continue
        for eff in ev.effects_hypothesis or []:
            target = getattr(eff, "target", None)
            delta = float(getattr(eff, "delta", 0.0) or 0.0)
            conf = float(getattr(eff, "confidence", 0.0) or 0.0)
            if not target or delta == 0.0:
                continue
            # locks
            if any(target.startswith(lock) for lock in (cfg.locks or [])):
                continue
            trait = _get_trait(model, target)
            if trait is None:
                continue
            # cooldown
            if _days_since(trait.last_update) < cfg.cooldown_days:
                continue
            # scale delta
            eff_delta = delta * (1.0 - cfg.inertia) * cfg.event_sensitivity
            # clamp by max step
            max_step = cfg.max_delta_per_reflection
            if eff_delta > max_step:
                eff_delta = max_step
            elif eff_delta < -max_step:
                eff_delta = -max_step
            if abs(eff_delta) < 1e-9:
                continue
            # bounds
            new_val = trait.score + eff_delta
            if new_val > cfg.bounds.max:
                new_val = cfg.bounds.max
            if new_val < cfg.bounds.min:
                new_val = cfg.bounds.min
            real_applied = new_val - trait.score
            if abs(real_applied) < 1e-9:
                continue
            # apply
            trait.score = round(new_val, 4)
            trait.last_update = date.today().isoformat()
            trait.origin = "drift"
            trait.conf = max(trait.conf, conf)
            key = target.rsplit(".score", 1)[0]
            net[key] = round(net.get(key, 0.0) + real_applied, 4)
            model.meta_cognition.identity_evolution.append(
                IdentityChange(t=now, change=f"{key} {'+' if real_applied>=0 else ''}{round(real_applied,4)} (conf {conf}) via {ev.id}")
            )
            model.meta_cognition.self_modification_count += 1
        # mark processed
        if isinstance(ev.meta, dict):
            ev.meta["processed"] = True

    # update drift_velocity metrics
    for k, v in net.items():
        model.metrics.drift_velocity[k] = v
    return net
