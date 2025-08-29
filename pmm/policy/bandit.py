from __future__ import annotations

import os
import json
import random
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

try:
    # Local import: PMM's SQLite wrapper (preferred)
    from pmm.storage.sqlite_store import SQLiteStore
except Exception:  # pragma: no cover - fallback for isolated imports
    SQLiteStore = None  # type: ignore


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _env_flag(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None else default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_flag(name, str(default)))
    except Exception:
        return default


_DEFAULT_DB_PATH = _env_flag("PMM_DB_PATH", "pmm.db")


@dataclass
class _PolicyRow:
    action: str
    value: float
    pulls: int
    updated_at: str


class _BanditCore:
    """Minimal ε-greedy contextual bandit over two actions.

    Actions: "reflect_now" | "continue"

    - Epsilon decays after each selection until a floor.
    - Q-values stored as running average in SQLite.
    - Rewards recorded into a separate table for audit.
    """

    def __init__(self, store: Optional[SQLiteStore] = None):
        self._lock = threading.RLock()
        self.store = store or (SQLiteStore(_DEFAULT_DB_PATH) if SQLiteStore else None)
        self._ensure_tables()
        # In-memory epsilon state
        self._eps = _env_float("PMM_BANDIT_EPSILON", 0.10)
        self._eps_floor = _env_float("PMM_BANDIT_EPSILON_FLOOR", 0.02)

    # ---- schema management ----
    def _ensure_tables(self) -> None:
        if not self.store:
            return
        with self.store._lock:  # use same lock as store
            c = self.store.conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_policy (
                    action TEXT PRIMARY KEY,
                    value REAL NOT NULL,
                    pulls INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS bandit_rewards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    ctx JSON,
                    action TEXT NOT NULL,
                    reward REAL NOT NULL,
                    horizon INTEGER NOT NULL,
                    notes TEXT
                )
                """
            )
            # Seed two actions if missing
            for a in ("reflect_now", "continue"):
                c.execute("SELECT action FROM bandit_policy WHERE action=?", (a,))
                if c.fetchone() is None:
                    c.execute(
                        "INSERT INTO bandit_policy(action, value, pulls, updated_at) VALUES(?,?,?,?)",
                        (a, 0.0, 0, _now_utc()),
                    )
            self.store.conn.commit()

    # ---- policy I/O ----
    def load_policy(self) -> Dict[str, _PolicyRow]:
        if not self.store:
            return {
                "reflect_now": _PolicyRow("reflect_now", 0.0, 0, _now_utc()),
                "continue": _PolicyRow("continue", 0.0, 0, _now_utc()),
            }
        with self.store._lock:
            rows = list(
                self.store.conn.execute(
                    "SELECT action, value, pulls, updated_at FROM bandit_policy"
                )
            )
        out: Dict[str, _PolicyRow] = {}
        for r in rows:
            out[str(r[0])] = _PolicyRow(str(r[0]), float(r[1]), int(r[2]), str(r[3]))
        return out

    def save_policy(self, rows: Dict[str, _PolicyRow]) -> None:
        if not self.store:
            return
        with self.store._lock:
            for a, pr in rows.items():
                self.store.conn.execute(
                    "UPDATE bandit_policy SET value=?, pulls=?, updated_at=? WHERE action=?",
                    (float(pr.value), int(pr.pulls), pr.updated_at, a),
                )
            self.store.conn.commit()

    # ---- selection ----
    def _decay_eps(self) -> None:
        if self._eps > self._eps_floor:
            # Simple linear-ish decay per decision
            self._eps = max(self._eps_floor, round(self._eps - 0.001, 6))

    def select_action(self, context: Dict) -> Tuple[str, float, float, float, bool]:
        """Pick an action. Returns (action, eps_used, q_reflect, q_continue, hot_bias)."""
        policy = self.load_policy()
        q_reflect = policy.get("reflect_now", _PolicyRow("reflect_now", 0.0, 0, _now_utc())).value
        q_continue = policy.get("continue", _PolicyRow("continue", 0.0, 0, _now_utc())).value

        eps = float(self._eps)
        hot_bias = False
        try:
            # If hot and reflect is currently winning, reduce exploration temporarily
            hot = float(context.get("hot", 0.0) or 0.0) >= 1.0
            if hot and (q_reflect >= q_continue):
                eps = max(self._eps_floor, eps / 2.0)
                hot_bias = True
        except Exception:
            pass
        action: str
        if random.random() < eps:
            action = random.choice(["reflect_now", "continue"])  # explore
        else:
            action = "reflect_now" if q_reflect >= q_continue else "continue"  # exploit

        # Decay epsilon after each selection
        self._decay_eps()
        return action, eps, q_reflect, q_continue, hot_bias

    # ---- reward recording ----
    def record_outcome(
        self,
        context: Dict,
        action: str,
        reward: float,
        horizon: int,
        notes: str = "",
    ) -> None:
        """Record reward for the given action and update Q-values."""
        try:
            if action not in ["reflect_now", "continue"]:
                return
            
            # Step 5: Reward shaping for hot contexts
            original_reward = reward
            hot_strength = context.get("hot_strength", 0.0) if context else 0.0
            
            # Hot context reward shaping
            if hot_strength >= 0.5 and action == "reflect_now":
                # Boost reflection rewards in hot contexts
                hot_boost = float(os.getenv("PMM_BANDIT_HOT_REFLECT_BOOST", "0.3"))
                reward += hot_boost * hot_strength
            elif hot_strength >= 0.5 and action == "continue":
                # Mild penalty for continuing without reflection in hot contexts
                hot_penalty = float(os.getenv("PMM_BANDIT_HOT_CONTINUE_PENALTY", "0.1"))
                reward -= hot_penalty * hot_strength
            
            # Clamp reward after shaping
            if reward < -1.0:
                reward = -1.0
            if reward > 1.0:
                reward = 1.0

            # Step 4: Track reflection ID for credit assignment
            reflect_id = context.get("reflect_id") if context else None

            # Persist reward row
            if self.store:
                try:
                    with self.store._lock:
                        # Update running average for the action
                        row = self.store.conn.execute(
                            "SELECT value, pulls FROM bandit_policy WHERE action=?",
                            (action,),
                        ).fetchone()
                        if row is None:
                            # Should not happen; ensure row exists
                            self.store.conn.execute(
                                "INSERT OR REPLACE INTO bandit_policy(action, value, pulls, updated_at) VALUES(?,?,?,?)",
                                (action, float(reward), 1, _now_utc()),
                            )
                        else:
                            old_v, old_n = float(row[0]), int(row[1])
                            new_n = old_n + 1
                            new_v = old_v + (float(reward) - old_v) / max(new_n, 1)
                            self.store.conn.execute(
                                "UPDATE bandit_policy SET value=?, pulls=?, updated_at=? WHERE action=?",
                                (new_v, new_n, _now_utc(), action),
                            )
                        self.store.conn.commit()
                except Exception:
                    # Degrade gracefully: ignore DB failures
                    pass

            # Telemetry with reflection ID tracking and reward shaping
            if os.getenv("PMM_TELEMETRY", "").lower() in ("1", "true", "yes", "on"):
                reflect_info = f", reflect_id={reflect_id}" if reflect_id else ""
                shaping_info = f", original_reward={original_reward:.3f}, hot_strength={hot_strength:.3f}" if hot_strength > 0 else ""
                print(f"[PMM_TELEMETRY] bandit_reward: action={action}, reward={reward:.3f}, horizon={horizon}{reflect_info}{shaping_info}")
                
        except Exception:
            # Graceful degradation
            pass


# ---- module-level singleton wired to PMM DB ----
_BANDIT_SINGLETON: Optional[_BanditCore] = None


def _get_core(store: Optional[SQLiteStore] = None) -> _BanditCore:
    global _BANDIT_SINGLETON
    if _BANDIT_SINGLETON is None:
        _BANDIT_SINGLETON = _BanditCore(store)
    elif store is not None and _BANDIT_SINGLETON.store is None:
        # Late-bind a store, if created after first import
        _BANDIT_SINGLETON.store = store
        _BANDIT_SINGLETON._ensure_tables()
    return _BANDIT_SINGLETON


# Public API
def select_action(context: Dict, store: Optional[SQLiteStore] = None) -> str:
    core = _get_core(store)
    action, _eps, _qr, _qc, _hb = core.select_action(context)
    # Only two actions allowed
    return "reflect_now" if action == "reflect_now" else "continue"


def select_action_info(context: Dict, store: Optional[SQLiteStore] = None) -> Tuple[str, float, float, float, bool]:
    """Select action and return (action, eps, q_reflect, q_continue, hot_bias)."""
    core = _get_core(store)
    action, eps, q_reflect, q_continue, hot_bias = core.select_action(context)
    action = "reflect_now" if action == "reflect_now" else "continue"
    return action, float(eps), float(q_reflect), float(q_continue), bool(hot_bias)


def record_outcome(
    context: Dict,
    action: str,
    reward: float,
    horizon: int,
    notes: str = "",
    store: Optional[SQLiteStore] = None,
) -> None:
    core = _get_core(store)
    core.record_outcome(context, action, reward, horizon, notes)


def load_policy(store: Optional[SQLiteStore] = None) -> Dict[str, Dict[str, float]]:
    core = _get_core(store)
    rows = core.load_policy()
    return {k: {"value": v.value, "pulls": v.pulls} for k, v in rows.items()}


def save_policy(policy: Dict[str, Dict[str, float]], store: Optional[SQLiteStore] = None) -> None:
    core = _get_core(store)
    rows: Dict[str, _PolicyRow] = {}
    for a, obj in policy.items():
        rows[a] = _PolicyRow(
            action=a,
            value=float(obj.get("value", 0.0)),
            pulls=int(obj.get("pulls", 0)),
            updated_at=_now_utc(),
        )
    core.save_policy(rows)


# Convenience: allow callers to explicitly wire the shared store if they have it
def set_store(sqlite_store: SQLiteStore) -> None:
    core = _get_core(sqlite_store)
    core.store = sqlite_store
    core._ensure_tables()


def get_status(store: Optional[SQLiteStore] = None) -> Dict[str, float]:
    """Return current epsilon and Qs for probe/telemetry."""
    core = _get_core(store)
    policy = core.load_policy()
    q_reflect = policy.get("reflect_now", _PolicyRow("reflect_now", 0.0, 0, _now_utc())).value
    q_continue = policy.get("continue", _PolicyRow("continue", 0.0, 0, _now_utc())).value
    return {"eps": float(core._eps), "q_reflect": float(q_reflect), "q_continue": float(q_continue)}


def get_winrate_reflect(last_n: int = 50, store: Optional[SQLiteStore] = None) -> Optional[float]:
    """Compute rolling acceptance rate for 'reflect_now' over last N outcomes.

    Uses bandit_rewards.notes text to detect accepted reflections.
    """
    s = store or (SQLiteStore(_DEFAULT_DB_PATH) if SQLiteStore else None)
    if not s:
        return None
    try:
        with s._lock:
            rows = list(
                s.conn.execute(
                    "SELECT notes FROM bandit_rewards WHERE action='reflect_now' ORDER BY id DESC LIMIT ?",
                    (int(last_n),),
                )
            )
        if not rows:
            return 0.0
        total = len(rows)
        wins = 0
        for (notes,) in rows:
            text = str(notes or "")
            if "accepted=True" in text:
                wins += 1
        return float(wins) / float(total) if total else 0.0
    except Exception:
        return None


def _to_float(x: Optional[float], default: float = 0.0) -> float:
    try:
        if x is None:
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clamp01(x: float) -> float:
    if x != x:  # NaN
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def compute_hot_strength(gas: float, close_rate: float, rolling_close_5: Optional[float] = None) -> float:
    """Compute hot_strength ∈ [0,1] based on GAS and close rates.
    
    gas_term = clamp((GAS-0.85)/0.10, 0, 1)
    close_term = max(commit_close_rate_15w, rolling_close_5) then clamp((close_term-0.60)/0.20, 0, 1)
    hot_strength = 0.5*gas_term + 0.5*close_term
    """
    gas_term = _clamp01((gas - 0.85) / 0.10)
    
    close_term = max(close_rate, rolling_close_5 or 0.0)
    close_term = _clamp01((close_term - 0.60) / 0.20)
    
    hot_strength = 0.5 * gas_term + 0.5 * close_term
    return _clamp01(hot_strength)


def build_context(
    *,
    gas: Optional[float] = None,
    ias: Optional[float] = None,
    close: Optional[float] = None,
    hot: Optional[bool] = None,
    identity_signal_count: Optional[float] = None,
    time_since_last_reflection_sec: Optional[float] = None,
    dedup_threshold: Optional[float] = None,
    inert_streak: Optional[float] = None,
    rolling_close_5: Optional[float] = None,
) -> Dict[str, float]:
    """Build bandit context dict with normalized/clamped features.

    Floats are clamped to [0,1] unless otherwise noted:
    - identity_signal_count -> min(count/4, 1.0)
    - time_since_last_reflection_sec -> min(t/30, 1.0)
    - inert_streak -> min(streak/3, 1.0)
    - hot -> 1.0 if hot_strength >= 0.5 else 0.0
    - hot_strength -> computed from GAS and close rates
    """

    g = _clamp01(_to_float(gas))
    i = _clamp01(_to_float(ias))
    c = _clamp01(_to_float(close))

    # Compute hot_strength and derive hot flag
    hot_strength = compute_hot_strength(g, c, rolling_close_5)
    h = 1.0 if hot_strength >= 0.5 else 0.0

    # Identity signal count scaled
    isc_raw = _to_float(identity_signal_count)
    isc = _clamp01(isc_raw / 4.0)

    # Time since last reflection scaled by 30s window
    ts_raw = max(0.0, _to_float(time_since_last_reflection_sec))
    ts = _clamp01(ts_raw / 30.0)

    # Dedup threshold (already 0..1 typically, clamp anyway)
    dt = _clamp01(_to_float(dedup_threshold))

    # Inert streak scaled
    is_raw = max(0.0, _to_float(inert_streak))
    ist = _clamp01(is_raw / 3.0)

    ctx = {
        "gas": g,
        "ias": i,
        "close": c,
        "hot": h,
        "hot_strength": hot_strength,
        "identity_signal_count": isc,
        "time_since_last_reflection_sec": ts,
        "dedup_threshold": dt,
        "inert_streak": ist,
    }
    # Ensure JSON-serializable by round-tripping via json when debug is enabled
    try:
        json.dumps(ctx)
    except Exception:
        # Fallback to stringify anything unexpected (shouldn't occur with primitives)
        for k, v in list(ctx.items()):
            try:
                json.dumps({k: v})
            except Exception:
                ctx[k] = float(v) if isinstance(v, (int, float)) else str(v)
    return ctx
