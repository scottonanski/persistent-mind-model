from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime, date, UTC
import uuid

# ===== Core Identity =====


@dataclass
class CoreIdentity:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = field(default_factory=lambda: f"Agent-{str(uuid.uuid4())[:8]}")
    birth_timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    aliases: List[str] = field(default_factory=list)


# ===== Personality =====


@dataclass
class TraitScore:
    score: float
    conf: float
    last_update: str
    origin: str


def _new_trait() -> TraitScore:
    return TraitScore(0.5, 0.5, date.today().isoformat(), "init")


@dataclass
class BigFiveTraits:
    openness: TraitScore = field(default_factory=_new_trait)
    conscientiousness: TraitScore = field(default_factory=_new_trait)
    extraversion: TraitScore = field(default_factory=_new_trait)
    agreeableness: TraitScore = field(default_factory=_new_trait)
    neuroticism: TraitScore = field(default_factory=_new_trait)


@dataclass
class HexacoTraits:
    honesty_humility: TraitScore = field(default_factory=_new_trait)
    emotionality: TraitScore = field(default_factory=_new_trait)
    extraversion: TraitScore = field(default_factory=_new_trait)
    agreeableness: TraitScore = field(default_factory=_new_trait)
    conscientiousness: TraitScore = field(default_factory=_new_trait)
    openness: TraitScore = field(default_factory=_new_trait)


@dataclass
class MBTIPoles:
    E: float = 0.5
    I: float = 0.5  # noqa: E741
    S: float = 0.5
    N: float = 0.5
    T: float = 0.5
    F: float = 0.5
    J: float = 0.5
    P: float = 0.5


@dataclass
class MBTI:
    label: str = "INTP"
    poles: MBTIPoles = field(default_factory=MBTIPoles)
    conf: float = 0.5
    last_update: str = field(default_factory=lambda: date.today().isoformat())
    origin: str = "display_only"


@dataclass
class ValueWeight:
    id: str
    weight: float


@dataclass
class Preferences:
    style: str = "concise"
    risk_tolerance: float = 0.5
    collaboration_bias: float = 0.5


@dataclass
class EmotionalTendencies:
    baseline_stability: float = 0.5
    assertiveness: float = 0.5
    cooperativeness: float = 0.5


@dataclass
class Traits:
    big5: BigFiveTraits = field(default_factory=BigFiveTraits)
    hexaco: HexacoTraits = field(default_factory=HexacoTraits)


@dataclass
class Personality:
    traits: Traits = field(default_factory=Traits)
    mbti: MBTI = field(default_factory=MBTI)
    values_schwartz: List[ValueWeight] = field(default_factory=list)
    preferences: Preferences = field(default_factory=Preferences)
    emotional_tendencies: EmotionalTendencies = field(
        default_factory=EmotionalTendencies
    )


# ===== Narrative Identity =====


@dataclass
class Chapter:
    id: str
    title: str
    start: str
    end: Optional[str]
    summary: str
    themes: List[str] = field(default_factory=list)


@dataclass
class SceneLinks:
    events: List[str] = field(default_factory=list)


@dataclass
class Scene:
    id: str
    t: str
    type: str
    summary: str
    valence: float = 0.5
    arousal: float = 0.5
    salience: float = 0.5
    tags: List[str] = field(default_factory=list)
    links: Dict[str, List[str]] = field(default_factory=lambda: {"events": []})


@dataclass
class TurningPoint:
    id: str
    title: str
    t: str
    summary: str


@dataclass
class PossibleSelf:
    label: str
    prob: float


@dataclass
class Goal:
    id: str
    desc: str
    priority: float


@dataclass
class FutureScripts:
    possible_selves: List[PossibleSelf] = field(default_factory=list)
    goals: List[Goal] = field(default_factory=list)


@dataclass
class NarrativeIdentity:
    chapters: List[Chapter] = field(default_factory=list)
    scenes: List[Scene] = field(default_factory=list)
    turning_points: List[TurningPoint] = field(default_factory=list)
    future_scripts: FutureScripts = field(default_factory=FutureScripts)
    themes: List[str] = field(default_factory=list)


# ===== Self Knowledge =====


@dataclass
class EffectHypothesis:
    target: str
    delta: float
    confidence: float = 0.0


@dataclass
class EventMeta:
    processed: bool = False


@dataclass
class Event:
    id: str
    t: str
    type: str
    summary: str
    valence: float = 0.5
    arousal: float = 0.5
    salience: float = 0.5
    tags: List[str] = field(default_factory=list)
    effects_hypothesis: List[EffectHypothesis] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=lambda: {"processed": False})


@dataclass
class Thought:
    id: str
    t: str
    content: str
    trigger: str = ""


@dataclass
class Insight:
    id: str
    t: str
    content: str
    references: Dict[str, List[str]] = field(default_factory=dict)


@dataclass
class SelfKnowledge:
    behavioral_patterns: Dict[str, int] = field(default_factory=dict)
    autobiographical_events: List[Event] = field(default_factory=list)
    thoughts: List[Thought] = field(default_factory=list)
    insights: List[Insight] = field(default_factory=list)
    commitments: Dict[str, Dict] = field(default_factory=dict)  # Store commitment data


# ===== Metrics / Drift / Meta =====


@dataclass
class Metrics:
    identity_coherence: float = 0.5
    self_consistency: float = 0.5
    drift_velocity: Dict[str, float] = field(default_factory=dict)
    reflection_cadence_days: int = 7
    last_reflection_at: Optional[str] = None
    commitments_open: int = 0
    commitments_closed: int = 0


@dataclass
class Bounds:
    min: float = 0.05
    max: float = 0.95


@dataclass
class DriftConfig:
    maturity_principle: bool = True
    inertia: float = 0.9
    max_delta_per_reflection: float = 0.02
    cooldown_days: int = 7
    event_sensitivity: float = 0.4
    bounds: Bounds = field(default_factory=Bounds)
    locks: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class IdentityChange:
    t: str
    change: str


@dataclass
class MetaCognition:
    times_accessed_self: int = 0
    self_modification_count: int = 0
    identity_evolution: List[IdentityChange] = field(default_factory=list)


# ===== Top-level Model =====


@dataclass
class PersistentMindModel:
    schema_version: int = 1
    inception_moment: str = field(
        default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    core_identity: CoreIdentity = field(default_factory=CoreIdentity)
    personality: Personality = field(default_factory=Personality)
    narrative_identity: NarrativeIdentity = field(default_factory=NarrativeIdentity)
    self_knowledge: SelfKnowledge = field(default_factory=SelfKnowledge)
    metrics: Metrics = field(default_factory=Metrics)
    drift_config: DriftConfig = field(default_factory=DriftConfig)
    meta_cognition: MetaCognition = field(default_factory=MetaCognition)
