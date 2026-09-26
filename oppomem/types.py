"""Public-view records and the opponent-model / response-memory pair."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class VisibleView:
    """The acting player's legal view at one decision, supplied by a game adapter.

    The adapter must never put another player's private information or evaluator
    fields into observation, public_history, or own_private.
    """

    game: str
    episode_id: str
    decision_index: int
    self_id: str
    target_ids: tuple[str, ...]
    observation: Mapping[str, Any] | str
    public_history: tuple[Mapping[str, Any], ...]
    legal_actions: tuple[str, ...]
    own_private: Mapping[str, Any] = field(default_factory=dict)
    phase: str | None = None

    def __post_init__(self) -> None:
        if self.self_id in self.target_ids or len(set(self.target_ids)) != len(self.target_ids):
            raise ValueError("target_ids must be distinct other participants")
        if self.decision_index < 0:
            raise ValueError("decision_index must be nonnegative")


@dataclass(frozen=True)
class TrainingRecord:
    """One completed, player-visible training segment used to construct a key."""

    record_id: str
    source_episode_id: str
    game: str
    target_id: str
    opponent_modeling: str
    event_indices: tuple[int, ...]
    visible_events: tuple[Mapping[str, Any], ...]
    focal_actions: tuple[str, ...]
    observed_rewards: tuple[float, ...]
    terminal_return: float
    forbidden_key_terms: tuple[str, ...] = ()
    response_intents: tuple[str, ...] = ()
    split: str = "train"

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("only training records may construct or revise memory")
        if not self.record_id or not self.source_episode_id or not self.event_indices:
            raise ValueError("a training record needs IDs and supporting events")
        if len(set(self.event_indices)) != len(self.event_indices) or self.event_indices != tuple(sorted(self.event_indices)):
            raise ValueError("supporting event indices must be unique and ordered")
        count = len(self.event_indices)
        if any(len(rows) != count for rows in
               (self.visible_events, self.focal_actions, self.observed_rewards)):
            raise ValueError("training record evidence lengths must match")
        if self.response_intents and len(self.response_intents) != count:
            raise ValueError("response intent count must match supporting events")
        if not math.isfinite(self.terminal_return) or any(
            not math.isfinite(value) for value in self.observed_rewards
        ):
            raise ValueError("training record returns and rewards must be finite")


@dataclass(frozen=True)
class MemoryItem:
    memory_id: str
    opponent_modeling: str
    response_strategy: str
    member_descriptions: tuple[str, ...]
    medoid_vector: tuple[float, ...]
    supporting_record_ids: tuple[str, ...]
    revision: int = 0

    def __post_init__(self) -> None:
        if not self.memory_id or self.revision < 0:
            raise ValueError("memory ID and nonnegative revision are required")
        if (not self.member_descriptions or
                len(self.member_descriptions) != len(self.supporting_record_ids) or
                self.opponent_modeling not in self.member_descriptions):
            raise ValueError("memory key must be an actual supported member")
        if not self.medoid_vector or any(not math.isfinite(x) for x in self.medoid_vector):
            raise ValueError("memory medoid vector must be finite and nonempty")

    def public_payload(self) -> dict[str, str]:
        return {
            "opponent_modeling": self.opponent_modeling,
            "response_strategy": self.response_strategy,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MemoryItem":
        return cls(
            memory_id=str(raw["memory_id"]),
            opponent_modeling=str(raw["opponent_modeling"]),
            response_strategy=str(raw["response_strategy"]),
            member_descriptions=tuple(raw["member_descriptions"]),
            medoid_vector=tuple(float(x) for x in raw["medoid_vector"]),
            supporting_record_ids=tuple(raw["supporting_record_ids"]),
            revision=int(raw.get("revision", 0)),
        )


@dataclass(frozen=True)
class MemoryMatch:
    item: MemoryItem
    e5_cosine: float
    nli_entailment: float


@dataclass(frozen=True)
class RetrievalEvent:
    decision_index: int
    target_id: str
    memory_id: str
    memory_revision: int
    enabled: bool
    advice_shown: str | None

    def __post_init__(self) -> None:
        if self.enabled != (self.advice_shown is not None):
            raise ValueError("advice_shown must match the activation decision")


@dataclass(frozen=True)
class EpisodeEvidence:
    """One completed game; each retrieved memory gets at most one return sample."""

    episode_id: str
    game: str
    configuration_id: str
    terminal_return: float
    retrievals: tuple[RetrievalEvent, ...]
    public_trajectory: tuple[Mapping[str, Any], ...]
    split: str = "train"

    def __post_init__(self) -> None:
        if not self.episode_id or not self.configuration_id:
            raise ValueError("episode and configuration IDs are required")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("unknown episode split")
        if not math.isfinite(self.terminal_return):
            raise ValueError("terminal return must be finite")

    def activation_by_memory(self) -> dict[str, bool]:
        result: dict[str, bool] = {}
        versions: dict[str, int] = {}
        for event in self.retrievals:
            if event.memory_revision < 0:
                raise ValueError("memory revision must be nonnegative")
            previous_version = versions.setdefault(event.memory_id, event.memory_revision)
            if previous_version != event.memory_revision:
                raise ValueError("a memory revision changed within one episode")
            previous = result.setdefault(event.memory_id, event.enabled)
            if previous != event.enabled:
                raise ValueError("a memory's activation changed within one episode")
        return result
