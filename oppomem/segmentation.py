"""Offline coarse semantic windows over completed player-visible training events."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from .memory import valid_opponent_modeling
from .types import TrainingRecord


@dataclass(frozen=True)
class VisibleEvent:
    index: int
    actor: str
    visible_context: Mapping[str, Any]
    public_event: Mapping[str, Any]
    focal_action: str | None
    observed_reward: float
    response_intent: str = ""

    def __post_init__(self) -> None:
        if self.index < 0 or not math.isfinite(self.observed_reward):
            raise ValueError("visible event index and reward must be valid")


@dataclass(frozen=True)
class OnlineDescription:
    event_index: int
    target_id: str
    opponent_modeling: str


@dataclass(frozen=True)
class TrainingEpisode:
    episode_id: str
    game: str
    terminal_return: float
    visible_events: tuple[VisibleEvent, ...]
    online_descriptions: tuple[OnlineDescription, ...]
    split: str = "train"

    def __post_init__(self) -> None:
        if self.split != "train":
            raise ValueError("validation/test episodes cannot construct memory")
        if not self.episode_id or not math.isfinite(self.terminal_return):
            raise ValueError("training episode ID and finite return are required")
        indices = [event.index for event in self.visible_events]
        if indices != sorted(set(indices)):
            raise ValueError("visible events need unique, chronological indices")


@dataclass(frozen=True)
class SemanticWindow:
    target_id: str
    opponent_modeling: str
    round_ranges: tuple[tuple[int, int], ...]


class Segmenter(Protocol):
    """Offline M1/M2 assistant; it sees only the focal visible trace."""

    def segment(self, episode: TrainingEpisode) -> Sequence[SemanticWindow]: ...


def training_records(episode: TrainingEpisode, windows: Sequence[SemanticWindow],
                     *, forbidden_key_terms: Sequence[str] = ()) -> tuple[TrainingRecord, ...]:
    """Normalize overlapping/gapped windows without inventing supporting events."""

    events = {event.index: event for event in episode.visible_events}
    supported = {(row.event_index, row.target_id) for row in episode.online_descriptions
                 if valid_opponent_modeling(row.opponent_modeling,
                                            extra_forbidden=forbidden_key_terms)}
    rows: list[TrainingRecord] = []
    for ordinal, window in enumerate(windows):
        if not window.round_ranges or not valid_opponent_modeling(
            window.opponent_modeling,
            extra_forbidden=forbidden_key_terms,
        ):
            continue
        selected: set[int] = set()
        for start, end in window.round_ranges:
            if start > end:
                raise ValueError("window ranges must be ordered")
            if any(start <= index <= end and target == window.target_id
                   for index, target in supported):
                selected.update(index for index in events if start <= index <= end)
        indices = tuple(sorted(selected))
        if not indices:
            continue
        chosen = [events[index] for index in indices]
        rows.append(TrainingRecord(
            record_id=f"{episode.episode_id}:window:{ordinal}",
            source_episode_id=episode.episode_id,
            game=episode.game,
            target_id=window.target_id,
            opponent_modeling=window.opponent_modeling,
            event_indices=indices,
            visible_events=tuple({
                "index": event.index, "actor": event.actor,
                "visible_context": event.visible_context,
                "public_event": event.public_event,
            } for event in chosen),
            focal_actions=tuple(event.focal_action or "" for event in chosen),
            observed_rewards=tuple(event.observed_reward for event in chosen),
            terminal_return=episode.terminal_return,
            forbidden_key_terms=tuple(forbidden_key_terms),
            response_intents=tuple(event.response_intent for event in chosen),
        ))
    return tuple(rows)
