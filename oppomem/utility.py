"""Per-configuration, per-game return comparison for one retrieved memory."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean
from typing import Callable, Mapping, Sequence

from .types import EpisodeEvidence


@dataclass(frozen=True)
class ConfigurationSummary:
    configuration_id: str
    enabled_episode_count: int
    disabled_episode_count: int
    mean_enabled_terminal_return: float | None
    mean_disabled_terminal_return: float | None
    difference: float | None
    configuration_weight: float
    support_sufficient: bool


@dataclass(frozen=True)
class UtilityEstimate:
    memory_id: str
    memory_revision: int
    configurations: tuple[ConfigurationSummary, ...]
    weighted_difference: float | None
    eligible_episode_ids: tuple[str, ...]


def estimate_memory_utility(
    episodes: Sequence[EpisodeEvidence], *, memory_id: str, memory_revision: int,
    configuration_weights: Mapping[str, float],
    sufficient_support: Callable[[int, int], bool],
) -> UtilityEstimate:
    """Count an episode once for a memory, even if retrieval occurs many times.

    Section 4.3 defines the enabled-minus-disabled mean within each opponent
    configuration. A weighted aggregate is only reported when the caller supplies
    configuration weights. The paper leaves support and weights to run settings.
    """

    if not memory_id or memory_revision < 0:
        raise ValueError("memory identity and revision are required")
    if len({ep.episode_id for ep in episodes}) != len(episodes):
        raise ValueError("duplicate episode IDs would double-count returns")
    grouped: dict[str, dict[bool, list[tuple[str, float]]]] = {}
    eligible_ids: list[str] = []
    for episode in episodes:
        if episode.split != "train":
            raise ValueError("validation/test returns cannot update memory utility")
        episode.activation_by_memory()
        matches = [event for event in episode.retrievals
                   if event.memory_id == memory_id and event.memory_revision == memory_revision]
        if not matches:
            continue
        enabled = matches[0].enabled
        if any(event.enabled != enabled for event in matches):
            raise ValueError("activation changed within a game")
        grouped.setdefault(episode.configuration_id, {True: [], False: []})[enabled].append(
            (episode.episode_id, episode.terminal_return)
        )
        eligible_ids.append(episode.episode_id)
    summaries: list[ConfigurationSummary] = []
    for configuration_id, arms in sorted(grouped.items()):
        on, off = arms[True], arms[False]
        on_mean = mean(value for _, value in on) if on else None
        off_mean = mean(value for _, value in off) if off else None
        difference = on_mean - off_mean if on_mean is not None and off_mean is not None else None
        if difference is not None and not math.isfinite(difference):
            raise ValueError("configuration utility difference must be finite")
        if difference is not None and configuration_id not in configuration_weights:
            raise ValueError(f"missing configuration weight: {configuration_id}")
        weight = float(configuration_weights.get(configuration_id, 0.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("configuration weights must be finite and nonnegative")
        sufficient = difference is not None and bool(sufficient_support(len(on), len(off)))
        summaries.append(ConfigurationSummary(
            configuration_id, len(on), len(off), on_mean, off_mean,
            difference, weight, sufficient,
        ))
    supported = [row for row in summaries if row.support_sufficient and row.configuration_weight > 0]
    total_weight = sum(row.configuration_weight for row in supported)
    weighted = (sum(row.configuration_weight * row.difference for row in supported) / total_weight
                if total_weight else None)
    if weighted is not None and not math.isfinite(weighted):
        raise ValueError("weighted memory utility must be finite")
    return UtilityEstimate(memory_id, memory_revision, tuple(summaries),
                           weighted, tuple(sorted(eligible_ids)))
