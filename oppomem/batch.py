"""Freeze a bank for one training batch, then evolve it from completed games."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from .evolution import EvolutionDecision, EvolutionPolicy, RevisionModel, evolve_bank
from .memory import MemoryBank, MemoryBuilder
from .types import EpisodeEvidence, TrainingRecord
from .utility import UtilityEstimate, estimate_memory_utility


@dataclass(frozen=True)
class BatchResult:
    next_bank: MemoryBank
    utility: Mapping[str, UtilityEstimate]
    decisions: Mapping[str, EvolutionDecision]
    excluded_record_ids: tuple[str, ...]


class TrainingBatch:
    """Keep a private frozen bank and accept only completed training evidence."""

    def __init__(self, bank: MemoryBank):
        self._frozen_bank = bank.snapshot()
        self._episodes: list[EpisodeEvidence] = []
        self._records: list[TrainingRecord] = []
        self._record_ids: set[str] = set()
        self._closed = False

    @property
    def frozen_bank(self) -> MemoryBank:
        return self._frozen_bank.snapshot()

    @property
    def episodes(self) -> tuple[EpisodeEvidence, ...]:
        return tuple(self._episodes)

    @property
    def records(self) -> tuple[TrainingRecord, ...]:
        return tuple(self._records)

    def add_episode(self, episode: EpisodeEvidence,
                    records: Sequence[TrainingRecord] = ()) -> None:
        if self._closed:
            raise RuntimeError("batch already finished")
        if episode.split != "train" or episode.game != self._frozen_bank.game:
            raise ValueError("only same-game training episodes belong in a batch")
        if any(existing.episode_id == episode.episode_id for existing in self._episodes):
            raise ValueError("duplicate episode")
        episode.activation_by_memory()
        for event in episode.retrievals:
            item = self._frozen_bank.items.get(event.memory_id)
            if (item is None or item.revision != event.memory_revision or
                    event.advice_shown != (item.response_strategy if event.enabled else None)):
                raise ValueError("retrieval does not match the frozen batch bank")
        if any(row.game != episode.game or row.split != "train" or
               row.source_episode_id != episode.episode_id or
               row.terminal_return != episode.terminal_return for row in records):
            raise ValueError("records must come from this same-game training episode")
        new_ids = [row.record_id for row in records]
        if len(set(new_ids)) != len(new_ids) or any(record_id in self._record_ids
                                                   for record_id in new_ids):
            raise ValueError("duplicate batch record IDs")
        self._episodes.append(episode)
        self._records.extend(records)
        self._record_ids.update(new_ids)

    def finish(self, *, configuration_weights: Mapping[str, float],
               sufficient_support: Callable[[int, int], bool],
               prior_supported_estimates: dict[tuple[str, int], Sequence[UtilityEstimate]],
               policy: EvolutionPolicy, revision_model: RevisionModel,
               builder: MemoryBuilder) -> BatchResult:
        if self._closed:
            raise RuntimeError("batch already finished")
        estimates = {
            item.memory_id: estimate_memory_utility(
                self._episodes, memory_id=item.memory_id,
                memory_revision=item.revision,
                configuration_weights=configuration_weights,
                sufficient_support=sufficient_support,
            ) for item in self._frozen_bank.items.values()
        }
        revised, decisions = evolve_bank(
            self._frozen_bank, estimates=tuple(estimates.values()),
            training_episodes=self._episodes, training_records=self._records,
            prior_supported_estimates=prior_supported_estimates,
            policy=policy, revision_model=revision_model,
            encoder=builder.encoder,
        )
        next_bank, excluded = builder.build(
            self._records, game=self._frozen_bank.game, previous=revised
        )
        self._closed = True
        return BatchResult(next_bank, estimates, decisions, excluded)
