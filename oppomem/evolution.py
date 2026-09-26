"""Apply supported keep/revise/delete decisions only between training batches."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from enum import Enum
from typing import Any, Protocol, Sequence

from .memory import MemoryBank, valid_opponent_modeling, valid_response_strategy
from .semantics import E5EncoderLike
from .types import EpisodeEvidence, MemoryItem, TrainingRecord
from .utility import UtilityEstimate


class EvolutionDecision(str, Enum):
    UNCHANGED = "unchanged"
    KEEP = "keep"
    REVISE = "revise"
    DELETE = "delete"


@dataclass(frozen=True)
class Revision:
    response_strategy: str
    # A key update can only select fresh, eligible training evidence by ID.
    additional_record_ids: tuple[str, ...] = ()


class EvolutionPolicy(Protocol):
    """Run-specific rule for support, mixed evidence and persistent harm.

    The manuscript does not give numerical decision cutoffs. A policy must
    choose DELETE only for persistently unhelpful guidance using supported history.
    """

    def choose(self, item: MemoryItem, estimate: UtilityEstimate,
               prior_supported_estimates: Sequence[UtilityEstimate]) -> EvolutionDecision: ...


class LLMReviewPolicy:
    """Evidence-based keep/revise/delete review without preset numeric cutoffs."""

    def __init__(self, llm: Any):
        self.llm = llm

    def choose(self, item: MemoryItem, estimate: UtilityEstimate,
               prior_supported_estimates: Sequence[UtilityEstimate]) -> EvolutionDecision:
        payload = {
            "OPPONENT_MODELING": item.opponent_modeling,
            "RESPONSE_STRATEGY": item.response_strategy,
            "CONFIGURATION_SUMMARIES": [asdict(row) for row in estimate.configurations],
            "PRIOR_SUPPORTED_BATCHES": [
                {"configurations": [asdict(row) for row in prior.configurations]}
                for prior in prior_supported_estimates
            ],
        }
        result = self.llm.complete_json(
            stage="memory_review",
            system=("Review a strategic memory using only completed training-game "
                    "evidence. Compare enabled and disabled returns within each "
                    "opponent configuration. Keep when evidence is weak or mixed; "
                    "revise only when the comparison and visible records support "
                    "a change. Delete only for persistently unhelpful guidance "
                    "across supported batches. Return JSON with decision equal "
                    "to keep, revise, or delete."),
            prompt="DATA:\n" + json.dumps(payload, ensure_ascii=False),
        )
        choice = EvolutionDecision(result.get("decision"))
        if choice is EvolutionDecision.DELETE and not prior_supported_estimates:
            return EvolutionDecision.KEEP
        return choice


class RevisionModel(Protocol):
    """Offline reflection assistant; receives no held-out episodes."""

    def revise(self, item: MemoryItem, estimate: UtilityEstimate,
               episodes: Sequence[EpisodeEvidence],
               records: Sequence[TrainingRecord]) -> Revision: ...


def _medoid_description(descriptions: tuple[str, ...], encoder: E5EncoderLike) -> tuple[str, tuple[float, ...]]:
    from .semantics import cosine

    vectors = encoder.encode_passages(descriptions)
    index = max(range(len(descriptions)), key=lambda i: (
        sum(cosine(vectors[i], vectors[j]) for j in range(len(descriptions))), -i
    ))
    return descriptions[index], tuple(float(x) for x in vectors[index])


def evolve_bank(
    bank: MemoryBank, *, estimates: Sequence[UtilityEstimate],
    training_episodes: Sequence[EpisodeEvidence],
    training_records: Sequence[TrainingRecord],
    prior_supported_estimates: dict[tuple[str, int], Sequence[UtilityEstimate]],
    policy: EvolutionPolicy, revision_model: RevisionModel,
    encoder: E5EncoderLike,
) -> tuple[MemoryBank, dict[str, EvolutionDecision]]:
    """Return the next bank; no mutation or revision occurs during an active batch.

    Unsupported memories remain unchanged. The supplied policy decides when
    supported history constitutes persistent harm. A revised entry increments
    its revision so old returns are never used as its new utility.
    """

    if bank.encoder_model != encoder.model_name:
        raise ValueError("encoder differs from frozen bank")
    if any(ep.split != "train" or ep.game != bank.game for ep in training_episodes):
        raise ValueError("only same-game training episodes may evolve memory")
    episode_ids = {ep.episode_id for ep in training_episodes}
    if len(episode_ids) != len(training_episodes):
        raise ValueError("duplicate training episodes")
    episode_by_id = {ep.episode_id: ep for ep in training_episodes}
    if any(row.game != bank.game or row.source_episode_id not in episode_ids
           or row.terminal_return != episode_by_id[row.source_episode_id].terminal_return
           for row in training_records):
        raise ValueError("revision records must come from supplied training episodes")
    if len({row.record_id for row in training_records}) != len(training_records):
        raise ValueError("duplicate revision record IDs")
    by_id = {estimate.memory_id: estimate for estimate in estimates}
    if len(by_id) != len(estimates):
        raise ValueError("duplicate utility estimates")
    updated = bank.snapshot()
    updated.batch_index += 1
    decisions: dict[str, EvolutionDecision] = {}
    for memory_id, item in sorted(bank.items.items()):
        estimate = by_id.get(memory_id)
        if estimate is None or estimate.memory_revision != item.revision:
            decisions[memory_id] = EvolutionDecision.UNCHANGED
            continue
        supported = [row for row in estimate.configurations
                     if row.support_sufficient and row.configuration_weight > 0]
        if not supported or estimate.weighted_difference is None:
            decisions[memory_id] = EvolutionDecision.UNCHANGED
            continue
        history = tuple(prior_supported_estimates.get((memory_id, item.revision), ()))
        choice = EvolutionDecision(policy.choose(item, estimate, history))
        if choice is EvolutionDecision.DELETE:
            del updated.items[memory_id]
        elif choice is EvolutionDecision.REVISE:
            eligible_ids = set(estimate.eligible_episode_ids)
            relevant = [ep for ep in training_episodes if ep.episode_id in eligible_ids]
            eligible_records = [row for row in training_records
                                if row.source_episode_id in eligible_ids]
            proposal = revision_model.revise(item, estimate, relevant, eligible_records)
            if not valid_response_strategy(proposal.response_strategy):
                raise ValueError("revised response strategy must be nonempty and under 40 words")
            record_lookup = {row.record_id: row for row in eligible_records}
            selected_ids = proposal.additional_record_ids
            if (len(set(selected_ids)) != len(selected_ids) or
                    any(record_id not in record_lookup or record_id in item.supporting_record_ids
                        for record_id in selected_ids)):
                raise ValueError("key revision may only add unused eligible training records")
            extra = tuple(record_lookup[record_id] for record_id in selected_ids)
            if proposal.response_strategy.strip() == item.response_strategy and not extra:
                decisions[memory_id] = EvolutionDecision.KEEP
                continue
            if any(not valid_opponent_modeling(row.opponent_modeling,
                                               extra_forbidden=(*row.forbidden_key_terms,
                                                                *row.focal_actions))
                   for row in extra):
                raise ValueError("key revision contains a nonabstract description")
            descriptions = item.member_descriptions + tuple(row.opponent_modeling for row in extra)
            if any(not valid_opponent_modeling(text) for text in descriptions):
                raise ValueError("revised key contains a nonabstract or unsupported description")
            canonical, vector = _medoid_description(descriptions, encoder)
            updated.items[memory_id] = replace(
                item, opponent_modeling=canonical,
                response_strategy=proposal.response_strategy.strip(),
                member_descriptions=descriptions,
                medoid_vector=vector,
                supporting_record_ids=item.supporting_record_ids + selected_ids,
                revision=item.revision + 1,
            )
        decisions[memory_id] = choice
    return updated, decisions
