"""LLM interfaces for method-wide modeling, segmentation, and reflection."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol, Sequence

from . import method_prompts
from .evolution import Revision
from .segmentation import SemanticWindow, TrainingEpisode
from .types import EpisodeEvidence, MemoryItem, TrainingRecord, VisibleView
from .utility import UtilityEstimate


class LLMClient(Protocol):
    def complete_json(self, *, stage: str, system: str, prompt: str) -> dict[str, Any]: ...


def _text_field(result: Mapping[str, Any], key: str) -> str:
    value = result.get(key)
    if not isinstance(value, str):
        raise ValueError(f"LLM response must contain text field {key}")
    return value.strip()


class PromptOpponentModeler:
    def __init__(self, llm: LLMClient, *, game_context: str):
        self.llm, self.game_context = llm, game_context

    def describe(self, view: VisibleView, target_id: str,
                 previous_description: str | None) -> str:
        if not view.public_history:
            return ""
        system, user = method_prompts.modeling_prompt(
            game_context=self.game_context, view=view, target_id=target_id,
            previous_description=previous_description,
        )
        result = self.llm.complete_json(stage="opponent_modeling", system=system, prompt=user)
        return _text_field(result, "opponent_modeling")


class OfflineSegmenter:
    def __init__(self, llm: LLMClient, *, game_context: str, target_id: str):
        self.llm, self.game_context, self.target_id = llm, game_context, target_id

    def segment(self, episode: TrainingEpisode) -> Sequence[SemanticWindow]:
        chain = [{"event_index": row.event_index, "target_id": row.target_id,
                  "opponent_modeling": row.opponent_modeling}
                 for row in episode.online_descriptions if row.target_id == self.target_id]
        events = [{"index": event.index, "actor": event.actor,
                   "visible_context": event.visible_context,
                   "public_event": event.public_event,
                   "focal_action": event.focal_action,
                   "observed_reward": event.observed_reward}
                  for event in episode.visible_events]
        system, user = method_prompts.segmentation_prompt(
            game_context=self.game_context, modeling_chain=chain,
            completed_visible_events=events,
        )
        raw = self.llm.complete_json(stage="memory_segmentation", system=system, prompt=user)
        if not isinstance(raw.get("windows"), list):
            raise ValueError("segmentation must return windows")
        windows = []
        for row in raw["windows"]:
            if not isinstance(row, dict) or set(row) != {"opponent_modeling", "round_ranges"}:
                raise ValueError("window must contain the two method fields")
            ranges = row["round_ranges"]
            if not isinstance(ranges, list) or any(
                not isinstance(pair, list) or len(pair) != 2 or
                any(type(index) is not int for index in pair)
                for pair in ranges
            ):
                raise ValueError("round_ranges must contain integer start/end pairs")
            windows.append(SemanticWindow(
                target_id=self.target_id,
                opponent_modeling=_text_field(row, "opponent_modeling"),
                round_ranges=tuple(tuple(pair) for pair in ranges),
            ))
        return windows


class OfflineReflector:
    def __init__(self, llm: LLMClient, *, game_context: str):
        self.llm, self.game_context = llm, game_context

    def construct(self, game: str, canonical_modeling: str,
                  records: Sequence[TrainingRecord]) -> str:
        evidence = [{"record_id": row.record_id, "target_id": row.target_id,
                     "event_indices": row.event_indices,
                     "visible_events": row.visible_events,
                     "focal_actions": row.focal_actions,
                     "response_intents": row.response_intents,
                     "observed_rewards": row.observed_rewards,
                     "terminal_return": row.terminal_return}
                    for row in records]
        system, user = method_prompts.construction_prompt(
            game_context=self.game_context, canonical_modeling=canonical_modeling,
            assigned_visible_records=evidence,
        )
        result = self.llm.complete_json(stage="memory_construction", system=system, prompt=user)
        return _text_field(result, "response_strategy")


class OfflineRevisionModel:
    """M5 response revision; an optional selector supplies evidence-bound key updates."""

    def __init__(self, llm: LLMClient, *, game_context: str,
                 key_record_selector: Callable[[MemoryItem, UtilityEstimate,
                                                Sequence[TrainingRecord]], Sequence[str]] | None = None):
        self.llm, self.game_context = llm, game_context
        self.key_record_selector = key_record_selector

    def revise(self, item: MemoryItem, estimate: UtilityEstimate,
               episodes: Sequence[EpisodeEvidence],
               records: Sequence[TrainingRecord]) -> Revision:
        summaries = [{
            "configuration_id": row.configuration_id,
            "enabled_episode_count": row.enabled_episode_count,
            "disabled_episode_count": row.disabled_episode_count,
            "mean_enabled_terminal_return": row.mean_enabled_terminal_return,
            "mean_disabled_terminal_return": row.mean_disabled_terminal_return,
            "configuration_weight": row.configuration_weight,
            "support_sufficient": row.support_sufficient,
        } for row in estimate.configurations]
        evidence = [{
            "episode_id": episode.episode_id,
            "configuration_id": episode.configuration_id,
            "memory_enabled": episode.activation_by_memory()[item.memory_id],
            "terminal_return": episode.terminal_return,
            "eligible_decision_indices": sorted({event.decision_index for event in episode.retrievals
                                                  if event.memory_id == item.memory_id and
                                                  event.memory_revision == item.revision}),
            "public_trajectory": list(episode.public_trajectory),
        } for episode in episodes]
        system, user = method_prompts.revision_prompt(
            game_context=self.game_context, canonical_modeling=item.opponent_modeling,
            prior_response=item.response_strategy, configuration_summaries=summaries,
            eligible_episode_evidence=evidence,
        )
        result = self.llm.complete_json(stage="memory_revision", system=system, prompt=user)
        selected = (tuple(self.key_record_selector(item, estimate, records))
                    if self.key_record_selector is not None else ())
        return Revision(response_strategy=_text_field(result, "response_strategy"),
                        additional_record_ids=selected)
