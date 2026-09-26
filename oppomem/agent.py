"""Online opponent modeling, one response per participant, and legal action selection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping, Protocol

from .activation import EpisodeActivation
from .memory import E5MedoidNLIRetriever, MemoryBank, valid_opponent_modeling
from .types import EpisodeEvidence, RetrievalEvent, VisibleView


class OpponentModeler(Protocol):
    def describe(self, view: VisibleView, target_id: str,
                 previous_description: str | None) -> str: ...


class ActionPolicy(Protocol):
    def choose(self, view: VisibleView, opponent_models: Mapping[str, str],
               enabled_memories: Mapping[str, Mapping[str, str]]) -> str | ActionChoice: ...


@dataclass(frozen=True)
class ActionChoice:
    action: str
    response_intent: str = ""


@dataclass(frozen=True)
class DecisionResult:
    action: str
    response_intent: str
    opponent_models: Mapping[str, str]
    enabled_memories: Mapping[str, Mapping[str, str]]
    retrievals: tuple[RetrievalEvent, ...]


class OppoMemAgent:
    """One game's focal policy; the bank is snapshotted when the game begins."""

    def __init__(self, *, game: str, actor: ActionPolicy, modeler: OpponentModeler,
                 bank: MemoryBank, retriever: E5MedoidNLIRetriever,
                 validate_action: Callable[[VisibleView, str], bool] | None = None):
        self.game = game
        self.actor = actor
        self.modeler = modeler
        self.source_bank = bank
        self.retriever = retriever
        self.validate_action = validate_action
        if modeler is None or bank is None or retriever is None or bank.game != game:
            raise ValueError("OppoMem requires a modeler, same-game bank, and retriever")
        self._episode_id: str | None = None
        self._configuration_id: str | None = None
        self._split: str | None = None
        self._bank: MemoryBank | None = None
        self._activation: EpisodeActivation | None = None
        self._previous: dict[str, str] = {}
        self._retrievals: list[RetrievalEvent] = []
        self._last_index = -1

    def begin_episode(self, *, episode_id: str, configuration_id: str,
                      split: str, rng: random.Random | None = None) -> None:
        if self._episode_id is not None:
            raise RuntimeError("finish the current episode first")
        if split not in {"train", "validation", "test"}:
            raise ValueError("unknown episode split")
        if not episode_id or not configuration_id:
            raise ValueError("episode and configuration IDs are required")
        self._episode_id, self._configuration_id, self._split = episode_id, configuration_id, split
        self._bank = self.source_bank.snapshot()
        self._activation = EpisodeActivation(training=(split == "train"), rng=rng)
        self._previous, self._retrievals, self._last_index = {}, [], -1

    def step(self, view: VisibleView) -> DecisionResult:
        if self._episode_id is None or self._activation is None:
            raise RuntimeError("begin_episode must be called first")
        if view.episode_id != self._episode_id or view.game != self.game:
            raise ValueError("decision view belongs to another episode or game")
        if view.decision_index <= self._last_index:
            raise ValueError("decision indices must increase")
        checkpoint = self._activation.checkpoint()
        staged_previous = dict(self._previous)
        models: dict[str, str] = {}
        memories: dict[str, dict[str, str]] = {}
        retrieved_now: list[RetrievalEvent] = []
        try:
            for target_id in view.target_ids:
                description = self.modeler.describe(
                    view, target_id, staged_previous.get(target_id)
                ).strip()
                # A weak or action-specific key is not substituted with an oracle
                # label. It can be shown as an uncertain model but cannot index memory.
                if not description:
                    continue
                models[target_id] = description
                staged_previous[target_id] = description
                if valid_opponent_modeling(
                    description, extra_forbidden=view.legal_actions
                ):
                    assert self._bank is not None and self.retriever is not None
                    match = self.retriever.retrieve(description, self._bank)
                    if match is not None:
                        enabled = self._activation.enabled(match.item.memory_id)
                        advice = match.item.public_payload() if enabled else None
                        if advice is not None:
                            memories[target_id] = advice
                        event = RetrievalEvent(
                            view.decision_index, target_id, match.item.memory_id,
                            match.item.revision, enabled,
                            match.item.response_strategy if enabled else None,
                        )
                        retrieved_now.append(event)
            visible_memories = MappingProxyType({
                target_id: MappingProxyType(payload)
                for target_id, payload in memories.items()
            })
            raw_choice = self.actor.choose(
                view, MappingProxyType(models), visible_memories
            )
            choice = raw_choice if isinstance(raw_choice, ActionChoice) else ActionChoice(raw_choice)
            action = choice.action
            if not isinstance(action, str) or not action.strip():
                raise ValueError("action policy must return a nonempty action")
            if not isinstance(choice.response_intent, str) or len(choice.response_intent.split()) > 20:
                raise ValueError("response intent must be text of at most 20 words")
            if view.legal_actions and action not in view.legal_actions:
                raise ValueError("action is not in the legal-action list")
            if self.validate_action is not None and not self.validate_action(view, action):
                raise ValueError("action fails the game adapter's validation")
        except Exception:
            self._activation.restore(checkpoint)
            raise
        self._previous = staged_previous
        self._retrievals.extend(retrieved_now)
        self._last_index = view.decision_index
        return DecisionResult(action, choice.response_intent, models, memories, tuple(retrieved_now))

    def finish_episode(self, *, terminal_return: float,
                       public_trajectory: tuple[Mapping, ...]) -> EpisodeEvidence:
        if self._episode_id is None or self._configuration_id is None or self._split is None:
            raise RuntimeError("no active episode")
        evidence = EpisodeEvidence(
            episode_id=self._episode_id,
            game=self.game,
            configuration_id=self._configuration_id,
            terminal_return=float(terminal_return),
            retrievals=tuple(self._retrievals),
            public_trajectory=public_trajectory,
            split=self._split,
        )
        evidence.activation_by_memory()
        self._episode_id = self._configuration_id = self._split = None
        self._bank = self._activation = None
        return evidence
