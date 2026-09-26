"""Connect the existing two-player OpenSpiel Hanabi adapter to OppoMem."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any

from benchmarks.hanabi import HanabiAdapter

from .agent import ActionChoice, OppoMemAgent
from .memory import E5MedoidNLIRetriever, MemoryBank
from .offline import PromptOpponentModeler
from .segmentation import OnlineDescription, TrainingEpisode, VisibleEvent
from .types import EpisodeEvidence, VisibleView


GAME = "adaptive_hanabi"
TARGET = "partner"
GAME_CONTEXT = (
    "Two-player cooperative Hanabi. The acting player sees only its own legal "
    "observation and completed public actions; a partner's private hand is hidden."
)


class HanabiActionPolicy:
    def __init__(self, llm: Any, *, target_id: str = TARGET):
        self.llm = llm
        self.target_id = target_id

    def choose(self, view: VisibleView, opponent_models, enabled_memories) -> ActionChoice:
        payload = {
            "VISIBLE_OBSERVATION": view.observation,
            "COMPLETED_PUBLIC_HISTORY": list(view.public_history),
            "PARTNER_MODELING": opponent_models.get(self.target_id),
            "RETRIEVED_MEMORY": (dict(enabled_memories[self.target_id])
                                 if self.target_id in enabled_memories else None),
            "LEGAL_ACTION_IDS": list(view.legal_actions),
        }
        answer = self.llm.complete_json(
            stage="hanabi_action",
            system=("Choose one legal Hanabi action ID using only the acting player's "
                    "visible observation. Return JSON with action as a string and "
                    "response_intent as at most 20 words."),
            prompt="DATA:\n" + json.dumps(payload, ensure_ascii=False),
        )
        action, intent = answer.get("action"), answer.get("response_intent", "")
        if action not in view.legal_actions or not isinstance(intent, str):
            raise ValueError("Hanabi policy must return a legal action ID and text intent")
        return ActionChoice(action, intent)


@dataclass(frozen=True)
class PlayedHanabiEpisode:
    evidence: EpisodeEvidence
    training: TrainingEpisode | None
    partner_evidence: EpisodeEvidence | None = None


class OppoMemHanabiPartner:
    """Second OppoMem seat for the paper's same-method Hanabi test self-play."""

    def __init__(self, *, episode_id: str, configuration_id: str,
                 bank: MemoryBank, retriever: E5MedoidNLIRetriever, llm: Any):
        self.episode_id = episode_id + "-partner"
        self.configuration_id = configuration_id
        self.agent = OppoMemAgent(
            game=GAME, actor=HanabiActionPolicy(llm, target_id="self"),
            modeler=PromptOpponentModeler(llm, game_context=GAME_CONTEXT),
            bank=bank, retriever=retriever,
        )
        self.index = 0

    def reset(self, seed: int) -> None:
        self.index = 0
        self.agent.begin_episode(
            episode_id=self.episode_id, configuration_id=self.configuration_id,
            split="test", rng=random.Random(seed),
        )

    def act_with_history(self, observation: str, legal_actions: tuple[int, ...],
                         action_labels: dict[int, str],
                         public_history: tuple[dict[str, Any], ...]) -> int:
        view = VisibleView(
            game=GAME, episode_id=self.episode_id, decision_index=self.index,
            self_id="partner", target_ids=("self",),
            observation={"player_visible_state": observation,
                         "legal_action_labels": action_labels},
            public_history=public_history,
            legal_actions=tuple(str(action) for action in legal_actions),
        )
        choice = self.agent.step(view)
        self.index += 1
        return int(choice.action)

    def act(self, observation: str, legal_actions: tuple[int, ...],
            action_labels: dict[int, str]) -> int:
        raise RuntimeError("Self-play requires an adapter that supplies public history")

    def finish_episode(self, terminal_return: float,
                       public_history: tuple[dict[str, Any], ...]) -> EpisodeEvidence:
        return self.agent.finish_episode(
            terminal_return=terminal_return, public_trajectory=public_history)


def play_episode(*, adapter: HanabiAdapter, episode_id: str, split: str,
                 opponent_id: str, seed: int, bank: MemoryBank,
                 retriever: E5MedoidNLIRetriever, llm: Any,
                 seat: int = 0) -> PlayedHanabiEpisode:
    """Use the existing game engine; no checkpoint or OpenSpiel data is bundled."""
    agent = OppoMemAgent(
        game=GAME, actor=HanabiActionPolicy(llm),
        modeler=PromptOpponentModeler(llm, game_context=GAME_CONTEXT),
        bank=bank, retriever=retriever,
    )
    agent.begin_episode(episode_id=episode_id, configuration_id=opponent_id,
                        split=split, rng=random.Random(seed ^ 0xA11CE))
    events: list[VisibleEvent] = []
    descriptions: list[OnlineDescription] = []
    try:
        frame = adapter.reset(opponent_id=opponent_id, seed=seed, seat=seat,
                              training=(split == "train"))
        index = 0
        while not frame.terminal:
            observation = json.loads(frame.observation)
            history = tuple(dict(row) for row in observation["completed_public_actions"])
            view = VisibleView(
                game=GAME, episode_id=episode_id, decision_index=index,
                self_id="self", target_ids=(TARGET,), observation=observation,
                public_history=history, legal_actions=frame.legal_actions,
            )
            decision = agent.step(view)
            transition = adapter.step(decision.action)
            settled = json.loads(transition.frame.observation)["completed_public_actions"]
            events.append(VisibleEvent(
                index=index, actor="self_and_partner",
                visible_context={"completed_public_actions": history},
                public_event={"new_public_actions": settled[len(history):]},
                focal_action=decision.action,
                observed_reward=transition.reward_delta,
                response_intent=decision.response_intent,
            ))
            if TARGET in decision.opponent_models:
                descriptions.append(OnlineDescription(
                    index, TARGET, decision.opponent_models[TARGET]))
            frame = transition.frame
            index += 1
        trajectory = tuple(dict(row) for row in
                           json.loads(frame.observation)["completed_public_actions"])
        evidence = agent.finish_episode(terminal_return=frame.cumulative_return,
                                        public_trajectory=trajectory)
        training = (TrainingEpisode(
            episode_id=episode_id, game=GAME,
            terminal_return=frame.cumulative_return,
            visible_events=tuple(events), online_descriptions=tuple(descriptions),
        ) if split == "train" else None)
        partner_evidence = None
        if isinstance(getattr(adapter, "partner", None), OppoMemHanabiPartner):
            partner_evidence = adapter.partner.finish_episode(
                frame.cumulative_return, trajectory)
        return PlayedHanabiEpisode(evidence, training, partner_evidence)
    finally:
        adapter.close()
