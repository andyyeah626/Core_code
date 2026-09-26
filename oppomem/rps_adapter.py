"""Repeated RPS bridge. An injected game adapter may supply a held-out LLM rival."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from typing import Any, Mapping

from benchmarks.roshambo import ACTIONS, INFERENCE_CONTEXT, RoshamboAdapter

from .agent import ActionChoice, OppoMemAgent
from .memory import E5MedoidNLIRetriever, MemoryBank
from .offline import PromptOpponentModeler
from .segmentation import OnlineDescription, TrainingEpisode, VisibleEvent
from .types import EpisodeEvidence, VisibleView


GAME = "open_spiel_roshambo"
TARGET = "opponent"


class RPSActionPolicy:
    def __init__(self, llm: Any):
        self.llm = llm

    def choose(self, view: VisibleView, opponent_models: Mapping[str, str],
               enabled_memories: Mapping[str, Mapping[str, str]]) -> ActionChoice:
        payload = {
            "COMPLETED_ROUNDS": list(view.public_history),
            "ROUND": view.decision_index + 1,
            "HORIZON": view.observation["round"] + view.observation["remaining_rounds"],
            "OPPONENT_MODELING": opponent_models.get(TARGET),
            "RETRIEVED_MEMORY": (dict(enabled_memories[TARGET])
                                 if TARGET in enabled_memories else None),
        }
        result = self.llm.complete_json(
            stage="rps_action",
            system=(INFERENCE_CONTEXT + " Choose one legal action. Return JSON with "
                    "action and response_intent (at most 20 words)."),
            prompt="DATA:\n" + json.dumps(payload, ensure_ascii=False),
        )
        action, intent = result.get("action"), result.get("response_intent", "")
        if action not in ACTIONS or not isinstance(intent, str):
            raise ValueError("RPS policy must return Rock, Paper, or Scissors")
        return ActionChoice(action, intent)


@dataclass(frozen=True)
class PlayedRPSEpisode:
    evidence: EpisodeEvidence
    training: TrainingEpisode | None


def play_episode(*, episode_id: str, split: str, opponent_id: str, seed: int,
                 bank: MemoryBank, retriever: E5MedoidNLIRetriever, llm: Any,
                 horizon: int, adapter: Any | None = None,
                 seat: int = 0) -> PlayedRPSEpisode:
    """Use Roshambo by default, or a compatible player-visible adapter."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    game = adapter if adapter is not None else RoshamboAdapter(horizon=horizon)
    agent = OppoMemAgent(
        game=GAME, actor=RPSActionPolicy(llm),
        modeler=PromptOpponentModeler(llm, game_context=INFERENCE_CONTEXT),
        bank=bank, retriever=retriever,
    )
    agent.begin_episode(episode_id=episode_id, configuration_id=opponent_id,
                        split=split, rng=random.Random(seed ^ 0x5A17))
    events: list[VisibleEvent] = []
    descriptions: list[OnlineDescription] = []
    try:
        frame = game.reset(opponent_id=opponent_id, seed=seed, seat=seat,
                           training=(split == "train"))
        index = 0
        while not frame.terminal:
            observation = json.loads(frame.observation)
            history = tuple(dict(row) for row in observation["completed_joint_actions"])
            view = VisibleView(
                game=GAME, episode_id=episode_id, decision_index=index,
                self_id="self", target_ids=(TARGET,), observation=observation,
                public_history=history, legal_actions=frame.legal_actions,
            )
            decision = agent.step(view)
            transition = game.step(decision.action)
            settled = json.loads(transition.frame.observation)["completed_joint_actions"][-1]
            events.append(VisibleEvent(
                index=index, actor="both",
                visible_context={"completed_joint_actions": history},
                public_event=dict(settled), focal_action=decision.action,
                observed_reward=transition.reward_delta,
                response_intent=decision.response_intent,
            ))
            if TARGET in decision.opponent_models:
                descriptions.append(OnlineDescription(
                    index, TARGET, decision.opponent_models[TARGET]))
            frame = transition.frame
            index += 1
        if index != horizon:
            raise ValueError(f"RPS episode ended after {index} rounds, expected {horizon}")
        trajectory = tuple(dict(row.public_event) for row in events)
        evidence = agent.finish_episode(terminal_return=frame.cumulative_return,
                                        public_trajectory=trajectory)
        training = (TrainingEpisode(
            episode_id=episode_id, game=GAME,
            terminal_return=frame.cumulative_return,
            visible_events=tuple(events), online_descriptions=tuple(descriptions),
        ) if split == "train" else None)
        return PlayedRPSEpisode(evidence, training)
    finally:
        game.close()
