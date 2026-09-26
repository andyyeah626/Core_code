"""Bridge the existing Axelrod-Python adapter to the OppoMem method core."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from benchmarks.axelrod_ipd import ACTIONS, INFERENCE_CONTEXT, AxelrodAdapter

from .agent import ActionChoice, OppoMemAgent
from .memory import E5MedoidNLIRetriever, MemoryBank
from .offline import PromptOpponentModeler
from .segmentation import OnlineDescription, TrainingEpisode, VisibleEvent
from .types import EpisodeEvidence, VisibleView


GAME = "axelrod_ipd"
TARGET = "opponent"


class IPDActionPolicy:
    def __init__(self, llm: Any):
        self.llm = llm

    def choose(self, view: VisibleView, opponent_models: Mapping[str, str],
               enabled_memories: Mapping[str, Mapping[str, str]]) -> ActionChoice:
        payload = {
            "VISIBLE_OBSERVATION": view.observation,
            "COMPLETED_PUBLIC_HISTORY": list(view.public_history),
            "OPPONENT_MODELING": opponent_models.get(TARGET),
            "RETRIEVED_MEMORY": (dict(enabled_memories[TARGET])
                                 if TARGET in enabled_memories else None),
            "LEGAL_ACTIONS": list(view.legal_actions),
        }
        result = self.llm.complete_json(
            stage="ipd_action",
            system=("You are SELF in iterated Prisoner's Dilemma. Choose using only "
                    "the completed player-visible history. Opponent modeling and "
                    "retrieved advice are fallible. Return JSON with exactly action "
                    "(Cooperate or Defect) and response_intent (at most 20 words)."),
            prompt=INFERENCE_CONTEXT + "\nDATA:\n" + json.dumps(payload, ensure_ascii=False),
        )
        action, intent = result.get("action"), result.get("response_intent", "")
        if action not in ACTIONS or not isinstance(intent, str):
            raise ValueError("the action model must choose Cooperate or Defect")
        return ActionChoice(action, intent)


@dataclass(frozen=True)
class PlayedEpisode:
    evidence: EpisodeEvidence
    training: TrainingEpisode | None


def play_episode(*, episode_id: str, split: str, opponent_id: str,
                 seed: int, horizon: int, bank: MemoryBank,
                 retriever: E5MedoidNLIRetriever, llm: Any,
                 axelrod_source: str | Path | None = None) -> PlayedEpisode:
    """Run actual Axelrod turns; keep evaluator identity out of LLM prompts."""
    if horizon < 2:
        raise ValueError("horizon must be at least two rounds")
    adapter = AxelrodAdapter(horizon=horizon, source=axelrod_source)
    agent = OppoMemAgent(
        game=GAME, actor=IPDActionPolicy(llm),
        modeler=PromptOpponentModeler(llm, game_context=INFERENCE_CONTEXT),
        bank=bank, retriever=retriever,
    )
    agent.begin_episode(episode_id=episode_id, configuration_id=opponent_id,
                        split=split, rng=random.Random(seed ^ 0xB0A7))
    events: list[VisibleEvent] = []
    descriptions: list[OnlineDescription] = []
    try:
        frame = adapter.reset(opponent_id=opponent_id, seed=seed,
                              seat=0, training=(split == "train"))
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
            transition = adapter.step(decision.action)
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
        trajectory = tuple(dict(event.public_event) for event in events)
        evidence = agent.finish_episode(terminal_return=frame.cumulative_return,
                                        public_trajectory=trajectory)
        training = (TrainingEpisode(
            episode_id=episode_id, game=GAME,
            terminal_return=frame.cumulative_return,
            visible_events=tuple(events), online_descriptions=tuple(descriptions),
        ) if split == "train" else None)
        return PlayedEpisode(evidence, training)
    finally:
        adapter.close()
