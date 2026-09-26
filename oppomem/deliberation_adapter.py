"""Connect the existing LLM-Deliberation protocol to the OppoMem method core.

The game description and background-party policy are supplied by the caller.
Only public utterances reach opponent modeling; private plans and utility tables
remain with their respective acting party and the evaluator.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping

from benchmarks.deliberation_protocol import Game, parse_response

from .agent import ActionChoice, OppoMemAgent
from .memory import E5MedoidNLIRetriever, MemoryBank
from .offline import PromptOpponentModeler
from .segmentation import OnlineDescription, TrainingEpisode, VisibleEvent
from .types import EpisodeEvidence, VisibleView


GAME = "llm_deliberation"
BackgroundPolicy = Callable[[str, str, str, str], str]


class DeliberationActionPolicy:
    def __init__(self, game: Game, llm: Any):
        self.game, self.llm = game, llm

    def choose(self, view: VisibleView, opponent_models: Mapping[str, str],
               enabled_memories: Mapping[str, Mapping[str, str]]) -> ActionChoice:
        payload = {
            "OWN_ROUND_PROMPT": view.own_private["round_prompt"],
            "PUBLIC_HISTORY": list(view.public_history),
            "OPPONENT_MODELING": dict(opponent_models),
            "RETRIEVED_MEMORIES": {name: dict(row) for name, row in enabled_memories.items()},
        }
        answer = self.llm.complete_json(
            stage="deliberation_action",
            system=(self.game.initial_prompt(self.game.p1) + "\nReturn JSON with "
                    "raw_response containing the required public answer tags and "
                    "response_intent of at most 20 words. Keep private scratchpad "
                    "and plan text inside their own tags."),
            prompt="DATA:\n" + json.dumps(payload, ensure_ascii=False),
        )
        raw, intent = answer.get("raw_response"), answer.get("response_intent", "")
        if not isinstance(raw, str) or not isinstance(intent, str):
            raise ValueError("deliberation policy must return response and text intent")
        return ActionChoice(raw, intent)


@dataclass(frozen=True)
class PlayedDeliberationEpisode:
    evidence: EpisodeEvidence
    training: TrainingEpisode | None
    metrics: Mapping[str, Any]


def _public_turn(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in ("index", "agent", "phase", "public_answer")}


def play_episode(*, game: Game, episode_id: str, configuration_id: str,
                 split: str, seed: int, bank: MemoryBank,
                 retriever: E5MedoidNLIRetriever, llm: Any,
                 background_policy: BackgroundPolicy,
                 discussion_turns: int = 48,
                 window_size: int = 6) -> PlayedDeliberationEpisode:
    """Play the existing speaker schedule with focal-only memory injection.

    background_policy(name, phase, own_initial_prompt, own_round_prompt)
    returns a raw tagged response for that background party.
    """
    if discussion_turns < 1 or window_size < 1:
        raise ValueError("discussion_turns and window_size must be positive")
    order = game.speaker_order(seed, rounds_num=discussion_turns)
    if not order or order[0] != game.p1 or order[-1] != game.p1:
        raise ValueError("the speaker schedule must start and end with the focal party")
    agent = OppoMemAgent(
        game=GAME, actor=DeliberationActionPolicy(game, llm),
        modeler=PromptOpponentModeler(llm, game_context=game.public_rules),
        bank=bank, retriever=retriever,
        validate_action=lambda view, raw: parse_response(
            raw, view.phase or "", game.option_counts)["valid_response"],
    )
    agent.begin_episode(episode_id=episode_id, configuration_id=configuration_id,
                        split=split, rng=random.Random(seed ^ 0xDE1B))
    turns: list[dict[str, Any]] = []
    events: list[VisibleEvent] = []
    descriptions: list[OnlineDescription] = []
    last_focal_turn = -1
    for index, name in enumerate(order):
        phase = "opening" if index == 0 else "final" if index == len(order) - 1 else "discussion"
        round_prompt = game.round_prompt(
            name, turns, phase, index,
            rounds_num=discussion_turns, window_size=window_size,
        )
        if name == game.p1:
            history = tuple(_public_turn(row) for row in turns[-window_size:])
            seen = {row["agent"] for row in history}
            targets = tuple(row["name"] for row in game.agents
                            if row["name"] != name and row["name"] in seen)
            view = VisibleView(
                game=GAME, episode_id=episode_id, decision_index=index,
                self_id=name, target_ids=targets,
                observation={"public_rules": game.public_rules, "phase": phase},
                public_history=history, legal_actions=(),
                own_private={"round_prompt": round_prompt}, phase=phase,
            )
            decision = agent.step(view)
            raw = decision.action
            for target, description in decision.opponent_models.items():
                descriptions.append(OnlineDescription(index, target, description))
        else:
            raw = background_policy(
                name, phase, game.initial_prompt(name), round_prompt)
            if not isinstance(raw, str):
                raise ValueError("background policy must return a raw text response")
        parsed = parse_response(raw, phase, game.option_counts)
        if not parsed["valid_response"]:
            raise ValueError(f"invalid response envelope for party {name}")
        turns.append({
            "index": index, "agent": name, "phase": phase,
            "public_answer": parsed["public_answer"], "plan": parsed["plan"],
        })
        if name == game.p1:
            events.append(VisibleEvent(
                index=index, actor=name,
                visible_context={"public_history": history},
                public_event={"completed_turns": [
                    _public_turn(row) for row in turns[last_focal_turn + 1:]]},
                focal_action=parsed["public_answer"],
                observed_reward=0.0,
                response_intent=decision.response_intent,
            ))
            last_focal_turn = index
    metrics = game.score_episode(turns)
    terminal_return = float(metrics["focal_realized_utility"])
    if events:
        events[-1] = replace(events[-1], observed_reward=terminal_return)
    trajectory = tuple(_public_turn(row) for row in turns)
    evidence = agent.finish_episode(terminal_return=terminal_return,
                                    public_trajectory=trajectory)
    training = (TrainingEpisode(
        episode_id=episode_id, game=GAME, terminal_return=terminal_return,
        visible_events=tuple(events), online_descriptions=tuple(descriptions),
    ) if split == "train" else None)
    return PlayedDeliberationEpisode(evidence, training, metrics)
