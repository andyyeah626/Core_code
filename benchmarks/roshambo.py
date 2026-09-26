from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any

from .protocol import DecisionFrame, Transition


ACTIONS = ("Rock", "Paper", "Scissors")
ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}

# Lanctot et al. 2023 (arXiv:2303.03196v2) section 2.2: "The resulting population
# consists of 43 bots: 25 entrant bots and 18 seed bots from the first competition."
# These are the 18 seed bots, i.e. the intentionally simple reference opponents.
# All 18 names are verified members of pyspiel.roshambo_bot_names().
SEED_BOTS = (
    # Simplest: do not consult the observation at all.
    "randbot", "rockbot", "r226bot", "rotatebot",
    "pibot", "debruijn81", "textbot",
    # Recall R = 1: only the most recent joint action.
    "switchbot", "switchalot", "copybot",
    "driftbot", "adddriftbot2", "foxtrotbot",
    # Historical observations or statistical summaries.
    "flatbot3", "addshiftbot3", "antiflatbot",
    "antirotnbot", "freqbot2",
)

INFERENCE_CONTEXT = (
    "Repeated simultaneous Rock-Paper-Scissors: both players choose without seeing "
    "the current-round action. Completed joint actions are public. Rock beats "
    "Scissors, Scissors beats Paper, Paper beats Rock; a win pays +1, a draw 0, "
    "and a loss -1."
)


class RoshamboAdapter:
    """OpenSpiel's official 43-member Roshambo population.

    A runner should create at most one instance per process. Several upstream C
    bots retain global state and are unsafe to repeatedly construct in-process.
    """

    game_id = "open_spiel_roshambo"

    def __init__(self, *, horizon: int, recall: int = 1) -> None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.horizon = int(horizon)
        self.recall = int(recall)
        self._state = None
        self._bot = None
        self._history: list[dict[str, Any]] = []
        self._total = 0.0
        self._opponent_id = ""
        self._seat = 0

    @staticmethod
    def available_opponents() -> tuple[str, ...]:
        import pyspiel

        return tuple(sorted(pyspiel.roshambo_bot_names()))

    def reset(self, *, opponent_id: str, seed: int, seat: int = 0,
              training: bool = False) -> DecisionFrame:
        import numpy as np
        import pyspiel

        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        if opponent_id not in self.available_opponents():
            raise ValueError(f"Unknown Roshambo bot {opponent_id!r}")
        np.random.seed(int(seed) & 0xFFFFFFFF)
        reuse_bot = (self._bot is not None and self._opponent_id == opponent_id
                     and self._seat == seat)
        self._seat = seat
        self._opponent_id = opponent_id
        self._history = []
        self._total = 0.0
        game = pyspiel.load_game(
            "repeated_game(stage_game=matrix_rps(),"
            f"num_repetitions={self.horizon},recall={self.recall})"
        )
        self._state = game.new_initial_state()
        if not reuse_bot:
            self._bot = pyspiel.make_roshambo_bot(1 - seat, opponent_id, self.horizon)
        self._bot.restart()
        return self._frame()

    def _frame(self) -> DecisionFrame:
        payload: dict[str, Any] = {
            "game": "Repeated simultaneous Rock Paper Scissors",
            "round": len(self._history),
            "remaining_rounds": self.horizon - len(self._history),
            "cumulative_return": self._total,
            "completed_joint_actions": self._history,
            "timing": (
                "Simultaneous: neither player can observe the other's action for the "
                "round being chosen. An action can depend only on rounds completed "
                "before it."
            ),
            "payoff": "win=+1, draw=0, loss=-1",
        }
        terminal = len(self._history) >= self.horizon
        return DecisionFrame(
            observation=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            legal_actions=() if terminal else ACTIONS,
            cumulative_return=self._total,
            terminal=terminal,
            inference_context=INFERENCE_CONTEXT,
        )

    def step(self, action: str) -> Transition:
        if self._state is None or self._bot is None:
            raise RuntimeError("Call reset first")
        if action not in ACTIONS or self._state.is_terminal():
            raise ValueError(f"Illegal Roshambo action {action!r}")
        opponent_action_id = int(self._bot.step(self._state))
        own_action_id = ACTION_TO_ID[action]
        joint = ([own_action_id, opponent_action_id] if self._seat == 0
                 else [opponent_action_id, own_action_id])
        self._state.apply_actions(joint)
        reward = float(self._state.rewards()[self._seat])
        opponent_action = ACTIONS[opponent_action_id]
        self._total += reward
        self._history.append({
            "round": len(self._history),
            "self": action,
            "opponent": opponent_action,
            "reward": reward,
        })
        frame = self._frame()
        if frame.terminal:
            terminal_return = float(self._state.returns()[self._seat])
            if abs(terminal_return - self._total) > 1e-9:
                raise RuntimeError("Roshambo incremental rewards disagree with terminal return")
        return Transition(
            frame=frame,
            opponent_event=opponent_action,
            reward_delta=reward,
            terminal=frame.terminal,
            evaluator_metrics={"round": len(self._history)},
        )

    def close(self) -> None:
        self._state = None
        self._bot = None

    def fingerprint(self) -> dict[str, Any]:
        import pyspiel

        return {
            "adapter": type(self).__name__,
            "protocol_id": "open-spiel-roshambo-policy-induction-v3",
            "pyspiel_version": getattr(pyspiel, "__version__", "unknown"),
            "population_size": len(self.available_opponents()),
            "horizon": self.horizon,
            "recall": self.recall,
        }


class RPSCallbackAdapter:
    """Repeated RPS against an externally supplied LLM opponent callback.

    The callback receives only completed rounds, never the focal player's
    current simultaneous move or any evaluator-only opponent label.
    """

    game_id = "open_spiel_roshambo"

    def __init__(self, opponent: Callable[[tuple[Mapping[str, Any], ...], int], str],
                 *, horizon: int):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        self.opponent = opponent
        self.horizon = horizon
        self._history: list[dict[str, Any]] = []
        self._total = 0.0

    def reset(self, *, opponent_id: str, seed: int, seat: int = 0,
              training: bool = False) -> DecisionFrame:
        if seat not in (0, 1):
            raise ValueError("seat must be 0 or 1")
        self._history = []
        self._total = 0.0
        return self._frame()

    def _frame(self) -> DecisionFrame:
        terminal = len(self._history) >= self.horizon
        return DecisionFrame(
            observation=json.dumps({
                "game": "Repeated simultaneous Rock Paper Scissors",
                "round": len(self._history),
                "remaining_rounds": self.horizon - len(self._history),
                "completed_joint_actions": self._history,
                "cumulative_return": self._total,
            }, ensure_ascii=False),
            legal_actions=() if terminal else ACTIONS,
            cumulative_return=self._total, terminal=terminal,
            inference_context=INFERENCE_CONTEXT,
        )

    def step(self, action: str) -> Transition:
        if action not in ACTIONS or len(self._history) >= self.horizon:
            raise ValueError("Illegal RPS action")
        opponent_action = self.opponent(
            tuple(dict(row) for row in self._history), len(self._history))
        if opponent_action not in ACTIONS:
            raise ValueError("Opponent callback returned an illegal RPS action")
        reward = float((ACTION_TO_ID[action] - ACTION_TO_ID[opponent_action]) % 3)
        reward = 0.0 if reward == 0 else (1.0 if reward == 1 else -1.0)
        self._total += reward
        self._history.append({"round": len(self._history), "self": action,
                              "opponent": opponent_action, "reward": reward})
        frame = self._frame()
        return Transition(frame, opponent_action, reward, frame.terminal,
                          evaluator_metrics={"round": len(self._history)})

    def close(self) -> None:
        pass

