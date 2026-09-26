"""Per-game, per-memory randomized activation from Section 4.3."""

from __future__ import annotations

import random


class EpisodeActivation:
    """Sample once on first retrieval and reuse that decision for the full game."""

    def __init__(self, *, training: bool, rng: random.Random | None = None,
                 probability: float = 0.5):
        if probability != 0.5:
            raise ValueError("the paper protocol fixes the training activation probability at 0.5")
        self.training = bool(training)
        self.rng = rng if rng is not None else random.Random()
        self._decisions: dict[str, bool] = {}

    def enabled(self, memory_id: str) -> bool:
        if not memory_id:
            raise ValueError("memory_id is required")
        if memory_id not in self._decisions:
            self._decisions[memory_id] = (
                self.rng.random() < 0.5 if self.training else True
            )
        return self._decisions[memory_id]

    def checkpoint(self) -> tuple[dict[str, bool], object]:
        """Capture decisions and RNG state before an uncommitted decision step."""
        return dict(self._decisions), self.rng.getstate()

    def restore(self, checkpoint: tuple[dict[str, bool], object]) -> None:
        decisions, rng_state = checkpoint
        self._decisions = dict(decisions)
        self.rng.setstate(rng_state)

    @property
    def decisions(self) -> dict[str, bool]:
        return dict(self._decisions)
