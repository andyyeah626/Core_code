"""Player-visible game adapter interfaces bundled with OppoMem."""

from .protocol import DecisionFrame, GameAdapter, Transition

__all__ = ["DecisionFrame", "GameAdapter", "Transition"]
