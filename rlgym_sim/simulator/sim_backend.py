"""
Abstract backend protocol for simulator engines.

Every backend must implement this interface so that the Gym wrapper and Match
can remain engine-agnostic.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from rlgym_sim.utils.gamestates import GameState


class SimBackend(ABC):
    """Protocol that every simulator backend must satisfy."""

    @abstractmethod
    def new_game(self, tick_skip: int, team_size: int, spawn_opponents: bool) -> None:
        """Create / reset the internal arena and cars."""
        ...

    @abstractmethod
    def reset(self, state_vals) -> GameState:
        """
        Reset the arena to the state described by *state_vals*
        (flat list / array produced by ``StateWrapper.format_state``).

        Returns the resulting ``GameState``.
        """
        ...

    @abstractmethod
    def step(self, controls) -> GameState:
        """Advance the simulation by one logical step and return the new state."""
        ...

    @abstractmethod
    def render(self, render_fn) -> None:
        """Pass current visual state to *render_fn* for display."""
        ...

    # -- optional settings / properties ----------------------------------------

    @property
    @abstractmethod
    def tick_skip(self) -> int:
        ...

    @tick_skip.setter
    @abstractmethod
    def tick_skip(self, value: int) -> None:
        ...

    def update_settings(self, *, gravity: Optional[float] = None,
                        boost_consumption: Optional[float] = None,
                        tick_skip: Optional[int] = None) -> None:
        """Apply mutator-style settings.  Backends may override for richer support."""
        if tick_skip is not None:
            self.tick_skip = tick_skip
