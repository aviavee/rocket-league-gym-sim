"""
    The Rocket League gym environment.

    Supports both legacy ``gym`` and modern ``gymnasium`` packages via the
    compat shim.  The step signature adapts automatically:
      - gymnasium → ``(obs, reward, terminated, truncated, info)``
      - gym       → ``(obs, reward, done, info)``
"""
from __future__ import annotations

from typing import List, Union, Tuple, Dict, Any, Optional

from rlgym_sim.utils.compat import Env, GYMNASIUM
from rlgym_sim.simulator.sim_backend import SimBackend
from rlgym_sim.utils import common_values

try:
    import rlviser_py as rlviser
    rlviser.set_boost_pad_locations(common_values.BOOST_LOCATIONS)
except ImportError:
    rlviser = None


def _create_backend(backend: str, match, *, copy_gamestate, dodge_deadzone,
                    tick_skip, **extra_kw) -> SimBackend:
    """Instantiate the requested simulator backend."""
    if backend == "legacy":
        from rlgym_sim.simulator.legacy_backend import LegacyBackend
        return LegacyBackend(match, copy_gamestate=copy_gamestate,
                             dodge_deadzone=dodge_deadzone, tick_skip=tick_skip)
    elif backend == "jax":
        from rlgym_sim.simulator.jax_backend import JaxBackend
        n_envs = extra_kw.get("n_envs", 1)
        return JaxBackend(match, copy_gamestate=copy_gamestate,
                          dodge_deadzone=dodge_deadzone, tick_skip=tick_skip,
                          n_envs=n_envs)
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Choose 'legacy' or 'jax'."
        )


class Gym(Env):
    def __init__(self, match, copy_gamestate_every_step, dodge_deadzone,
                 tick_skip, gravity, boost_consumption,
                 backend: str = "legacy", **backend_kw):
        super().__init__()

        self._match = match
        self.observation_space = match.observation_space
        self.action_space = match.action_space
        self._prev_state = None
        self.rendered = False
        self._backend_name = backend

        self._game: SimBackend = _create_backend(
            backend, match,
            copy_gamestate=copy_gamestate_every_step,
            dodge_deadzone=dodge_deadzone,
            tick_skip=tick_skip,
            **backend_kw,
        )

        self._game.update_settings(gravity=gravity,
                                   boost_consumption=boost_consumption,
                                   tick_skip=tick_skip)

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None,
              return_info: bool = False) -> Any:
        """
        Reset the environment.

        * **gymnasium mode** → always returns ``(obs, info)``
        * **gym mode** → returns ``obs`` unless *return_info* is True
        """
        state_str = self._match.get_reset_state()
        state = self._game.reset(state_str)

        self._match.episode_reset(state)
        self._prev_state = state

        obs = self._match.build_observations(state)
        info = {
            'state': state,
            'result': self._match.get_result(state),
        }

        if GYMNASIUM:
            return obs, info

        # Legacy gym path
        if return_info:
            return obs, info
        return obs

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, actions: Any):
        """
        Advance the environment by one logical step.

        * **gymnasium** → ``(obs, reward, terminated, truncated, info)``
        * **gym**       → ``(obs, reward, done, info)``
        """
        actions = self._match.format_actions(
            self._match.parse_actions(actions, self._prev_state)
        )

        state = self._game.step(actions)

        obs = self._match.build_observations(state)
        done = self._match.is_done(state)
        reward = self._match.get_rewards(state, done)
        self._prev_state = state

        info: Dict[str, Any] = {
            'state': state,
            'result': self._match.get_result(state),
        }

        if GYMNASIUM:
            # terminated = episode ended naturally; truncated = time limit etc.
            return obs, reward, done, False, info

        return obs, reward, done, info

    # ------------------------------------------------------------------
    # render / close / settings
    # ------------------------------------------------------------------

    def render(self):
        if rlviser is None:
            raise ImportError(
                "rlviser_py not installed. Install it to use render()."
            )
        if self._prev_state is None:
            return
        self.rendered = True
        self._game.render(rlviser.render)

    def close(self):
        if self.rendered:
            rlviser.quit()

    def update_settings(self, gravity=None, boost_consumption=None,
                        tick_skip=None):
        """
        Update RocketSim mutator settings on the active backend.
        """
        self._game.update_settings(gravity=gravity,
                                   boost_consumption=boost_consumption,
                                   tick_skip=tick_skip)
        if tick_skip is not None:
            self._match.tick_skip = tick_skip
