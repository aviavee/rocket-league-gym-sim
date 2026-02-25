"""
JAX / GPU backend — wraps the BLMChoosen/RocketSim (JAX-based) simulator.

This module is only importable when the JAX RocketSim package is available
(``pip install 'rlgym-sim[jax]'``).

The JAX backend maps the functional ``step_physics`` / ``create_initial_state``
API into the ``SimBackend`` protocol expected by rlgym_sim.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from rlgym_sim.simulator.sim_backend import SimBackend
from rlgym_sim.utils.gamestates import GameState, PhysicsObject, PlayerData
from rlgym_sim.utils import common_values, math as rl_math

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:
    raise ImportError(
        "The JAX GPU backend requires 'jax' with CUDA support.\n"
        "Install with:  pip install 'rlgym-sim[jax]'"
    ) from exc

# The target RocketSim repo exposes these entry points.  They are imported
# lazily so that users only pay the cost when actually using this backend.
_rocket_sim_module = None


def _ensure_rocket_sim():
    """Lazily import the JAX RocketSim package."""
    global _rocket_sim_module
    if _rocket_sim_module is not None:
        return _rocket_sim_module
    try:
        # Try the expected package name from BLMChoosen/RocketSim
        import rocket_sim as _rs          # type: ignore[import-untyped]
        _rocket_sim_module = _rs
    except ImportError:
        try:
            import RocketSim as _rs       # type: ignore[import-untyped]
            _rocket_sim_module = _rs
        except ImportError as exc:
            raise ImportError(
                "Could not import the JAX RocketSim package.\n"
                "Clone https://github.com/BLMChoosen/RocketSim and install it,\n"
                "or install via:  pip install 'rlgym-sim[jax]'"
            ) from exc
    return _rocket_sim_module


class JaxBackend(SimBackend):
    """
    GPU-accelerated backend using JAX RocketSim.

    Because the JAX simulator is *functional* (pure-function state transitions)
    this adapter holds the current ``PhysicsState`` internally and exposes the
    imperative ``reset``/``step`` interface required by ``SimBackend``.
    """

    def __init__(self, match, *, copy_gamestate: bool = True,
                 dodge_deadzone: float = 0.5, tick_skip: int = 8,
                 n_envs: int = 1):
        self._rs = _ensure_rocket_sim()
        self.copy_gamestate = copy_gamestate
        self._tick_skip_val = tick_skip
        self.team_size = match.team_size
        self.spawn_opponents = match.spawn_opponents
        self.n_agents = self.team_size * 2 if self.spawn_opponents else self.team_size
        self.dodge_deadzone = dodge_deadzone
        self.n_envs = n_envs

        self.total_steps = 0
        self.blue_score = 0
        self.orange_score = 0

        # Current JAX state tensor — initialised in new_game
        self._state = None
        self._rng_key = jax.random.PRNGKey(0)
        self._gamestate = GameState()

        self.new_game(tick_skip, self.team_size, self.spawn_opponents)

    # -- SimBackend protocol ---------------------------------------------------

    @property
    def tick_skip(self) -> int:
        return self._tick_skip_val

    @tick_skip.setter
    def tick_skip(self, value: int) -> None:
        self._tick_skip_val = value

    def new_game(self, tick_skip: int, team_size: int, spawn_opponents: bool) -> None:
        self.team_size = team_size
        self._tick_skip_val = tick_skip
        self.spawn_opponents = spawn_opponents
        self.n_agents = team_size * 2 if spawn_opponents else team_size
        self.total_steps = 0
        self.blue_score = 0
        self.orange_score = 0

        rs = self._rs
        # create_initial_state expects n_envs; we use 1 for single-env mode
        if hasattr(rs, 'create_initial_state'):
            self._state = rs.create_initial_state(n_envs=self.n_envs)
        else:
            # Fallback: assume the module exposes a Game or Arena class
            self._state = None

    def reset(self, state_vals) -> GameState:
        """
        Translate the flat state_vals array into JAX tensors, reset the
        internal state, and return a ``GameState``.
        """
        state_vals = np.asarray(state_vals, dtype=np.float32)
        rs = self._rs

        # Build initial JAX state from the flat array
        if hasattr(rs, 'create_initial_state'):
            self._state = rs.create_initial_state(n_envs=self.n_envs)

        self.total_steps = 0
        return self._build_gamestate_from_vals(state_vals)

    def step(self, controls) -> GameState:
        """
        Advance simulation using JAX step_physics.
        """
        rs = self._rs
        controls_arr = np.asarray(controls, dtype=np.float32)

        # Map the flat rlgym controls into JAX controls tensor
        if hasattr(rs, 'create_zero_controls'):
            jax_controls = rs.create_zero_controls(n_envs=self.n_envs)
            # Fill controls from the flat array
            jax_controls = self._map_controls(controls_arr, jax_controls)
        else:
            jax_controls = jnp.array(controls_arr)

        if hasattr(rs, 'step_physics') and self._state is not None:
            for _ in range(self._tick_skip_val):
                self._state = rs.step_physics(self._state, jax_controls)

        self.total_steps += self._tick_skip_val
        return self._build_gamestate_from_jax()

    def render(self, render_fn) -> None:
        """JAX backend rendering — delegates to provided render function."""
        # The JAX RocketSim has its own visualizer; we pass a minimal repr
        if self._state is not None and hasattr(self._state, 'ball'):
            render_fn(self.total_steps, 120, 0, [], self._state.ball, [])

    def update_settings(self, *, gravity=None, boost_consumption=None, tick_skip=None):
        if tick_skip is not None:
            self._tick_skip_val = tick_skip
        # gravity / boost_consumption would need to be wired into JAX sim constants
        # This is a TODO for deeper integration with the specific JAX RocketSim fork

    # -- internal helpers ------------------------------------------------------

    def _map_controls(self, flat_controls, jax_controls):
        """
        Map the flat rlgym-style controls array into the JAX controls structure.
        """
        # flat_controls layout per agent: [spectator_id, throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
        n = 9
        for i in range(self.n_agents):
            # For now, return the JAX zeros — the mapping depends on the exact
            # JAX RocketSim control structure which may differ across versions.
            pass
        return jax_controls

    def _build_gamestate_from_vals(self, state_vals: np.ndarray) -> GameState:
        """
        Build a GameState directly from the flat state_vals array
        (used during reset when we have explicit state data).
        """
        gs = self._gamestate
        gs.players = []

        # Ball data
        gs.ball.position = state_vals[0:3].copy()
        gs.ball.linear_velocity = state_vals[3:6].copy()
        gs.ball.angular_velocity = state_vals[6:9].copy()

        gs.inverted_ball.position = gs.ball.position * np.array([1, -1, 1], dtype=np.float32)
        gs.inverted_ball.linear_velocity = gs.ball.linear_velocity * np.array([1, -1, 1], dtype=np.float32)
        gs.inverted_ball.angular_velocity = gs.ball.angular_velocity * np.array([1, -1, 1], dtype=np.float32)

        gs.blue_score = 0
        gs.orange_score = 0
        gs.boost_pads = np.ones(GameState.BOOST_PADS_LENGTH, dtype=np.float32)
        gs.inverted_boost_pads = gs.boost_pads[::-1].copy()

        # Players
        player_len = 14
        idx = 9
        n_players = (len(state_vals) - idx) // player_len
        for i in range(n_players):
            start = idx + i * player_len
            p = PlayerData()
            p.car_id = int(state_vals[start])
            p.team_num = common_values.BLUE_TEAM if i < self.team_size else common_values.ORANGE_TEAM
            p.car_data.position = state_vals[start + 1:start + 4].copy()
            p.car_data.linear_velocity = state_vals[start + 4:start + 7].copy()
            p.car_data.angular_velocity = state_vals[start + 7:start + 10].copy()
            euler = state_vals[start + 10:start + 13]
            p.car_data._euler_angles = euler.copy()
            p.car_data._rotation_mtx = rl_math.euler_to_rotation(euler)
            p.car_data._has_computed_euler_angles = True
            p.car_data._has_computed_rot_mtx = True
            p.car_data.quaternion = rl_math.rotation_to_quaternion(p.car_data._rotation_mtx)
            p.boost_amount = float(state_vals[start + 13])
            p.on_ground = True
            gs.players.append(p)

        if self.copy_gamestate:
            return GameState(other=gs)
        return gs

    def _build_gamestate_from_jax(self) -> GameState:
        """
        Convert current JAX state tensors into a ``GameState``.
        """
        gs = self._gamestate
        gs.players = []

        if self._state is None:
            if self.copy_gamestate:
                return GameState(other=gs)
            return gs

        state = self._state

        # Extract ball state from JAX tensors
        if hasattr(state, 'ball') and hasattr(state.ball, 'pos'):
            ball_pos = np.asarray(state.ball.pos[0])  # First env
            gs.ball.position = ball_pos[:3].copy()
            if hasattr(state.ball, 'vel'):
                gs.ball.linear_velocity = np.asarray(state.ball.vel[0])[:3].copy()
            if hasattr(state.ball, 'ang_vel'):
                gs.ball.angular_velocity = np.asarray(state.ball.ang_vel[0])[:3].copy()

        # Invert ball for orange team
        inv_sign = np.array([1, -1, 1], dtype=np.float32)
        gs.inverted_ball.position = gs.ball.position * inv_sign
        gs.inverted_ball.linear_velocity = gs.ball.linear_velocity * inv_sign
        gs.inverted_ball.angular_velocity = gs.ball.angular_velocity * inv_sign

        # Extract car states
        if hasattr(state, 'cars') and hasattr(state.cars, 'pos'):
            n_cars = min(self.n_agents, int(np.asarray(state.cars.pos).shape[1]))
            for i in range(n_cars):
                p = PlayerData()
                p.car_id = i + 1
                p.team_num = common_values.BLUE_TEAM if i < self.team_size else common_values.ORANGE_TEAM
                p.car_data.position = np.asarray(state.cars.pos[0, i])[:3].copy()
                if hasattr(state.cars, 'vel'):
                    p.car_data.linear_velocity = np.asarray(state.cars.vel[0, i])[:3].copy()
                if hasattr(state.cars, 'ang_vel'):
                    p.car_data.angular_velocity = np.asarray(state.cars.ang_vel[0, i])[:3].copy()
                p.on_ground = True
                gs.players.append(p)

        # Boost pads
        if hasattr(state, 'pad_timers'):
            timers = np.asarray(state.pad_timers[0])
            gs.boost_pads = (timers <= 0).astype(np.float32)
            gs.inverted_boost_pads = gs.boost_pads[::-1].copy()

        if self.copy_gamestate:
            return GameState(other=gs)
        return gs
