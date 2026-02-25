from typing import List

from rlgym_sim.gym import Gym
from rlgym_sim.envs import Match
from rlgym_sim.utils.terminal_conditions import common_conditions
from rlgym_sim.utils.reward_functions import DefaultReward
from rlgym_sim.utils.obs_builders import DefaultObs
from rlgym_sim.utils.action_parsers import DefaultAction
from rlgym_sim.utils.state_setters import DefaultState
from rlgym_sim.utils import common_values


def make(tick_skip: int = 8,
         spawn_opponents: bool = False,
         team_size: int = 1,
         gravity: float = 1,
         boost_consumption: float = 1,
         copy_gamestate_every_step=True,
         dodge_deadzone=0.8,
         terminal_conditions: List[object] = (common_conditions.TimeoutCondition(225), common_conditions.GoalScoredCondition()),
         reward_fn: object = DefaultReward(),
         obs_builder: object = DefaultObs(),
         action_parser: object = DefaultAction(),
         state_setter: object = DefaultState(),
         backend: str = "legacy",
         **backend_kw):
    """
    :param tick_skip: The amount of physics ticks your action will be repeated for
    :param spawn_opponents: Whether you want opponents or not
    :param team_size: Cars per team
    :param gravity: Game gravity, 1 is normal gravity
    :param boost_consumption: Car boost consumption rate, 1 is normal consumption
    :param dodge_deadzone: Dodge deadzone setting.

    :param copy_gamestate_every_step: Whether a new instance of a GameState object should be returned after every call
    to env.step().

    :param terminal_conditions: List of terminal condition objects
    :param reward_fn: Reward function object
    :param obs_builder: Observation builder object
    :param action_parser: Action parser object
    :param state_setter: State Setter object

    :param backend: Simulator backend to use. ``"legacy"`` for the original C++ RocketSim,
        ``"jax"`` for the GPU-accelerated JAX backend.
    :param backend_kw: Extra keyword arguments forwarded to the backend constructor
        (e.g. ``n_envs=1024`` for the JAX backend).
    :return: Gym object
    """

    match = Match(reward_function=reward_fn,
                  terminal_conditions=terminal_conditions,
                  obs_builder=obs_builder,
                  action_parser=action_parser,
                  state_setter=state_setter,
                  team_size=team_size,
                  spawn_opponents=spawn_opponents)

    return Gym(match, tick_skip=tick_skip, gravity=gravity,
               boost_consumption=boost_consumption,
               copy_gamestate_every_step=copy_gamestate_every_step,
               dodge_deadzone=dodge_deadzone,
               backend=backend, **backend_kw)
