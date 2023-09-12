"""Markov Decisipn Process (MDP)."""
import dataclasses
from typing import Optional, Sequence, Set, Union

import numpy as np

from marl import specs, types, worlds


@dataclasses.dataclass
class Transition:
  """Single transition in a game's dynamics.

  If multiple state/actions are specified the `MDP` is assumed multiagent.
  With the position of the state/action corresponding to the `PlayerID`.
  """

  state: Union[int, Sequence[int]]
  action: Union[int, Sequence[int]]
  next_state: Union[int, Sequence[int]]
  reward: Union[float, Sequence[float]]
  probability: float = 1.0


@dataclasses.dataclass
class MDP(worlds.Game):
  """Markov Decision Process.

  Args:
    dynamics: Sequence of possible transitions. The MDP assumes discrete
      state and actions, and always starts in state 0.
  """

  dynamics: Sequence[Transition]
  terminal_states: Set[int] = dataclasses.field(default_factory=set)

  def __post_init__(self):
    """Post initializer."""
    example_state = self.dynamics[0].state
    self.num_players = 1 if isinstance(example_state, int) else len(example_state)

    for t in self.dynamics:
      to_check = [t.state, t.action, t.next_state, t.reward]

      if (self.num_players == 1) and not all(isinstance(x, int) for x in to_check):
        raise ValueError("Dynamics do not all have same number of players.")

      elif (self.num_players > 1) and not all(
          len(x) == self.num_players for x in to_check
      ):
        raise ValueError("Dynamics do not all have same number of players.")

    # TODO: Validate that the dynamics have valid probability distributions.

    # TODO: Preprocess transitions to make dynamics more efficient.

    self.state = None

  def reset(self) -> Union[worlds.TimeStep, worlds.PlayerIDToTimestep]:
    """Starts a new sequence and returns the first `TimeStep` of this sequence."""
    timestep = worlds.TimeStep(step_type=worlds.StepType.FIRST, reward=0, observation=0)
    if self.num_players == 1:
      self.state = 0
      return timestep
    else:
      self.state = np.zeros(self.num_players)
      return {i: timestep for i in range(self.num_players)}

  def step(
      self, actions: types.PlayerIDToAction
  ) -> Union[worlds.TimeStep, worlds.PlayerIDToTimestep]:
    """Updates the environment according to the action and returns a `TimeStep`."""
    if self.state is None:
      raise RuntimeError("`reset` must be called before `step`.")

    if self.num_players > 1:
      actions = [actions[i] for i in range(self.num_players)]

    # Find valid transitions.
    candidates = []
    for trans in self.dynamics:
      if not np.all(self.state == trans.state):
        continue
      if not np.all(actions == trans.action):
        continue
      candidates.append(trans)

    # TODO: Allow probabilistic transitions.
    if (len(candidates) > 1) or (candidates[0].probability != 1.0):
      raise NotImplementedError(
          "MDP currently only supports deterministic transitions."
      )

    self.state = candidates[0].next_state
    reward = candidates[0].reward

    if self.num_players == 1:
      step_type = (
          worlds.StepType.LAST
          if self.state in self.terminal_states
          else worlds.StepType.MID
      )
      return worlds.TimeStep(step_type=step_type, reward=reward, observation=self.state)
    else:
      return {
          i: worlds.TimeStep(
              step_type=worlds.StepType.LAST
              if self.state[i] in self.terminal_states
              else worlds.StepType.MID,
              reward=reward[i],
              observation=self.state[i],
          )
          for i in range(self.num_players)
      }

  def reward_specs(self) -> specs.PlayerIDToSpec:
    """Describes the reward returned by the game to each player."""
    spec = specs.ArraySpec(shape=(), dtype=float, name="reward")
    if self.num_players == 1:
      return spec
    else:
      return {i: spec for i in range(self.num_players)}

  def observation_specs(self) -> specs.PlayerIDToSpec:
    """Defines the observations provided by the game to each player."""
    spec = specs.ArraySpec(shape=(), dtype=int, name="observation")
    if self.num_players == 1:
      return spec
    else:
      return {i: spec for i in range(self.num_players)}

  def action_specs(self) -> specs.PlayerIDToSpec:
    """Defines the actions that should be provided to `step` by each player."""
    spec = specs.ArraySpec(shape=(), dtype=int, name="action")
    if self.num_players == 1:
      return spec
    else:
      return {i: spec for i in range(self.num_players)}
