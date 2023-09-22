"""Interface for adders which transmit data to a replay buffer.

TODO(maxsmith): Promote this to be used without reverb.
"""

import abc

from marl import types, worlds


class BaseAdder(abc.ABC):
  """The Adder interface.

  An adder packs together data to send to the replay buffer, and potentially
  performs some reduction/transformation to this data in the process.
  All adders will use this API. Below is an illustrative example of how they
  are intended to be used in a typical RL run-loop. We assume that the
  environment conforms to the dm_env environment API.

  ```python
  # Reset the environment and add the first observation.
  timestep = env.reset()
  adder.add_first(timestep.observation)
  while not timestep.last():
      # Generate an action from the policy and step the environment.
      action = my_policy(timestep)
      timestep = env.step(action)
      # Add the action and the resulting timestep.
      adder.add(action, next_timestep=timestep)
  ```

  Note that for all adders, the `add()` method expects an action taken and the
  *resulting* timestep observed after taking this action. Note that this
  timestep is named `next_timestep` precisely to emphasize this point.
  """

  @abc.abstractmethod
  def add(
      self,
      timestep: worlds.TimeStep,
      action: types.Tree = None,
      extras: types.Tree = (),
  ):
    """Defines the adder `add` interface.

    Args:
        timestep: A dm_env Timestep object corresponding to the resulting
            data obtained by taking the given action, or the first timestep
            if no action is provided.
        action: A possibly nested structure corresponding to a_t.
        extras: A possibly nested structure of extra data to add to replay.
    """

  @abc.abstractmethod
  def reset(self):
    """Resets the adder's buffer."""
