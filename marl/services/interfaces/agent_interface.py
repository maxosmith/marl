import abc

import dm_env

from marl import _types, worlds


class AgentInterface(abc.ABC):
    """Interface for an agent that can act.

    This interface defines an API for an Actor to interact with an EnvironmentLoop
    (see acme.environment_loop), e.g. a simple RL loop where each step is of the
    form:

        # Make the first observation.
        timestep = env.reset()
        actor.observe_first(timestep)

        # Take a step and observe.
        action = actor.select_action(timestep.observation)
        next_timestep = env.step(action)
        actor.observe(action, next_timestep)

        # Update the actor policy/parameters.
        actor.update()
    """

    @abc.abstractmethod
    def step(self, timestep: worlds.TimeStep) -> _types.Tree:
        """Samples from the policy and returns an action."""

    @abc.abstractmethod
    def update(self, wait: bool = False):
        """Perform an update of the actor parameters from past observations.

        Args:
            wait: if True, the update will be blocking.
        """
