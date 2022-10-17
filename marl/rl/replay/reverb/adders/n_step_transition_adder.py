"""Transition adders.

This implements an N-step transition adder which collapses trajectory sequences
into a single transition, simplifying to a simple transition adder when N=1.
"""

import copy
from typing import Optional, Tuple

import numpy as np
import reverb
import tree

from marl import _types, worlds
from marl.rl.replay.reverb.adders import reverb_adder, utils
from marl.utils import tree_utils


class NStepTransitionAdder(reverb_adder.ReverbAdder):
    """An N-step transition adder.

    This will buffer a sequence of N timesteps in order to form a single N-step
    transition which is added to reverb for future retrieval.

    For N=1 the data added to replay will be a standard one-step transition which
    takes the form:

            (s_t, a_t, r_t, d_t, s_{t+1}, e_t)

    where:

        s_t = state observation at time t
        a_t = the action taken from s_t
        r_t = reward ensuing from action a_t
        d_t = environment discount ensuing from action a_t. This discount is
            applied to future rewards after r_t.
        e_t [Optional] = extra data that the agent persists in replay.

    For N greater than 1, transitions are of the form:

            (s_t, a_t, R_{t:t+n}, D_{t:t+n}, s_{t+N}, e_t),

    where:

        s_t = State (observation) at time t.
        a_t = Action taken from state s_t.
        g = the additional discount, used by the agent to discount future returns.
        R_{t:t+n} = N-step discounted return, i.e. accumulated over N rewards:
            R_{t:t+n} := r_t + g * d_t * r_{t+1} + ...
                            + g^{n-1} * d_t * ... * d_{t+n-2} * r_{t+n-1}.
        D_{t:t+n}: N-step product of agent discounts g_i and environment
        "discounts" d_i.
            D_{t:t+n} := g^{n-1} * d_{t} * ... * d_{t+n-1},
        For most environments d_i is 1 for all steps except the last,
        i.e. it is the episode termination signal.
        s_{t+n}: The "arrival" state, i.e. the state at time t+n.
        e_t [Optional]: A nested structure of any 'extras' the user wishes to add.

    Notes:
        - At the beginning and end of episodes, shorter transitions are added.
        That is, at the beginning of the episode, it will add:
                (s_0 -> s_1), (s_0 -> s_2), ..., (s_0 -> s_n), (s_1 -> s_{n+1})

        And at the end of the episode, it will add:
                (s_{T-n+1} -> s_T), (s_{T-n+2} -> s_T), ... (s_{T-1} -> s_T).
        - We add the *first* `extra` of each transition, not the *last*, i.e.
            if extras are provided, we get e_t, not e_{t+n}.
    """

    def __init__(
        self,
        client: reverb.Client,
        n_step: int,
        discount: float,
        *,
        priority_fns: Optional[reverb_adder.PriorityFnMapping] = None,
        max_in_flight_items: int = 5,
    ):
        """Creates an N-step transition adder.

        Args:
          client: A `reverb.Client` to send the data to replay through.
          n_step: The "N" in N-step transition. See the class docstring for the
            precise definition of what an N-step transition is. `n_step` must be at
            least 1, in which case we use the standard one-step transition, i.e.
            (s_t, a_t, r_t, d_t, s_t+1, e_t).
          discount: Discount factor to apply. This corresponds to the agent's
            discount in the class docstring.
          priority_fns: See docstring for BaseAdder.
          max_in_flight_items: The maximum number of items allowed to be "in flight"
            at the same time. See `block_until_num_items` in
            `reverb.TrajectoryWriter.flush` for more info.

        Raises:
          ValueError: If n_step is less than 1.
        """
        # Makes the additional discount a float32, which means that it will be
        # upcast if rewards/discounts are float64 and left alone otherwise.
        self.n_step = n_step
        self._discount = tree.map_structure(np.float32, discount)
        self._first_idx = 0
        self._last_idx = 0

        raise NotImplementedError("This class still considers discounts explicitly, needs to be refactored.")

        super().__init__(
            client=client,
            max_sequence_length=n_step + 1,
            priority_fns=priority_fns,
            max_in_flight_items=max_in_flight_items,
        )

    def add(self, timestep: worlds.TimeStep, action: _types.Tree = None, extras: _types.Tree = ()):
        if not timestep.first():
            # Increment the indices for the start and end of the window for computing
            # n-step returns.
            if self._writer.episode_steps >= self.n_step:
                self._first_idx += 1
            self._last_idx += 1

        super().add(timestep=timestep, action=action, extras=extras)

    def reset(self):
        super().reset()
        self._first_idx = 0
        self._last_idx = 0

    @property
    def _n_step(self) -> int:
        """Effective n-step, which may vary at starts and ends of episodes."""
        return self._last_idx - self._first_idx

    def _write(self):
        # Convenient getters for use in tree operations.
        get_first = lambda x: x[self._first_idx]
        get_last = lambda x: x[self._last_idx]
        # Note: this getter is meant to be used on a TrajectoryWriter.history to
        # obtain its numpy values.
        get_all_np = lambda x: x[self._first_idx : self._last_idx].numpy()

        # Get the state, action, next_state, as well as possibly extras for the
        # transition that is about to be written.
        history = self._writer.history
        s, a = tree.map_structure(get_first, (history["observation"], history["action"]))
        s_ = tree.map_structure(get_last, history["observation"])

        # Maybe get extras to add to the transition later.
        if "extras" in history:
            extras = tree.map_structure(get_first, history["extras"])

        # Note: at the beginning of an episode we will add the initial N-1
        # transitions (of size 1, 2, ...) and at the end of an episode (when
        # called from write_last) we will write the final transitions of size (N,
        # N-1, ...). See the Note in the docstring.
        # Get numpy view of the steps to be fed into the priority functions.
        reward, discount = tree.map_structure(get_all_np, (history["reward"], history["discount"]))

        # Compute discounted return and geometric discount over n steps.
        n_step_return, total_discount = self._compute_cumulative_quantities(reward, discount)

        # Append the computed n-step return and total discount.
        # Note: if this call to _write() is within a call to _write_last(), then
        # this is the only data being appended and so it is not a partial append.
        self._writer.append(
            dict(n_step_return=n_step_return, total_discount=total_discount),
            partial_step=self._writer.episode_steps <= self._last_idx,
        )
        # This should be done immediately after self._writer.append so the history
        # includes the recently appended data.
        history = self._writer.history

        # Form the n-step transition by using the following:
        # the first observation and action in the buffer, along with the cumulative
        # reward and discount computed above.
        n_step_return = tree.map_structure(lambda x: x[-1], history["n_step_return"])
        transition = _types.Transition(
            observation=s,
            action=a,
            reward=n_step_return,
            next_observation=s_,
            extras=(extras if "extras" in history else ()),
        )

        # Calculate the priority for this transition.
        table_priorities = utils.calculate_priorities(self._priority_fns, transition)

        # Insert the transition into replay along with its priority.
        for table, priority in table_priorities.items():
            self._writer.create_item(table=table, priority=priority, trajectory=transition)
            self._writer.flush(self._max_in_flight_items)

    def _write_last(self):
        # Write the remaining shorter transitions by alternating writing and
        # incrementingfirst_idx. Note that last_idx will no longer be incremented
        # once we're in this method's scope.
        self._first_idx += 1
        while self._first_idx < self._last_idx:
            self._write()
            self._first_idx += 1

    def _compute_cumulative_quantities(
        self, rewards: _types.Tree, discounts: _types.Tree
    ) -> Tuple[_types.Tree, _types.Tree]:

        # Give the same tree structure to the n-step return accumulator,
        # n-step discount accumulator, and self.discount, so that they can be
        # iterated in parallel using tree.map_structure.
        rewards, discounts, self_discount = tree_utils.broadcast_structures(rewards, discounts, self._discount)
        flat_rewards = tree.flatten(rewards)
        flat_discounts = tree.flatten(discounts)
        flat_self_discount = tree.flatten(self_discount)

        # Copy total_discount as it is otherwise read-only.
        total_discount = [np.copy(a[0]) for a in flat_discounts]

        # Broadcast n_step_return to have the broadcasted shape of
        # reward * discount.
        n_step_return = [
            np.copy(np.broadcast_to(r[0], np.broadcast(r[0], d).shape)) for r, d in zip(flat_rewards, total_discount)
        ]

        # NOTE: total_discount will have one less self_discount applied to it than
        # the value of self._n_step. This is so that when the learner/update uses
        # an additional discount we don't apply it twice. Inside the following loop
        # we will apply this right before summing up the n_step_return.
        for i in range(1, self._n_step):
            for nsr, td, r, d, sd in zip(
                n_step_return, total_discount, flat_rewards, flat_discounts, flat_self_discount
            ):
                # Equivalent to: `total_discount *= self._discount`.
                td *= sd
                # Equivalent to: `n_step_return += reward[i] * total_discount`.
                nsr += r[i] * td
                # Equivalent to: `total_discount *= discount[i]`.
                td *= d[i]

        n_step_return = tree.unflatten_as(rewards, n_step_return)
        total_discount = tree.unflatten_as(rewards, total_discount)
        return n_step_return, total_discount

    # TODO(bshahr): make this into a standalone method. Class methods should be
    # used as alternative constructors or when modifying some global state,
    # neither of which is done here.
    @classmethod
    def signature(cls, environment_spec: worlds.EnvironmentSpec, extras_spec: worlds.TreeSpec = ()):
        transition_spec = _types.Transition(
            environment_spec.observation,
            environment_spec.action,
            environment_spec.reward,
            environment_spec.observation,  # next_observation
            extras_spec,
        )
        return tree.map_structure_with_path(reverb_adder.spec_like_to_tensor_spec, transition_spec)


def _broadcast_specs(*args: worlds.ArraySpec) -> worlds.ArraySpec:
    """Like np.broadcast, but for specs.Array.

    Args:
      *args: one or more specs.Array instances.

    Returns:
      A specs.Array with the broadcasted shape and dtype of the specs in *args.
    """
    bc_info = np.broadcast(*tuple(a.generate_value() for a in args))
    dtype = np.result_type(*tuple(a.dtype for a in args))
    return worlds.ArraySpec(shape=bc_info.shape, dtype=dtype)
