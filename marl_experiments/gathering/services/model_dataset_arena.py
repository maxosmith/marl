import dataclasses
import warnings
from typing import Mapping, Optional

import numpy as np

from marl import _types, individuals, worlds
from marl.services.arenas import base
from marl.services.replay.reverb import adders as reverb_adders
from marl.utils import signals


@dataclasses.dataclass
class Arena(base.ArenaInterface):
    """Training arenas play games between learning agents and non-learning agents (bots).

    Attributes:
        game:
        players:
    """

    game: worlds.Game
    players: Mapping[_types.PlayerID, individuals.Individual]
    reverb_adder: reverb_adders.SequenceAdder

    def run_episode(self):
        """Run one episode."""
        timesteps = self.game.reset()
        states = {id: None for id in self.players.keys()}
        length = 0

        while not np.any([ts.last() for ts in timesteps.values()]):
            # Action selection.
            actions = {}
            for id, player in self.players.items():
                actions[id], states[id] = player.step(timesteps[id], states[id])

            self.reverb_adder.add(
                timestep=timesteps[0],
                action=np.asarray(actions[0], dtype=np.int32),
                extras=np.array([actions[0], actions[1]], dtype=np.int32),
            )

            # Environment transition.
            timesteps = self.game.step(actions)
            length += 1

        # Add the final timestep, with dummy action/extras.
        self.reverb_adder.add(
            timestep=timesteps[0],
            action=np.zeros_like(actions[0], dtype=np.int32),
            extras=np.zeros_like(np.array([actions[0], actions[1]], dtype=np.int32)),
        )

        return dict(episode_length=length)

    def run(self, num_episodes: Optional[int] = None, num_timesteps: Optional[int] = None):
        """Runs many episodes serially.

        Runs either `num_episodes` or at least `num_timesteps` timesteps (episodes are run until completion,
        so the true number of timesteps run will include completing the last episode). If neither are arguments
        are specified then this continues to run episodes infinitely.

        Args:
            num_episodes: number of episodes to run.
            num_timesteps: minimum number of timesteps to run.
        """
        if not (num_episodes is None or num_timesteps is None):
            warnings.warn("Neither `num_episodes` nor `num_timesteps` were specified, running indefinitely.")

        def _should_terminate(episodes: int, timesteps: int) -> bool:
            episodes_finished = (num_episodes is not None) and (episodes >= num_episodes)
            timesteps_finished = (num_timesteps is not None) and (timesteps >= num_timesteps)
            return episodes_finished or timesteps_finished

        episode_count, timestep_count = 0, 0

        with signals.runtime_terminator():
            while not _should_terminate(episodes=episode_count, timesteps=timestep_count):
                result = self.run_episode()
                episode_count += 1
                timestep_count += result["episode_length"]
