"""Interface for services that simulate episodes."""
import abc


class ArenaInterface:
    """Interface for a agent-environment interaction (arena) services."""

    @abc.abstractmethod
    def run_episode(self):
        """Run a single episode of interaction."""

    def run_episodes(self, *args, num_episodes: int, **kwargs):
        """ "Run several episodes of interaction.

        Args:
            num_episodes: Number of episodes to run.

        Returns:
            The results from each episode.
        """
        return [self.run_episode(*args, **kwargs) for _ in range(num_episodes)]
