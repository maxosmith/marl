import functools
import itertools
import pickle
from typing import Mapping, Optional, Sequence

import numpy as np
import torch
import zarr

from marl import _types
from marl_experiments.roshambo import roshambo_bot
from marl_experiments.roshambo.coplay_main import Demonstration


def build_dataset(coplay_result_dir: str, filepath: str):
    """Builds a Zarr dataset from coplay demonstrations.

    Args:
        coplay_result_dir: Directory containing the coplay demonstrations from `coplay_main.py`.
        filepath: Filepath for resulting dataset.
    """
    num_episodes = 1000
    num_bots = len(roshambo_bot.ROSHAMBO_BOT_NAMES)
    episode_length = 1001
    prefix = (num_bots, num_bots, num_episodes, episode_length)

    dataset = zarr.open(filepath, mode="w")
    observations = dataset.zeros("observations", shape=prefix + (6,), dtype=np.int32)
    rewards = dataset.zeros("rewards", shape=prefix + (2,), dtype=np.int32)
    actions = dataset.zeros("actions", shape=prefix + (2,), dtype=np.int32)
    step_type = dataset.zeros("step_type", shape=prefix, dtype=np.int32)

    bot_names = roshambo_bot.ROSHAMBO_BOT_NAMES

    for name0, name1 in itertools.product(bot_names, repeat=2):
        demonstration: Demonstration = pickle.load(open(f"{coplay_result_dir}/{name0}_vs_{name1}.pb", "rb"))

        idx0 = bot_names.index(name0)
        idx1 = bot_names.index(name1)

        observations[idx0, idx1] = demonstration.observation.astype(np.int32)
        rewards[idx0, idx1] = demonstration.rewards.astype(np.int32)
        actions[idx0, idx1] = demonstration.actions.astype(np.int32)
        step_type[idx0, idx1] = demonstration.step_type.astype(np.int32)


class BCDataset(torch.utils.data.Dataset):
    """Behavioural cloning dataset.

    NOTE: Assumes that the demonstrator is always playing in position 0.
    """

    def __init__(
        self,
        path: str,
        *,
        sequence_len: int,
        period_len: Optional[int] = None,
        bot_names: Optional[Sequence[str]] = None,
        opponent_names: Optional[Sequence[str]] = None,
        episode_length: Optional[int] = None,
    ):
        """Initialize an instance of `BCDataset`.

        Args:
            path: Path to the Zarr dataset containing demonstration data.
            sequence_len: Length of episode sub-sequences.
            period_len: Period of sub-sequences within an episode. If not specified,
                it's assumed to be one less than the sequence length, so that sequence
                overlap by one.
            bot_names: Bot names that are included as demonstrators.
            opponent_names: Bot names that are included as opponents against the demonstrators.
            episode_length: Length of an episode. If not specified, it's assumed to be
                the default number of throws specified by OpenSpiel (1_000).
        """
        self._dataset = zarr.open(path, mode="r")
        self._sequence_len = sequence_len
        self._period_len = period_len if period_len else (sequence_len - 1)
        self._bot_names = bot_names if bot_names else roshambo_bot.ROSHAMBO_BOT_NAMES
        self._opponent_names = opponent_names if opponent_names else roshambo_bot.ROSHAMBO_BOT_NAMES
        self._episode_length = episode_length if episode_length else roshambo_bot.ROSHAMBO_NUM_THROWS

    def __len__(self):
        """Length of the dataset."""
        return self._num_profiles * self._num_episodes * self._num_examples_per_episode

    @functools.cached_property
    def _num_profiles(self):
        """Number of profiles played within the dataset."""
        return len(self._bot_names) * len(self._opponent_names)

    @functools.cached_property
    def _num_episodes(self):
        """Number of demonstration epsiodes per profile."""
        return int(self._dataset["step_type"].shape[2])

    @property
    def _num_examples_per_episode(self):
        """Number of examples that can be generated from a single episode."""
        return np.ceil(self._episode_length / self._period_len).astype(int)

    def __getitem__(self, idx: int):
        """Fetch an item from the Dataset."""
        # The index is an int encoding for (Demonstrator, Opponent, Episode, Example).
        bot, opponent, episode_idx, example_idx = np.unravel_index(
            idx, [len(self._bot_names), len(self._opponent_names), self._num_episodes, self._num_examples_per_episode]
        )

        # Bot and opponent need to be converted from the partitioned version of the dataset to
        # the full-sized dataset.
        bot = roshambo_bot.ROSHAMBO_BOT_NAMES.index(self._bot_names[bot])
        opponent = roshambo_bot.ROSHAMBO_BOT_NAMES.index(self._opponent_names[opponent])

        # An example is a periodic subtrajectory of an episode.
        example_slice = slice(example_idx * self._period_len, (example_idx * self._period_len) + self._sequence_len)

        # Slice all fields of the dataset.
        item = {key: value[bot, opponent, episode_idx, example_slice] for key, value in self._dataset.items()}
        item = self._maybe_pad_item(item)
        item["padding_mask"] = self._padding_mask(item)
        item["bot_ids"] = np.tile([bot, opponent], (self._sequence_len, 1))
        return item

    def _maybe_pad_item(self, item: Mapping[str, _types.Array]) -> Mapping[str, _types.Array]:
        """Pad an item to match the sequence length.

        Args:
            item: Dictionary of features of shape [T, ...].

        Returns:
            Dictionary augmented with zero padding so all features have shape [SEQ_LEN, ...].
        """
        for key, value in item.items():
            # Determine the amount of padding, if any.
            padding_size = self._sequence_len - value.shape[0]
            if not padding_size:
                break

            padding_shape = (padding_size,) + value.shape[1:] if value.shape[1:] else [padding_size]
            padding = np.zeros(padding_shape, dtype=value.dtype)
            item[key] = np.concatenate([value, padding], axis=0)
        return item

    @staticmethod
    def _padding_mask(item: Mapping[str, _types.Array]) -> _types.Array:
        """Builds a padding mask for the item."""
        end_of_episode = item["step_type"] == 2
        # Put >=1s on the last step and all padding.
        mask = np.cumsum(end_of_episode, axis=-1)
        mask = np.logical_not(mask)
        return mask
