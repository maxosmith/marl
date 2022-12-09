import itertools

import numpy as np
import zarr

from marl_experiments.roshambo import roshambo_bot
from marl_experiments.roshambo.coplay_main import Demonstration

num_bots = len(roshambo_bot.ROSHAMBO_BOT_NAMES)
dataset = zarr.open("/scratch/wellman_root/wellman1/mxsmith/data/roshambo/demonstrations2.zarr", mode="r")


for row, col in itertools.product(np.arange(num_bots), repeat=2):
    unique = {}
    profile_conflicts = 0

    for episode in range(1_000):
        for timestep in range(1_000):
            observation = dataset["observations"][row, col, episode, timestep]
            observation = "".join(observation.astype(str))

            action = dataset["actions"][row, col, episode, timestep, 0]

            if (observation in unique) and (unique[observation] == action):
                pass

            elif (observation in unique) and (unique[observation] != action):
                profile_conflicts += 1

            else:
                unique[observation] = action

    print(f"{row}, {col}: {profile_conflicts}")
