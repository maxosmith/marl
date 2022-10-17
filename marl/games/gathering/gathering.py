"""Common-pool resource allocation environments.

References:
 - https://github.com/HumanCompatibleAI/multi-agent
 - https://arxiv.org/pdf/1707.06600.pdf
"""
from __future__ import absolute_import, division, print_function

import collections
import copy
import enum
import itertools
import os.path as osp
import tkinter as tk
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from marl import _types, worlds


class GatheringActions(enum.IntEnum):
    """Actions available in `Gathering`."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    ROTATE_RIGHT = 4
    ROTATE_LEFT = 5
    LASER = 6
    NOOP = 7


@dataclass
class Gathering(worlds.Game):

    n_agents: int = 1
    map_name: str = "default"
    symmetric: bool = False
    scale: int = 20
    global_observation: bool = False
    viewbox_width: int = 5
    viewbox_depth: int = 5
    agent_colors: List[Tuple] = field(default_factory=lambda: [(0, 0, 255), (255, 0, 0)])
    beam_width: int = 3
    beam_length: int = 1
    timeout_length: int = 10

    metadata = {"render.modes": ["image", "text"]}

    def __post_init__(self):
        self.name = f"gathering_{self.map_name}"
        self.root = None
        self.padding = max(self.viewbox_width // 2, self.viewbox_depth - 1)

        # Read the map configuration from file.
        map_path = osp.dirname(osp.abspath(__file__))
        map_path = osp.join(map_path, "maps", f"{self.map_name}.txt")
        if not osp.exists(map_path):
            raise ValueError(f"Map file not found: {map_path}")
        with open(map_path) as f:
            self._text_to_map(f.read().strip())

        self.width = self.initial_food.shape[0]
        self.height = self.initial_food.shape[1]

        if self.global_observation:
            self.state_size = self.walls.shape[0] * self.walls.shape[1] * 4
        else:
            self.state_size = self.viewbox_width * self.viewbox_depth * 4

        self.reset()
        self.done = False

    def reset(self) -> worlds.PlayerIDToTimestep:
        """Returns the first `TimeStep` of a new episode."""
        observations = self._reset()
        return {
            i: worlds.TimeStep(
                observation=observations[i].astype(np.int32),
                reward=np.asarray(0.0, dtype=np.float32),
                step_type=worlds.StepType.FIRST,
            )
            for i in range(self.n_agents)
        }

    def step(self, actions: _types.Action) -> worlds.PlayerIDToTimestep:
        """Updates the environment according to the action."""
        # Convert GatheringActions to integers and stack.
        actions = np.stack(
            [
                actions[i] if not isinstance(actions[i], GatheringActions) else actions[i].value
                for i in range(self.n_agents)
            ]
        )

        # Perform the underlying transition in the original environment.
        os, rs, *_ = self._step(actions)

        # Convert back into DM environment data structures.
        os = os.astype(np.int32)
        rs = np.array(rs).astype(np.float32)

        timesteps = {}
        for agent_id in range(self.n_agents):
            if self.done:
                timesteps[agent_id] = worlds.TimeStep(
                    step_type=worlds.StepType.LAST,
                    reward=rs[agent_id],
                    observation=os[agent_id],
                )
            else:
                timesteps[agent_id] = worlds.TimeStep(
                    step_type=worlds.StepType.MID,
                    reward=rs[agent_id],
                    observation=os[agent_id],
                )
        return timesteps

    def observation_specs(self) -> worlds.PlayerIDToSpec:
        """Returns the observation spec."""
        if self.global_observation:
            shape = [self.walls.shape[0], self.walls.shape[1], 4]
        else:
            shape = [self.viewbox_width, self.viewbox_depth, 4]
        spec = worlds.ArraySpec(dtype=np.int32, shape=shape, name="state")
        return {i: spec for i in range(self.n_agents)}

    def action_specs(self) -> worlds.PlayerIDToSpec:
        """Returns the action spec."""
        spec = worlds.DiscreteArraySpec(dtype=np.int32, num_values=8, name="action")
        return {i: spec for i in range(self.n_agents)}

    def reward_specs(self) -> worlds.PlayerIDToSpec:
        """Describes the reward returned by the environment."""
        return {id: worlds.ArraySpec(shape=(), dtype=np.float32, name="reward") for id in range(self.n_agents)}

    # ---------------------------------------------

    def _text_to_map(self, text):
        m = [list(row) for row in text.splitlines()]
        l = len(m[0])
        for row in m:
            if len(row) != l:
                raise ValueError("the rows in the map are not all the same length")

        def pad(a):
            return np.pad(a, self.padding + 1, "constant")

        a = np.array(m).T
        self.initial_food = pad(a == "O").astype(np.int)
        self.walls = pad(a == "#").astype(np.int)

    def _step(self, action_n):
        assert len(action_n) == self.n_agents
        action_n = [GatheringActions.NOOP if self.tagged[i] else a for i, a in enumerate(action_n)]
        self.beams[:] = 0
        movement_n = [(0, 0) for a in action_n]
        for i, (a, orientation) in enumerate(zip(action_n, self.orientations)):
            if GatheringActions(a) not in [
                GatheringActions.UP,
                GatheringActions.DOWN,
                GatheringActions.LEFT,
                GatheringActions.RIGHT,
            ]:
                continue
            # a is relative to the agent's orientation, so add the orientation
            # before interpreting in the global coordinate system.
            #
            # This line is really not obvious to read. Replace it with something
            # clearer if you have a better idea.
            a = (a + orientation) % 4
            movement_n[i] = [(0, -1), (1, 0), (0, 1), (-1, 0),][
                a
            ]  # up/forward  # right  # down/backward  # left
        next_locations = [a for a in self.agents]
        next_locations_map = collections.defaultdict(list)
        for agent_i, ((dx, dy), (x, y)) in enumerate(zip(movement_n, self.agents)):
            if self.tagged[agent_i]:
                continue
            next_ = ((x + dx), (y + dy))
            if self.walls[next_]:
                next_ = (x, y)
            next_locations[agent_i] = next_
            next_locations_map[next_].append(agent_i)
        for overlappers in next_locations_map.values():
            if len(overlappers) > 1:
                for agent_i in overlappers:
                    next_locations[agent_i] = self.agents[agent_i]

        self.agents = next_locations

        # Determine if more than one agent will be tagged.
        tagged_area = np.copy(self.beams)
        shuffled_ids = [agent_id for agent_id, _ in enumerate(action_n)]
        np.random.shuffle(shuffled_ids)
        for agent_id in shuffled_ids:
            action = action_n[agent_id]

            if GatheringActions(action) == GatheringActions.LASER:
                tagged_area[self._viewbox_slice(agent_id, self.beam_width, self.beam_length, offset=1)] = 1

            # Determine if any agents' actions are effected by this laser.
            for opponent_id, _ in enumerate(action_n):
                if opponent_id == agent_id:
                    continue
                if tagged_area[self.agents[opponent_id]]:
                    # Change LASER action to NOOP for tagged agents.
                    # print(f"Changing {opponent_id}'s action to NOOP.")
                    action_n[opponent_id] = GatheringActions.NOOP

        for i, act in enumerate(action_n):
            act = GatheringActions(act)
            if act == GatheringActions.ROTATE_RIGHT:
                self.orientations[i] = (self.orientations[i] + 1) % 4
            elif act == GatheringActions.ROTATE_LEFT:
                self.orientations[i] = (self.orientations[i] - 1) % 4
            elif act == GatheringActions.LASER:
                self.beams[self._viewbox_slice(i, self.beam_width, self.beam_length, offset=1)] = 1

        obs_n = self.state_n
        reward_n = [0 for _ in range(self.n_agents)]
        done_n = [self.done] * self.n_agents
        info_n = [{}] * self.n_agents

        self.food = (self.food + self.initial_food).clip(max=1)

        for i, a in enumerate(self.agents):
            if self.tagged[i]:
                continue
            if self.food[a] == 1:
                self.food[a] = -15
                reward_n[i] = 1
            if self.beams[a]:
                self.tagged[i] = self.timeout_length

        for i, tag in enumerate(self.tagged):
            if tag == 1:
                # Relocate to a respawn point.
                for spawn_point in self.spawn_points:
                    if spawn_point not in self.agents:
                        self.agents[i] = spawn_point
                        break

        self.tagged = [max(i - 1, 0) for i in self.tagged]

        return obs_n, reward_n, done_n, info_n

    def _viewbox_slice(self, agent_index, width, depth, offset=0):
        left = width // 2
        right = left if width % 2 == 0 else left + 1
        x, y = self.agents[agent_index]
        return tuple(
            itertools.starmap(
                slice,
                (
                    ((x - left, x + right), (y - offset, y - offset - depth, -1)),  # up
                    ((x + offset, x + offset + depth), (y - left, y + right)),  # right
                    ((x + left, x - right, -1), (y + offset, y + offset + depth)),  # down
                    ((x - offset, x - offset - depth, -1), (y + left, y - right, -1)),  # left
                )[self.orientations[agent_index]],
            )
        )

    @property
    def state_n(self):
        agents = np.zeros_like(self.food)
        for i, a in enumerate(self.agents):
            if not self.tagged[i]:
                agents[a] = 1

        food = self.food.clip(min=0)

        if self.global_observation:
            states = np.zeros((self.n_agents, self.walls.shape[0], self.walls.shape[1], 4))
        else:
            states = np.zeros((self.n_agents, self.viewbox_width, self.viewbox_depth, 4))

        for i, (orientation, (x, y)) in enumerate(zip(self.orientations, self.agents)):
            if self.tagged[i]:
                continue
            full_state = np.stack([food, np.zeros_like(food), agents, self.walls], axis=-1)
            full_state[x, y, 1] = 1
            full_state[x, y, 2] = 0

            if self.global_observation:
                states[i] = full_state

            else:
                # Slice the full state into a rectangular region in front of the agent.
                xs, ys = self._viewbox_slice(i, self.viewbox_width, self.viewbox_depth)
                observation = full_state[xs, ys, :]

                states[i] = (
                    observation
                    if orientation in [GatheringActions.UP.value, GatheringActions.DOWN.value]
                    else observation.transpose(1, 0, 2)
                )

        # return states.reshape((self.n_agents, self.state_size))
        return states

    def _reset(self):
        self.food = self.initial_food.copy()

        p = self.padding
        self.walls[p:-p, p] = 1
        self.walls[p:-p, -p - 1] = 1
        self.walls[p, p:-p] = 1
        self.walls[-p - 1, p:-p] = 1

        self.beams = np.zeros_like(self.food)

        if self.symmetric:
            self.agents = [(i + self.padding + 1, self.padding + 1) for i in range(self.n_agents)]
            np.random.shuffle(self.agents)
            self.spawn_points = copy.copy(self.agents)
        else:
            self.agents = [(i + self.padding + 1, self.padding + 1) for i in range(self.n_agents)]
            self.spawn_points = list(self.agents)
        self.orientations = [GatheringActions.UP.value for _ in self.agents]
        self.tagged = [0 for _ in self.agents]

        return self.state_n

    def _close_view(self):
        if self.root:
            self.root.destroy()
            self.root = None
            self.canvas = None
        self.done = True

    def render(self, mode: str = "text"):
        """."""
        if mode == "image":
            return self._render_image()
        elif mode == "text":
            return self._render_text()
        elif mode == "observation":
            return self._render_observation()
        else:
            raise ValueError(f"Unknown mode: '{mode}'.")

    def _render_text(self):
        canvas_width = self.width
        canvas_height = self.height

        img = np.array([[" " for _ in range(canvas_height)] for _ in range(canvas_width)])

        # Draw in the environment.
        for x in range(self.width):
            for y in range(self.height):
                if self.beams[x, y] == 1:
                    img[x, y] = "X"
                if self.food[x, y] == 1:
                    img[x, y] = "O"
                if self.walls[x, y] == 1:
                    img[x, y] = "#"

        # Add agents onto environment.
        for agent_id, (x, y) in enumerate(self.agents):
            if not self.tagged[agent_id]:
                img[x, y] = str(agent_id)

        img = img.T
        img = ["".join(row) for row in img]
        img = "\n".join(img)
        return img

    def _render_image(self):
        img = Image.new("RGB", (self.width * self.scale, self.height * self.scale), color="black")
        draw = ImageDraw.Draw(img)
        scale = self.scale

        # Draw in the environment.
        for x in range(self.width):
            for y in range(self.height):
                if self.beams[x, y] == 1:
                    draw.rectangle([(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=(255, 255, 0))
                if self.food[x, y] == 1:
                    draw.rectangle([(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=(0, 255, 0))
                if self.walls[x, y] == 1:
                    draw.rectangle([(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=(80, 80, 80))

        # Add agents onto environment.
        for agent_id, (x, y) in enumerate(self.agents):
            if not self.tagged[agent_id]:
                draw.rectangle(
                    [(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=self.agent_colors[agent_id]
                )

        return img

    def _render_observation(self):
        img = Image.new("RGB", (self.viewbox_width * self.scale, self.viewbox_depth * self.scale), color="black")
        draw = ImageDraw.Draw(img)
        scale = self.scale

        p1_state = self.state_n[0].reshape(self.viewbox_width, self.viewbox_depth, 4)
        p1_state = p1_state[:, ::-1]
        for x in range(self.viewbox_width):
            for y in range(self.viewbox_depth):
                food, me, other, wall = p1_state[x, y]
                assert sum((food, me, other, wall)) <= 1
                if food:
                    draw.rectangle([(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=(0, 255, 0))
                elif me:
                    draw.rectangle(
                        [(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=self.agent_colors[0]
                    )
                elif other:
                    draw.rectangle(
                        [(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=self.agent_colors[1]
                    )
                elif wall:
                    draw.rectangle([(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=(80, 80, 80))

        # Draw "me".
        x = self.viewbox_width // 2
        y = self.viewbox_depth - 1
        draw.rectangle([(x * scale, y * scale), ((x + 1) * scale, (y + 1) * scale)], fill=self.agent_colors[0])
        return img

    def _render(self, mode="human", close=False):
        if close:
            self._close_view()
            return

        canvas_width = self.width * self.scale
        canvas_height = self.height * self.scale

        if self.root is None:
            self.root = tk.Tk()
            self.root.title("Gathering")
            self.root.protocol("WM_DELETE_WINDOW", self._close_view)
            self.canvas = tk.Canvas(self.root, width=canvas_width, height=canvas_height)
            self.canvas.pack()

        self.canvas.delete(tk.ALL)
        self.canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="black")

        def fill_cell(x, y, color):
            self.canvas.create_rectangle(
                x * self.scale,
                y * self.scale,
                (x + 1) * self.scale,
                (y + 1) * self.scale,
                fill=color,
            )

        for x in range(self.width):
            for y in range(self.height):
                if self.beams[x, y] == 1:
                    fill_cell(x, y, "yellow")
                if self.food[x, y] == 1:
                    fill_cell(x, y, "green")
                if self.walls[x, y] == 1:
                    fill_cell(x, y, "grey")

        for i, (x, y) in enumerate(self.agents):
            if not self.tagged[i]:
                fill_cell(x, y, self.agent_colors[i])

        if mode == "viewbox":
            # Debug view: see the first player's viewbox perspective.
            p1_state = self.state_n[0].reshape(self.viewbox_width, self.viewbox_depth, 4)
            for x in range(self.viewbox_width):
                for y in range(self.viewbox_depth):
                    food, me, other, wall = p1_state[x, y]
                    assert sum((food, me, other, wall)) <= 1
                    y_ = self.viewbox_depth - y - 1
                    if food:
                        fill_cell(x, y_, "green")
                    elif me:
                        fill_cell(x, y_, "cyan")
                    elif other:
                        fill_cell(x, y_, "red")
                    elif wall:
                        fill_cell(x, y_, "gray")
            self.canvas.create_rectangle(
                0,
                0,
                self.viewbox_width * self.scale,
                self.viewbox_depth * self.scale,
                outline="blue",
            )

        self.root.update()

    def _close(self):
        self._close_view()
