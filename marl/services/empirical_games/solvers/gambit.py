""" Wrapper to solve game through Gambit.

References:
    - http://www.gambit-project.org/gambit15/tools.html
    - http://www.gambit-project.org/gambit13/formats.html
"""
import logging
import os
import os.path as osp
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict

import numpy as np

from marl import _types

logger = logging.getLogger(__name__)


class GambitAlgorithms(Enum):
    """Gambit equilibrium solving algorithms."""

    ENUM_PURE = "enumpure"
    ENUM_POLY = "enumpoly"
    ENUM_MIXED = "enummixed"
    GNM = "gnm"
    IPA = "ipa"
    LCP = "lcp"
    LP = "lp"
    LIAP = "liap"
    SIMP_DIV = "simpdiv"


_TWO_PLAYER_ALGORITHMS = frozenset([GambitAlgorithms.ENUM_MIXED, GambitAlgorithms.LCP, GambitAlgorithms.LP])


def _assert_valid_player_count(algorithm: str, payoffs: np.ndarray):
    """Assert that the algorithm supports the number of players provided.

    Args:
        payoffs: Game matrix of shape [|Pi_0|, |Pi_1|, ..., |Pi_{n-1}|, n].
    """
    num_players = payoffs.shape[-1]
    assert (num_players != 0) and (num_players != 1), "Gambit not defined for games with less than 2 players."
    if num_players != 2:
        assert (
            algorithm not in _TWO_PLAYER_ALGORITHMS
        ), f"Gambit-{algorithm} does not support >2 players ({num_players} players detected)."


@dataclass
class Gambit:
    """Solve for Nash equilibrium using Gambit."""

    timeout: int = 3600
    algorithm: GambitAlgorithms = GambitAlgorithms.LCP

    def __call__(self, payoffs: np.ndarray, **kwargs) -> Dict[_types.PlayerID, np.ndarray]:
        """Calculate an equilibrium from a game matrix.

        Args:
            payoffs: Game matrix of shape [|Pi_0|, |Pi_1|, ..., |Pi_{n-1}|, n].

        Returns:
            Mixed-strategy for each agent.
        """
        _assert_valid_player_count(self.algorithm, payoffs)

        with tempfile.TemporaryDirectory() as temp_dir:
            nfg_file = osp.join(temp_dir, "game.nfg")
            nash_file = osp.join(temp_dir, "nash.txt")

            self._write_nfg_file(payoffs, nfg_file)
            self._run_gambit_analysis(nfg_file, nash_file, len(payoffs.shape) - 1)
            strategies = self._read_nash_file(nash_file, payoffs.shape[:-1])
            return {agent_id: mixture for agent_id, mixture in enumerate(strategies)}

    def _write_nfg_file(self, payoffs: np.ndarray, path: str):
        """Write payoffs into a normal-form game file."""
        assert path[-4:] == ".nfg", "Must save as NFG file format."

        with open(path, "w") as nfg_file:
            # Designate file as NFG file version 1 using rational numbers (R).
            nfg_file.write('NFG 1 R "GambitAnalyzer"\n')

            # Generate prologue information about agents.
            num_agents = len(payoffs.shape) - 1
            name_line = "{ "
            num_policies_line = "{ "
            for agent_id in range(num_agents):
                name_line += f'"{agent_id}" '
                num_policies_line += f"{payoffs.shape[agent_id]} "
            name_line += "}"
            num_policies_line += "}"

            nfg_file.write(name_line)
            nfg_file.write(f"{num_policies_line}\n\n")

            # for col in range(payoffs.shape[1]):
            #     for row in range(payoffs.shape[0]):
            #         nfg_file.write(str(payoffs[row][col][0]) + " ")
            #         nfg_file.write(str(payoffs[row][col][1]) + " ")

            # Write the payoffs to the file.
            new_axes = list(range(len(payoffs.shape)))
            new_axes[:-1] = reversed(new_axes[:-1])
            payoffs = np.transpose(payoffs, new_axes)
            payoffs = " ".join(np.ravel(payoffs).astype(str))
            payoffs += " "  # Gambit needs this trailing whitespace.
            nfg_file.write(payoffs)

    def _read_nash_file(self, path: str, num_agent_policies: np.ndarray):
        """Read the Nash output of Gambit."""
        assert osp.exists(path), "Nash equilibrium file does not exist."

        with open(path, "r") as nash_file:
            nash = nash_file.readline()
            print(nash)
            if len(nash.strip()) == 0:
                return [np.array(0) for _ in num_agent_policies]

        # Remove header and split each strategy weight.
        nash = nash[3:]
        nash = nash.split(",")

        # Convert strategy weights from string to floats.
        for i in range(len(nash)):
            try:
                nash[i] = float(nash[i])
            except ValueError:
                num, denom = nash[i].split("/")
                nash[i] = float(num) / float(denom)

        # Row player's equilibrium is first, then column.
        nash = np.round(nash, decimals=4)
        strategies = []
        index = 0
        for num_policies in num_agent_policies:
            strategies += [nash[index : index + num_policies]]
            index += num_policies
        return strategies

    def _run_gambit_analysis(self, nfg_path: str, nash_path: str):
        """Run gambit analsys on the current empirical game."""
        assert osp.exists(nfg_path), "NFG file not found."

        command = f"gambit-{self.algorithm.value} "
        command += "-q "
        command += "-d 4 "
        command += f"{nfg_path} "
        command += f"> {nash_path}"

        call_and_wait_with_timeout(command, self.timeout)


def call_and_wait_with_timeout(command: str, timeout: int):
    logger.info(f"Launching subprocess: {command}")
    process = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)

    try:
        process.wait(timeout=timeout)

    except subprocess.TimeoutExpired:
        logger.critical(f"Subprocess ran for more than {timeout} seconds.")
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        logger.critical("Subprocess has been killed.")
        exit(1)

    time.sleep(5)  # Seconds.
    process.kill()
