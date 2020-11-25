import os
import sys
from typing import List
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from enviroments import Environment, Episode

class SimpleContinuous(object):
    """
    Simple one step environment with a fixed best action solution.
    Used for test purposes.
    """

    def __init__(self, step_target: int, max_reward_step: float, steps: int = 1):
        """
        :param target: value to get to
        :param max_reward_step: maximum reward per each step
        :param target_range: maximum error allowed
        :param steps: amount of steps the game has
        """

        self.steps = steps
        self.step_target = step_target
        self.target = step_target * steps
        self.sum_target = 0
        self.max_reward_step = max_reward_step
        self.state = None
        self.game_over = False
        self.target = 0

    def get_state(self):
        if self.state is None:
            self.state = np.array([1])
        else:
            self.state.append(len(self.state) + 1)

    def step(self, action=float):
        reward = -((action - 4.0) ** 2) + 1.0

        self.sum_target += action
        self.get_state()

        state = np.array([self.state[len(self.state) - 1]])

        if self.state[len(self.state) - 1] == self.steps:
            self.game_over = True

        return state, reward, self.game_over

class SimpleContinuousEnvironment(Environment):

    def __init__(self):
        env = SimpleContinuous(target=4., max_reward_step=1., steps=1)
        action_space = 1
        state_space = 1
        actions = ["action"]
        state_names = ["unique_state"]

        Environment.__init__(self, env, action_space, state_space,
                             actions, state_names, action_space="continuous")

    def reset_environment(self):
        pass

    def get_environment_state(self) -> np.array:
        state = self.env.get_state()
        return np.array([state])

    def environment_step(self, action: int) -> (np.array, float, bool):

        action = (action * 4.) + 4.

        return self.env.step(action)

    def get_possible_states(self) -> np.array:
        return self.get_environment_state()

    def policy_values_plot(self, save_fig: Path = None, show_plot: bool = False):
        raise NotImplementedError

    def render_environment(self):
        pass

    @staticmethod
    def win_condition(episode: Episode):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
