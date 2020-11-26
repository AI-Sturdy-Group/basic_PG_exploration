import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from environments.environments import Environment, Episode


class SimpleContinuous(object):
    """Simple one step environment with a fixed best action solution.
    Used for test purposes.
    """

    def __init__(self, step_target: np.array, max_reward_step: float, steps: int = 1, fixed_states: bool = True):
        """Creates a new simple continuous game

        Args:
            step_target: value to get to
            max_reward_step: maximum reward per step
            steps: number of steps to finish the game
        """

        self.fixed_states = fixed_states
        self.steps = steps
        self.step_target = step_target
        self.target = step_target * steps
        self.sum_target = 0
        self.max_reward_step = max_reward_step
        self.state = None
        self.game_over = False
        self.target = 0

    def define_state(self):
        if self.fixed_states:
            if self.state is None:
                self.state = np.array([1])
            else:
                np.append(self.state, [len(self.state) + 1])

    @staticmethod
    def get_state():
        return 1

    def step(self, action=np.array):
        reward = 0.

        for action in np.nditer(action):
            reward += -((action - 4.0) ** 2) + 1.0
            self.sum_target += action

        self.define_state()

        state = np.array([self.state[len(self.state) - 1]])

        if self.state[len(self.state) - 1] == self.steps:
            self.game_over = True

        return state, reward, self.game_over


class SimpleContinuousEnvironment(Environment):

    def __init__(self):
        env = SimpleContinuous(step_target=np.array([4.]), max_reward_step=1., steps=1)
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


def main():
    env = SimpleContinuousEnvironment()
    print(env.get_environment_state())
    print(env.environment_step())


if __name__ == '__main__':
    main()
