import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from envs.environments import Environment, Episode
from envs.reward_functions import LinearReward


class BaseSimpleContinuous(object):
    """Simple one step environment with a fixed best action solution.
    Only one possible state. Different reward functions available.
    """

    def __init__(self, max_reward: float = 1., min_reward: float = 0., min_action: float = -1.,
                 max_action: float = 1., target_action: float = 0.5, reward_function: str = "linear"):
        """Creates a new simple continuous game

        Args:
            max_reward: Maximum reward per step
            min_reward: Minimum reward per step
            min_action: Lowest possible action
            max_action: Biggest possible action
            target_action: Best possible action
            reward_function: The function to use. One of ["linear"].
        """

        self.max_reward = max_reward
        self.min_reward = min_reward
        self.min_action = min_action
        self.max_action = max_action
        self.target_action = target_action
        self.reward_function = reward_function
        self.state = [1.]
        if reward_function == "linear":
            self.reward_function = LinearReward(max_reward=max_reward,
                                                min_reward=min_reward,
                                                min_action=min_action,
                                                max_action=max_action,
                                                target_action=target_action)
        else:
            raise ValueError(f"Reward function {reward_function} was not found.")

    def get_state(self):
        return self.state

    def step(self, action=np.array):
        state = self.get_state()
        done = True
        reward = self.reward_function.calculate_reward(action)
        return state, reward, done


class SimpleContinuous(object):
    """Simple one step environment with a fixed best action solution.
    Used for test purposes.
    """

    def __init__(self, step_target: np.array, max_reward_step: float, steps: int = 1,
                 fixed_states: bool = True):
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
            reward += -((action - self.step_target) ** 2) + 1.0
            self.sum_target += action

        self.define_state()

        state = np.array([self.state[len(self.state) - 1]])

        if self.state[len(self.state) - 1] == self.steps:
            self.game_over = True

        return state, reward, self.game_over


class SimpleContinuousEnvironment(Environment):

    def __init__(self, step_target: np.array, max_reward_step: float, steps: int = 1):
        env = SimpleContinuous(step_target, max_reward_step, steps)
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

    def environment_step(self, action: float) -> (np.array, float, bool):
        """Take an action in the environment.

        Args:
            action: A number between from the range [-1, 1]

        Returns:
            next_environment_state (np.array), reward (float), terminated_environment (bool)
        """
        action = (action * 4.) + 4.  # Action gets converted to range [0, 8]
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


class BaseSimpleContinuousEnvironment(Environment):

    def __init__(self, max_reward: float = 1., min_reward: float = 0., min_action: float = -1.,
                 max_action: float = 1., target_action: float = 0.5, reward_function: str = "linear"):

        env = BaseSimpleContinuous(max_reward=max_reward,
                                   min_reward=min_reward,
                                   min_action=min_action,
                                   max_action=max_action,
                                   target_action=target_action,
                                   reward_function=reward_function)
        action_space = 1
        state_space = 1
        actions = ["action"]
        state_names = ["unique_state"]
        self.range_size = max_action - min_action

        Environment.__init__(self, env, action_space, state_space,
                             actions, state_names, action_space="continuous")

    def reset_environment(self):
        pass

    def get_environment_state(self) -> np.array:
        state = self.env.get_state()
        return np.array([state])

    def environment_step(self, action: float) -> (np.array, float, bool):
        """Take an action in the environment.

        Args:
            action: A number from the range [-1, 1]

        Returns:
            next_environment_state (np.array), reward (float), terminated_environment (bool)
        """

        # Action gets converted to possible values range (simple linear conversion)
        action = (((action - -1.) * self.range_size) / 2.) + self.env.min_action
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
    env = BaseSimpleContinuousEnvironment()
    print(env.get_environment_state())
    print(env.environment_step(np.array)(0))


if __name__ == '__main__':
    main()
