import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from agents.base_pg import BasePolicyGradientAgent, TrainingExperience
from envs import Environment, EpisodesBatch
from models import SimpleModel
from code_utils.config_utils import BaseConfig


class BaseAgentConfig(BaseConfig):

    def __init__(self, config_dict: dict):
        """Agent configurations for Naive and Reward to Go Policy Gradient training.

        Args:
            config_dict: Configurations dictionary
        """
        BaseConfig.__init__(self, config_dict=config_dict, name=config_dict["name"],
                            desc=config_dict["desc"])
        self.training_steps = self.config_dict["training_steps"]
        self.show_every = self.config_dict["show_every"]
        self.learning_rate = self.config_dict["learning_rate"]
        self.experience_size = self.config_dict["experience_size"]
        self.minibatch_size = self.config_dict["minibatch_size"]
        self.hidden_layer_sizes = self.config_dict["hidden_layer_sizes"]
        self.mu_activation = self.config_dict["mu_activation"]
        self.hidden_activation = self.config_dict["hidden_activation"]
        self.save_policy_every = self.config_dict["save_policy_every"]
        self.actions_size = self.config_dict["actions_size"]
        self.sigma_activation = self.config_dict["sigma_activation"]
        self.true_action = self.config_dict["true_action"]
        self.start_mu = float(self.config_dict["start_mu"])
        self.start_sigma = float(self.config_dict["start_sigma"])
        self.normalize_rewards = self.config_dict["normalize_rewards"]


class REINFORCEAgentConfig(BaseAgentConfig):

    def __init__(self, config_dict: dict):
        """Agent configurations for REINFORCE Policy Gradient training.

        Args:
            config_dict: Configurations dictionary
        """
        BaseAgentConfig.__init__(self, config_dict=config_dict)
        self.discount_factor = self.config_dict["discount_factor"]


class NaivePolicyGradientAgent(BasePolicyGradientAgent):
    """Agent that implements naive policy gradient to train the policy.

    Naive -> The gradient of the log probability of the actions is weighted by
    the total reward of the episode.

    Found here as Basic Policy Gradient:
        https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html
    """

    def __init__(self, env: Environment, agent_path: Path, policy: SimpleModel,
                 agent_config: BaseAgentConfig):
        """See base class."""
        BasePolicyGradientAgent.__init__(self,
                                         env=env,
                                         agent_path=agent_path,
                                         policy=policy,
                                         agent_config=agent_config)

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """See base class."""

        states_batch = []
        weights_batch = []
        actions_batch = []
        total_rewards = []
        episode_lengths = []

        for episode in episodes:
            states_batch.append(episode.states)
            actions_batch.append(episode.actions)
            weights_batch += [episode.total_reward] * len(episode)
            total_rewards.append(episode.total_reward)
            episode_lengths.append(len(episode))

        states_batch = np.concatenate(states_batch, axis=0)
        actions_batch = np.concatenate(actions_batch, axis=0)

        return TrainingExperience(states_batch, weights_batch, actions_batch,
                                  total_rewards, episode_lengths)


class RewardToGoPolicyGradientAgent(BasePolicyGradientAgent):

    def __init__(self, env: Environment, agent_path: Path, policy: SimpleModel,
                 agent_config: BaseAgentConfig):

        BasePolicyGradientAgent.__init__(self,
                                         env=env,
                                         agent_path=agent_path,
                                         policy=policy,
                                         agent_config=agent_config)

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """See base class."""

        states_batch = []
        weights_batch = []
        actions_batch = []
        total_rewards = []
        episode_lengths = []

        for episode in episodes:
            states_batch.append(episode.states)
            actions_batch.append(episode.actions)

            episode_rewards_to_go = []
            for i in range(len(episode.rewards)):
                episode_rewards_to_go.append(sum(episode.rewards[i:]))

            weights_batch.append(episode_rewards_to_go)
            total_rewards.append(episode.total_reward)
            episode_lengths.append(len(episode))

        states_batch = np.concatenate(states_batch, axis=0)
        actions_batch = np.concatenate(actions_batch, axis=0)
        weights_batch = np.concatenate(weights_batch, axis=0)

        return TrainingExperience(states_batch, weights_batch, actions_batch,
                                  total_rewards, episode_lengths)


class REINFORCEPolicyGradientAgent(BasePolicyGradientAgent):

    def __init__(self, env: Environment, agent_path: Path, policy: SimpleModel,
                 agent_config: REINFORCEAgentConfig):

        self.discount_factor = agent_config.discount_factor
        BasePolicyGradientAgent.__init__(self,
                                         env=env,
                                         agent_path=agent_path,
                                         policy=policy,
                                         agent_config=agent_config)

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """See base class."""

        states_batch = []
        weights_batch = []
        actions_batch = []
        total_rewards = []
        episode_lengths = []

        for episode in episodes:
            states_batch.append(episode.states)
            actions_batch.append(episode.actions)

            episode_discounted_rewards_to_go = []
            for i in range(len(episode.rewards)):
                rewards_to_go = episode.rewards[i:]
                discounts = (np.ones(len(rewards_to_go)) * self.discount_factor) ** np.arange(len(rewards_to_go))
                discounted_rewards_to_go = discounts * rewards_to_go
                episode_discounted_rewards_to_go.append(sum(discounted_rewards_to_go))

            weights_batch.append(episode_discounted_rewards_to_go)
            total_rewards.append(episode.total_reward)
            episode_lengths.append(len(episode))

        states_batch = np.concatenate(states_batch, axis=0)
        actions_batch = np.concatenate(actions_batch, axis=0)
        weights_batch = np.concatenate(weights_batch, axis=0)

        return TrainingExperience(states_batch, weights_batch, actions_batch,
                                  total_rewards, episode_lengths)
