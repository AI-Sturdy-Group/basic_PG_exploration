import logging
import time
import os
import sys
from pathlib import Path

import wandb
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(os.path.abspath(sys.argv[0]))
sys.path.append(str(SCRIPT_DIR.parent.parent.parent.parent))

from models.models import SimpleModel
from code_utils import BaseConfig
from envs import Episode, Environment, EpisodesBatch


logger = logging.getLogger()


class TrainingExperience(object):
    """A batch of collected experience to do a training step in the network"""

    def __init__(self, states: list, weights: list, actions: list,
                 total_rewards: list, episode_lengths: list):
        """Creates an instance of collected experiences to be feed to the network.

        Args:
            states: The list of states
            weights: The weight used by the current algorithm to approximate
                the network gradients (ie. total reward, reward to go, etc.)
            actions: The list of actions
            total_rewards: The total reward obtained in each episode of the collected experience.
            episode_lengths: The length of each episode of the collected experience.
        """
        assert len(states) == len(weights) == len(actions)
        assert len(total_rewards) == len(episode_lengths)

        self.states = np.array(states, dtype=np.float32)
        self.weights = np.array(weights, dtype=np.float32)
        self.actions = np.array(actions, dtype=np.float32)

        self.total_rewards = np.array(total_rewards, dtype=np.float32)
        self.episode_lengths = np.array(episode_lengths, dtype=np.int32)

    def __str__(self):
        return str(self.__dict__)

    def __len__(self):
        """
        Returns:
            The total number of stored environment steps
        """
        return len(self.states)


class BasePolicyGradientAgent(object):
    """
    Base class for basic policy gradient algorithms.
     - Creates a FFNN model to use as policy.
     - Has training logic
    """

    def __init__(self, env: Environment, policy: SimpleModel, agent_path: Path,
                 agent_config: BaseConfig):
        """Create an agent that uses a FFNN model to represent its policy.

        Args:
            env: The environment the agent is trying to solve
            policy: The model to use as policy
            agent_path: The agent output directory
        """
        self.env = env
        self.agent_path = agent_path
        self.policy = policy
        self.agent_config = agent_config

        self.ckpt = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=self.policy.optimizer,
                                        net=self.policy)
        self.progress_ckpt = tf.train.Checkpoint(step=tf.Variable(-1), optimizer=self.policy.optimizer,
                                                 net=self.policy)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, str(self.policy.model_path) + "\\checkpoints",
                                                       max_to_keep=3)
        self.progress_ckpt_manager = tf.train.CheckpointManager(
            self.progress_ckpt, str(self.policy.model_path) + "\\progress_checkpoints", max_to_keep=None)

    def get_training_experience(self, episodes: EpisodesBatch) -> TrainingExperience:
        """Transforms an EpisodesBatch into a TrainingExperience.
        Each algorithm should implement this with the appropriate conversion.

        Args:
            episodes: The raw collected episodes batch

        Returns:
            A TrainingExperience object with data ready to be feed to the network
        """
        raise NotImplementedError

    def collect_experience(self, size: int) -> TrainingExperience:
        """Collects a batch of steps in the environment using the current policy
        to be feed to the neural network.

        Args:
            size: The max amount of environment steps to collect

        Returns:
            A TrainingExperience object with data ready to be feed to the network
        """

        episodes_batch = EpisodesBatch(max_size=size)

        while not episodes_batch.is_full():
            self.env.reset_environment()
            done = False
            states = []
            rewards = []
            actions = []

            while not done:
                current_state = self.env.get_environment_state()
                tf_current_state = tf.constant(np.array([current_state]), dtype=tf.float32)
                action = self.policy.produce_actions(tf_current_state)[0][0]
                action_list = []
                for a in action:
                    if a > 0:
                        action_list.append(min(1.0, a))
                    elif a < 0:
                        action_list.append(max(-1.0, a))
                next_state, reward, done = self.env.environment_step(np.array(action_list))

                states.append(current_state)
                actions.append(action)
                rewards.append(reward)

            episode = Episode(states, actions, rewards, self.env.action_space)
            episodes_batch.add_episode(episode=episode)

        training_experience = self.get_training_experience(episodes=episodes_batch)

        return training_experience

    def train_policy(self, train_steps: int, experience_size: int,
                     save_policy_every: int=None, show_every: int=None,
                     minibatch_size: int=None) -> float:
        """Train the agent to solve the current environment.

        Args:
            train_steps: The number of training steps
            experience_size: The number of environment steps used on each training step
            show_every: How often to show training info (in training steps)
            save_policy_every: How often to save policy information during training
                               (in training steps)
            minibatch_size: How many environment steps are pass to the NN at once.
                            If None, the total number of steps collected for each
                            training step is used (experience_size)
        Returns
            The final policy test mean reward
        """

        wandbrun = wandb.init(project="BaseSimpleContinuous",
                              group="naive",
                              name=self.agent_config.name,
                              notes=self.agent_config.desc,
                              config=self.agent_config.config_dict,
                              reinit=True,
                              dir="experiments")

        train_steps_avg_rewards = []
        start_time = time.time()
        training_steps = 0
        progress_save = int(train_steps * 0.05)
        best_step = None
        best_checkpoint = None
        best_mean_score = float("-inf")
        for i in range(train_steps):
            training_experience = self.collect_experience(experience_size)
            mean_reward = np.mean(training_experience.total_rewards)

            if show_every is not None:
                if i > 0 and not i % show_every:
                    logger.info(f"Training step N° {i} - "
                                f"Last {len(training_experience.total_rewards)} episodes reward mean: {mean_reward} - "
                                f"Batch time = {time.time() - start_time} sec")
                    start_time = time.time()

            if self.agent_config.normalize_rewards:
                normalized_weights = training_experience.weights - np.mean(training_experience.weights)
                normalized_weights /= np.std(normalized_weights)
                weights_batch = tf.constant(normalized_weights, dtype=np.float32)
            else:
                weights_batch = tf.constant(training_experience.weights, dtype=np.float32)

            states_batch = tf.constant(training_experience.states, dtype=np.float32)
            actions_batch = tf.constant(training_experience.actions, dtype=np.float32)

            batch_size = minibatch_size if minibatch_size is not None else len(states_batch)
            data = tf.data.Dataset.from_tensor_slices((states_batch, actions_batch, weights_batch))
            # TODO: Check if randomizing has some effect
            data = data.shuffle(buffer_size=len(states_batch)).batch(batch_size)

            for minibatch_step, data_batch in enumerate(data):

                policy_outputs, loss, log_probabilities, _ = self.policy.train_step(
                    data_batch[0], data_batch[1], data_batch[2])

                if minibatch_step == len(data) - 1:

                    wandb.log({'training_step': training_steps, 'loss': loss, 'mean_reward': mean_reward},
                              step=training_steps)
                    try:
                        wandb.log({"log_probabilities": wandb.Histogram(log_probabilities)}, step=training_steps)
                    except ValueError:
                        logger.info(f"Failed to save log probabilities: {log_probabilities}")
                    wandb.log({"weights": wandb.Histogram(data_batch[2])}, step=training_steps)

                    for action_idx, action in enumerate(self.env.actions):
                        if len(self.env.actions) == 1:
                            action_mu = policy_outputs[0][0, 0]
                            action_sigma = policy_outputs[1][0, 0]
                            wandb.log({'mu': action_mu, 'sigma': action_sigma}, step=training_steps)
                            wandb.log({'mu_w': self.policy.get_layer("dense_mu").kernel.numpy()[0],
                                       'mu_b': self.policy.get_layer("dense_mu").bias.numpy()[0],
                                       'sigma_w': self.policy.get_layer("dense_sigma").kernel.numpy()[0],
                                       'sigma_b': self.policy.get_layer("dense_sigma").bias.numpy()[0]},
                                      step=training_steps)

                        else:
                            action_mus = policy_outputs[0][:, action_idx]
                            action_sigmas = policy_outputs[1][:, action_idx]
                            wandb.log({"mu": wandb.Histogram(action_mus),
                                       "sigma": wandb.Histogram(action_sigmas)}, step=training_steps)
                            wandb.log({'mu_w': wandb.Histogram(self.policy.get_layer("dense_mu").kernel.numpy()),
                                       'mu_b': wandb.Histogram(self.policy.get_layer("dense_mu").bias.numpy()),
                                       'sigma_w': wandb.Histogram(
                                           self.policy.get_layer("dense_sigma").kernel.numpy()),
                                       'sigma_b': wandb.Histogram(
                                           self.policy.get_layer("dense_sigma").bias.numpy())})

            training_steps += 1
            train_steps_avg_rewards.append(mean_reward)

            self.ckpt.step.assign_add(1)
            self.progress_ckpt.step.assign_add(1)
            if not i % progress_save:
                progress_save_path = self.progress_ckpt_manager.save()
                logger.info(f"Progress checkpoint saved for step {int(self.progress_ckpt.step)}: {progress_save_path}")
            if save_policy_every is not None:
                if not i % save_policy_every:
                    if mean_reward >= best_mean_score:
                        best_mean_score = mean_reward
                        best_step = i
                        save_path = self.ckpt_manager.save()
                        best_checkpoint = save_path
                        logger.info(f"New best model - Test mean reward = {best_mean_score}")
                        logger.info(f"Checkpoint saved for step {int(self.ckpt.step)}: {save_path}")

        moving_avg = np.convolve(train_steps_avg_rewards, np.ones((show_every,)) / show_every, mode='valid')

        # Load best checkpoint and save it
        logger.info(f"Best model in step {best_step} - {best_checkpoint}")
        self.ckpt.restore(best_checkpoint)
        test_reward = self.test_agent(episodes=100)
        logger.info(f"Best model test: {100} episodes mean reward = {test_reward}")
        self.save_agent()
        self.plot_training_info(moving_avg, self.agent_path)

        wandbrun.finish()

        return test_reward

    def save_agent(self):
        """Save the policy neural network to files in the model path."""

        logger.info(f"Saving trained policy to {self.policy.model_path}")
        start = time.time()
        self.policy.save(self.policy.model_path)
        logger.info(f"Saving time {time.time() - start}")

    def load_model(self, model_dir: Path):
        """Loads a trained policy from files. If no save model is found,
        it loads the latest checkpoint available.

        Args:
            model_dir: Where the trained model is stored.
        """
        if Path(model_dir, "saved_model.pb").exists():
            self.policy = tf.keras.models.load_model(model_dir)
        else:
            self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
            logger.info(f"Restored model from checkpoint {self.ckpt_manager.latest_checkpoint}")

    @staticmethod
    def plot_training_info(moving_avg: np.array, agent_folder: Path=None):
        """Plots the training reward moving average.

        Args:
            moving_avg: The moving average data
            agent_folder: Where to save the generated plot
        """
        plt.figure(figsize=(5, 5))

        # Moving average plot
        plt.plot([i for i in range(len(moving_avg))], moving_avg)
        plt.ylabel(f"Reward")
        plt.xlabel("Training step #")
        plt.title("Reward moving average")

        if agent_folder is not None:
            plt.savefig(Path(agent_folder, "reward_moving_average.png"))

    def play_game(self, plot_game: bool=False, delay: float=None) -> Episode:
        """Plays a full episode using the current policy.

        Args:
            plot_game: If the environment should be plotted
            delay: Delay between environment steps (frames)

        Returns:
            The full played episode
        """
        self.env.reset_environment()
        done = False
        states = []
        rewards = []
        actions = []
        while not done:
            if plot_game:
                self.env.render_environment()
                if delay is not None:
                    time.sleep(delay)

            state = self.env.get_environment_state()
            tf_current_state = tf.constant(np.array([state]), dtype=tf.float32)
            action = self.policy.produce_actions(tf_current_state)[0][0]

            new_state, reward, done = self.env.environment_step(action)

            states.append(state)
            rewards.append(reward)
            actions.append(action)

        episode = Episode(states, actions, rewards, self.env.action_space)

        self.env.reset_environment()

        return episode

    def test_agent(self, episodes: int=100):
        total_rewards = []
        for i in range(episodes):
            episode = self.play_game(plot_game=False, delay=None)
            total_rewards.append(episode.total_reward)

        mean_reward = np.mean(total_rewards)
        return mean_reward
