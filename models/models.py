import logging
import math
from pathlib import Path
from typing import List, Tuple

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow.keras.initializers as initializers


logger = logging.getLogger()


class SimpleModel(Model):
    """Feed Forward Neural Network that represents a stochastic policy
    for continuous action spaces. Mu and sigma are calculated using
    the same internal layers.
    """

    def __init__(self, model_path: Path, layer_sizes: List[int], learning_rate: float,
                 actions_size: int, hidden_activation: str = "relu", mu_activation: str = "tanh",
                 sigma_activation: str = "relu", dense_outputs: bool = True):
        """
        Creates a new FFNN model to represent a policy. Implements all needed
        methods from tf.keras.Model.
        Args:
            model_path: Where to save the model and other training info
            layer_sizes: A list with the number of neurons on each hidden layer
            learning_rate: The training step size
            actions_size: The number of possible actions
            hidden_activation: Activation function for hidden layer neurons
            mu_activation: Activation function for mu
            sigma_activation: Activation function for sigma
            dense_outputs: If true, mu and sigma layers are dense, else they are
                only activation functions
        """

        super(SimpleModel, self).__init__()
        self.model_path = model_path
        self.layer_sizes = layer_sizes
        self.output_size = actions_size
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation
        self.mu_activation = mu_activation
        self.sigma_activation = sigma_activation
        self.dense_outputs = dense_outputs

        self.hidden_layers = []
        for i in self.layer_sizes:
            self.hidden_layers.append(Dense(i, activation=self.hidden_activation))

        if self.dense_outputs:
            self.mu = None
            raise NotImplementedError()
        else:
            self.mu = Dense(self.output_size, activation=self.mu_activation,
                            kernel_initializer=initializers.Ones(),
                            bias_initializer=initializers.Zeros(), trainable=False)
            self.sigma = Dense(self.output_size, activation=self.sigma_activation,
                               kernel_initializer=initializers.Ones(),
                               bias_initializer=initializers.Zeros(), trainable=False)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self.train_log_dir = Path(model_path, "train_log")
        self.summary_writer = tf.summary.create_file_writer(str(self.train_log_dir))

    def get_config(self):
        """Used by tf.keras to load a saved model."""
        return {"layer_sizes": self.layer_sizes,
                "learning_rate": self.learning_rate,
                "output_size": self.output_size,
                "hidden_activation": self.hidden_activation,
                "mu_activation": self.mu_activation,
                "sigma_activation": self.sigma_activation}

    @tf.function
    def call(self, inputs: tf.Tensor):
        """See base Class."""

        logger.info("[Retrace] call")
        x = inputs
        for layer in self.hidden_layers:
            x = layer(x)
        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma

    @tf.function
    def train_step(self, states: tf.Tensor, actions: tf.Tensor,
                   weights: tf.Tensor) -> (Tuple[tf.Tensor], tf.Tensor, tf.Tensor):
        """See base Class."""

        logger.info("[Retrace] train_step")
        with tf.GradientTape() as tape:
            mu, sigma = self(states)
            log_probabilities = self._get_log_probabilities(mu, sigma, actions)
            loss = -tf.reduce_mean(weights * log_probabilities)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return (mu, sigma), loss, log_probabilities

    @tf.function
    def _get_log_probabilities(self, mu: tf.Tensor, sigma: tf.Tensor, actions: tf.Tensor) -> tf.Tensor:
        """Gets the logarithmic probabilities of each action for each set of logits.

        Args:
            mu: The mean value for each action for each step
            sigma: The variance value for each action for each step
            actions: The actual actions used in each step

        Returns:
            The logarithmic probabilities for the actions
        """

        logger.info("[Retrace] get_log_probabilities")

        x1 = actions - mu
        x2 = x1 ** 2
        sigma2 = sigma ** 2
        x3 = x2 / sigma2
        logsigma = tf.math.log(sigma)
        x4 = x3 + (2 * logsigma)
        actions_sum = tf.reduce_sum(x4, axis=-1)
        x5 = actions_sum + self.output_size * tf.math.log(2 * math.pi)
        x6 = - x5 * 0.5
        log_probabilities = x6
        return log_probabilities

    @tf.function
    def produce_actions(self, states: tf.Tensor) -> tf.Tensor:
        """Get a sample from the action probability distribution produced
        by the model, for each passed state.

        Args:
            states: The list of states representations

        Returns:
            The sampled action for each state
        """

        logger.info("[Retrace] produce_actions")
        mu, sigma = self(states)
        actions = tfp.distributions.Normal(mu, sigma).sample([1])
        return actions
