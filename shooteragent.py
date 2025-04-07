import numpy as np
import pygame

pygame.init()
pygame.mixer.init()

from collections import defaultdict
import gymnasium as gym
import pickle
import os

class ShooterAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a reinforcement learning agent for the Cliff Walking environment.
        Args:
            env (gym.Env): The Cliff Walking environment.
            learning_rate (float): The learning rate for the agent.
            initial_epsilon (float): The initial exploration rate.
            epsilon_decay (float): The rate at which epsilon decreases.
            final_epsilon (float): The minimum exploration rate.
            discount_factor (float): Discount factor for computing the Q-value
        """
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Returns best action with probability (1 - epsilon).
        Otherwise returns random action with probability epsilon.
        """
        obs = tuple(obs)
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool]
    ) -> None:
        """Updates the Q-Value of the action
        Args:
            obs (tuple[int, int, bool]): The current observation.
            action (int): The action taken.
            reward (float): The reward received.
            terminated (bool): Whether the episode has terminated.
            next_obs (tuple[int, int, bool]): The next observation after taking the action.
        """
        # obs and next_obs are ndarrays, which are unhashable
        # unhashable data types can't be used as dict keys
        obs = tuple(obs)
        next_obs = tuple(next_obs)

        temporal_difference_error = reward + self.gamma * np.max(self.q_table[next_obs]) - self.q_table[obs][action]

        self.q_table[obs][action] = self.q_table[obs][action] + self.alpha * temporal_difference_error

        self.training_error.append(temporal_difference_error)

    def decay_epsilon(self) -> None:
        """Decay the exploration rate (epsilon) after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_snapshot(self, filename) -> None:
        """Save a snapshot of the current Q-values which can be used to resume training or evaluate the agent later.
        Args:
            filename (str): The filename to save the snapshot to.
        """
        if not os.path.exists('snapshots'):
            os.makedirs('snapshots')
        with open(os.path.join('snapshots', f'{filename}.pkl'), 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    def load_snapshot(self, filename) -> None:
        """Load the agent a snapshot of the Q-values from a file.
        Args:
            filename (str): The filename to load the snapshot from.
        """
        with open(f'{filename}.pkl', 'rb') as f:
            self.q_table = defaultdict(
                lambda: np.zeros(self.env.action_space.n),
                pickle.load(f)
            )