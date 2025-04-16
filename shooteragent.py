# modify discretize state such that it cuts game env in x-val bins with no y-val considerations
# find width of game env to use for this
# consider smaller bins at start and making them bigger as the game progresses
# line 205 in game engine to see self.world_width

import numpy as np
import pygame

pygame.init()
pygame.mixer.init()

from engine import GameEngine
from collections import defaultdict
import gymnasium as gym
import pickle
import os


class ShooterAgent:

    @staticmethod
    def bin_x_state(x_state: float) -> float:
        """Discretizes the state x space into a grid.
        Because the observation space is continuous, we need to create 'bins' to group the
        coordinates together to give the Q-learning algorithm better data.
        Returns the observation as a tuple because the Q-table needs hashable keys (np.ndarrays are not).
        Args:
            obs (np.ndarray): The observation to discretize.
            grid_size (int): The size of each grid cell
        Returns:
            x_bin (float): The current x bin the player is in
        """
        # on initialization, the world_width is 0, so ignore that state
        if GameEngine.world_width > 0:
            level_progress = x_state / GameEngine.world_width  # Calculate progress through the level (0 to 1)
            min_bin_size = GameEngine.world_width / 200  # Smaller bin size at the beginning
            max_bin_size = GameEngine.world_width / 100  # Larger bin size at the end
            bin_size = min_bin_size + (max_bin_size - min_bin_size) * level_progress

            # calculate the current bin the user is in
            # if bin_size is 50, then the bins are [0, 50), [50, 100), etc.
            x_bin = np.floor((x_state / bin_size))

        else:
            x_bin = 0

        return x_bin

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
        """Uses epsilon-greedy policy to select an action.
        Returns best action with probability (1 - epsilon).
        Otherwise returns random action with probability epsilon.
        Args:
            obs (tuple[int, int, bool]): The current observation.
        Returns:
            int: The action to take.
        """
        obs = (self.bin_x_state(obs[0]), *obs[1:])
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_table[obs]))

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ) -> None:
        """Updates the Q-Value of the action
        Args:
            obs (np.ndarray): The current observation.
            action (int): The action taken.
            reward (float): The reward received.
            terminated (bool): Whether the episode has terminated.
            next_obs (tuple[int, int, bool]): The next observation after taking the action.
        """
        obs = (self.bin_x_state(obs[0]), *obs[1:])
        next_obs = (self.bin_x_state(next_obs[0]), *next_obs[1:])

        future_q = (not terminated) * np.max(self.q_table[next_obs])

        temporal_difference_error = (
            reward + self.gamma * future_q - self.q_table[obs][action]
        )

        self.q_table[obs][action] = (
            self.q_table[obs][action] + self.alpha * temporal_difference_error
        )

        self.training_error.append(temporal_difference_error)

    def decay_epsilon(self) -> None:
        """Decay the exploration rate (epsilon) after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save_snapshot(self, level: str, filename: str) -> None:
        """Save a snapshot of the current Q-values which can be used to resume training or evaluate the agent later.
        Args:
            level (str): The game level folder name to save the snapshot to.
            filename (str): The filename to save the snapshot to.
        """
        snapshots_dir = os.path.join("snapshots", f"level{level}")
        if not os.path.exists(snapshots_dir):
            os.makedirs(snapshots_dir)
        
        with open(os.path.join(snapshots_dir, f"{filename}.pkl"), "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load_snapshot(self, level: str, filename: str) -> None:
        """Load the agent a snapshot of the Q-values from a file.
        Args:
            level (str): The game level folder name to load the snapshot from.
            filename (str): The filename to load the snapshot from.
        """
        with open(os.path.join(f"{level}", f"{filename}.pkl"), "rb") as f:
            self.q_table = defaultdict(
                lambda: np.zeros(self.env.action_space.n), pickle.load(f)
            )
