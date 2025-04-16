import pygame

pygame.init()
pygame.mixer.init()

import gymnasium as gym
from gymenv import ShooterEnv
from shooteragent import ShooterAgent
from tqdm import tqdm
import os

if __name__ == "__main__":
        levels = [1, 2, 3]
        episodes = [1000, 3000, 10000]

        for level, episode in zip(levels, episodes):
            # create environment and agent
            env = ShooterEnv(render_mode='human', level=level)
            env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=episode)
            agent = ShooterAgent(
                env=env,
                learning_rate=0,
                initial_epsilon=0, # only want exploitation for loaded q-table
                epsilon_decay=0,
                final_epsilon=0,
            )

            agent.load_snapshot(os.path.join("snapshots", f"level{level}"), episode)
            print(f"Playing level {level} episode {episode}")

            # run agent in level
            for episode in tqdm(range(1)):
                obs, info = env.reset()
                done = False

                while not done:
                    action = agent.get_action(obs)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    
                    env.render()

                    agent.update(obs, action, reward, terminated, next_obs)

                    done = terminated or truncated
                    obs = next_obs