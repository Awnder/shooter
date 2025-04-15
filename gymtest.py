import pygame

pygame.init()
pygame.mixer.init()

import gymnasium as gym
from gymenv import ShooterEnv
from shooteragent import ShooterAgent
from tqdm import tqdm
import argparse
import os

if __name__ == "__main__":
    learning_rate = 0.01
    n_episodes = 1000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    parser = argparse.ArgumentParser(description="Shooter Agent Training or Loading")
    load_train_group = parser.add_mutually_exclusive_group(required=True)
    load_train_group.add_argument(
        "-l",
        "--load", 
        type=str, 
        help="Path to the pickled agent snapshot to load. Use format 'game_level-episode_number'"
    )
    load_train_group.add_argument(
        "-t",
        "--train", 
        type=int, 
        help="Train a new agent from scratch on the specified game level. Use format 'game_level'"
    )
    train_group = parser.add_argument_group("train", "Arguments for training mode")
    train_group.add_argument(
        "-e",
        "--episodes", 
        type=int, 
        default=n_episodes, 
        help="Number of episodes to train the agent"
    )
    train_group.add_argument(
        "-r",
        "--render", 
        action="store_true", 
        help="Render the environment for humans"
    )

    load = vars(parser.parse_args())["load"]
    train = vars(parser.parse_args())["train"]
    render = vars(parser.parse_args())["render"]
    episodes = vars(parser.parse_args())["episodes"]

    if "-" not in load:
        print("Invalid load path. Must be in the format 'game_level-episode_number'.")
        exit()

    # load agent and replay the episode
    if load:
        game_level, episode_number = load.split("-")

        # create environment and agent
        env = ShooterEnv(render_mode='human', level=game_level)
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
        agent = ShooterAgent(
            env=env,
            learning_rate=learning_rate,
            initial_epsilon=0.1, # very low epsilon value sine only want exploitation for loaded q-table
            epsilon_decay=epsilon_decay,
            final_epsilon=0.1,
        )

        agent.load_snapshot(os.path.join("snapshots", game_level), episode_number)
        print(f"Playing episode {episode_number} of level {game_level}")

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
    
    # train a new agent
    if train:
        env = None
        if render:
            env = ShooterEnv(render_mode='human', game_level=train)
        else:
            env = ShooterEnv(render_mode=None, game_level=train)
        game_level = env.game.level
        env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
        agent = ShooterAgent(
            env=env,
            learning_rate=learning_rate,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
        )
        ep_count = 0
        n_episodes = episodes if episodes and episodes > 0 else n_episodes
        for episode in tqdm(range(n_episodes)):
            obs, info = env.reset()
            done = False

            while not done:
                action = agent.get_action(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                if render:
                    env.render()
                    
                agent.update(obs, action, reward, terminated, next_obs)

                done = terminated or truncated
                obs = next_obs

            agent.decay_epsilon()
            ep_count += 1

            if ep_count % 100 == 0:
                # Save a snapshot every 10 episodes
                agent.save_snapshot(f'{ep_count}')
                print(f"Saved snapshot at episode {ep_count}")

