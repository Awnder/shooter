import pygame

pygame.init()
pygame.mixer.init()

import gymnasium as gym
from gymenv import ShooterEnv
from shooteragent import ShooterAgent
from tqdm import tqdm
import argparse

def parse_args():
    """
    Parse command line arguments for the script.
    """
    


if __name__ == "__main__":
    learning_rate = 0.01
    n_episodes = 100
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    parser = argparse.ArgumentParser(description="Shooter Agent Training or Loading")
    parser.add_argument(
        "--load", 
        type=str, 
        help="Path to the pickled agent snapshot to load"
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Train a new agent from scratch"
    )
    args = parser.parse_args()

    if args.load and args.train:
        print("You cannot specify both --load and --train at the same time.")
        exit()
    if not args.load and not args.train:
        print("You must specify either --load or --train.")
        exit()

    env = ShooterEnv(render_mode=None)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)
    agent = ShooterAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # load agent and replay the episode
    if args.load:
        env = ShooterEnv(render_mode='human')
        agent.load_snapshot(args.load)
        print(f"Loaded agent from {args.load}")
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(tuple(obs))
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            agent.update(obs, action, reward, terminated, next_obs)

            done = terminated or truncated
            obs = next_obs
    
    # train a new agent
    if args.train:
        ep_count = 0
        for episode in tqdm(range(n_episodes)):
            obs, info = env.reset()
            done = False

            while not done:
                action = agent.get_action(tuple(obs))
                next_obs, reward, terminated, truncated, info = env.step(action)
                
                agent.update(obs, action, reward, terminated, next_obs)

                done = terminated or truncated
                obs = next_obs

            agent.decay_epsilon()
            ep_count += 1

            if ep_count % 100 == 0:
                # Save a snapshot every 100 episodes
                agent.save_snapshot(f'snapshot_{ep_count}')

