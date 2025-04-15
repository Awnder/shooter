import pygame

pygame.init()
pygame.mixer.init()

import gymnasium as gym
from gymenv import ShooterEnv
from shooteragent import ShooterAgent

if __name__ == "__main__":
    learning_rate = 0.01
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (1 / 2)  # reduce the exploration over time
    final_epsilon = 0.1

    env = ShooterEnv(render_mode='human')
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=1)
    agent = ShooterAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    total_reward = 0

    print("X, Y, Exit X, Exit Y, Health, Ammo, Grenades")

    while True:
        obs, info = env.reset()
        done = False

        while not done:
            action = None

            # use quick pygame loop to capture keyboard events
            # these are also capture in shooter.py, but doing it here to 
            # allow for manual control of the agent
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        action = 0  # left
                    if event.key == pygame.K_d:
                        action = 1  # right
                    if event.key == pygame.K_w:
                        action = 2  # jump
                    if event.key == pygame.K_w and event.key == pygame.K_a:
                        action = 3  # jump left
                    if event.key == pygame.K_w and event.key == pygame.K_d:
                        action = 4  # jump right
                    if event.key == pygame.K_SPACE:
                        action = 5  # shoot
                    if event.key == pygame.K_q:
                        action = 6  # grenade

            if action is None:
                continue

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            env.render()

            agent.update(obs, action, reward, terminated, next_obs)

            total_reward += reward
            for o in obs:
                print(o, end=", ")
            print(f"X_bin: {ShooterAgent.bin_x_state(obs[0])} Reward: {reward}, Total Reward: {total_reward}")

            done = terminated or truncated
            obs = next_obs
            
        agent.decay_epsilon()
