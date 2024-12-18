import gymnasium as gym

def run():
    env = gym.make("CarRacing-v3", render_mode="human")
    observation, info = env.reset()

    episode_over = False
    while not episode_over:
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()
if __name__ == "__main__":
    run()