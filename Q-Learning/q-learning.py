import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, isTraining = True, render = False):
    
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="human" if render else None)
    
    if isTraining:
        # If training, initilize new q-table
        q = np.zeros((env.observation_space.n, env.action_space.n)) # create an empty 64(game-squares) x 4(actions) array
    else:
        # If not, load existing q-table
        f = open('frozen_lake8x8.pk1', "rb")
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9   # Alpha / learning rate
    discount_factor_g = 0.9 # Gamme / discount factor

    rewards_per_episode = np.zeros(episodes)

    # Following code will be used to overtime decrease randomnes
    epsilon = 1                     # 1 = 100% random actions
    epsilon_decay_rate = 0.0001     # epsilon decay rate, 10,000 episodes (1 / 0.0001) to reach epslion = 0 
    rng = np.random.default_rng()   # random number generator

    for i in range (episodes):
        state = env.reset()[0]            # Puts agent in starting position
        terminated = False                # If agent sadly dies (falls in hole)
        truncated = False                 # If agent gets lost, actions > 200

        while (not terminated and not truncated):

            # If in training mode and the random number generated is less than epsilon pick random action
            if isTraining and rng.random() < epsilon:
                action = env.action_space.sample()      # Select random action, 0=left, 1=down, 2=right, 3=up

            # Else pick action from q-table. As the epsilon value decays, the q-table will be used more
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action) # Execute new action, and update variables

            if isTraining:      # Only update q-table if in training mode
                q[state, action] = q[state, action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if (epsilon == 0):
            learning_rate_a = 0.0001    # Stabilize q-values after exploring

        if reward == 1:
            rewards_per_episode[i] = 1  # Track each episode that gets a reward

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range (episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)]) # Shows running sum for every 100th episode
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake8x8.png')

    if isTraining:
        f = open("frozen_lake8x8.pk1", "wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(15000, isTraining = True, render = False)