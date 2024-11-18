import flappy_bird_gymnasium

import gymnasium as gym
import numpy as np

from discrete_ppo_v0 import Agent
from ppo_hyperparameters import PPOHyperparameters



def demo_current_model(n_eps = 1):
    try: # Wrap to handle KeyboardInterrupt and always closing env
        
        env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
        ppo_params = PPOHyperparameters(entropy_coeff=0)
        agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape,
                    ppo_params=ppo_params)
        
        agent.load_models()
        
        print("\n... Running demo ...")
        for ep in range(n_eps):
            obs, _ = env.reset()
            timesteps = 0
            score = 0
            pipes = 0
            terminated = False
            while not terminated:
                timesteps += 1

                action = agent.choose_action(obs, training=False)

                obs, reward, terminated, _, info = env.step(action)
                score += reward
                pipes = info["score"]

            print(f"Demo episode {ep + 1}:")
            print(f"  Time steps: {timesteps}")
            print(f"  Pipes: {pipes}")
            print(f"  Score: {score}\n")
    except KeyboardInterrupt: # Close env if keyboard interrupt
        print("\nDemo interrupted!")
    finally:
        env.close()
                
    
def eval_current_model(n_episodes):
    try: # Wrap to handle interrupt and always close env
        
        env = gym.make("FlappyBird-v0", use_lidar=False)
        # Setup eval agent and load current model
        ppo_params = PPOHyperparameters(entropy_coeff=0)
        agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape,
                      ppo_params=ppo_params)
        agent.load_models()
        
        print(f"\nEvaluating current model ({n_episodes} episodes)")
        
        max_pipes = 0
        episode_scores = []
        episode_pipes = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            timesteps = 0
            score = 0
            n_pipes = 0
            done = False
            while not done:
                timesteps += 1

                action = agent.choose_action(obs, training=False)

                obs, reward, terminated, _, info = env.step(action)
                score += reward

                done = terminated
                
                if not done:
                    n_pipes = info["score"]
                
            print(f"  ep {ep+1}, pipes: {n_pipes}, score: {score:.0f}")
            
            episode_scores.append(score)
            episode_pipes.append(n_pipes)
            
            max_pipes = max(n_pipes, max_pipes)
            
            
            

        mean_score = np.mean(episode_scores)
        avg_pipes = np.mean(episode_pipes)
        
        print(f"\nEvaluation results ({n_episodes} eps):")
        print(f"    Max pipes in an episode: {max_pipes}")
        print(f"    Average episode pipes:   {avg_pipes:.0f}")
        print(f"    Average episode score:   {mean_score:.2f}")
        
        return (max_pipes, mean_score, episode_scores)
    
    except KeyboardInterrupt: # Close env if keyboard interrupt
        print("\nEvaluation interrupted!")
    finally:
        env.close()
    
    return None


# For unit testing
if __name__ == "__main__":
    print("... Running demo and eval ...")
    print("To break demo or eval early, interrupt in console: ctrl+c")
    demo_current_model(1)
    eval_eps = 100
    eval_current_model(eval_eps)
