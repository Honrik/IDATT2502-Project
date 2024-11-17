import flappy_bird_gymnasium

import gymnasium as gym
import numpy as np

from utils.ppo_utils import plot_learning_curve, load_plot_data, save_plot_data
from discrete_ppo_v0 import Agent

import pickle
from pathlib import Path
import os





def train(n_training_episodes, max_training_steps, start_new_model = False,
          backup_dir_name = None, figure_file = 'PPO_v1/plots/flappy_bird.png',
          plt_file_path = "PPO_v1/plots/ppo_flappy_bird_plot_data",
          # Hyperparams
          alpha= 0.00002,
          policy_clip = 0.15,
          entropy_coeff = 0.00001):
    
    with gym.make("FlappyBird-v0", use_lidar=False) as env:
        rolling_avg_episodes = 200 # number of recent episodes to average for updating models
        
        # Load data from prev training runs
        prev_best_score, all_runs_score_history, runs_start_eps, runs_n_episodes = load_plot_data(plt_file_path)
        prev_best_score = np.mean(all_runs_score_history[-rolling_avg_episodes:])
        prev_best_score -= 0.01 # small offset to enable further improvements if end of buffer has row of perfect scores
        best_score = prev_best_score
        
        models_updated = False
        print("Prev best score: ", best_score)
        
        # Setup
        #n_training_episodes = 500 # maximum episodes
        #max_training_steps = 1000000 # maximum timesteps
        
        #figure_file = 'PPO_v1/plots/flappy_bird.png'

        # Hyper params
        N = 1024 # horizon. Number of steps to run policy before updating models
        batch_size = 32 # low minibatch size seems to give better generalization
        n_epochs = 2 # number of times to train on the collected data
        max_episode_len = 2048 # cap episode length, to prevent overfitting
        
        #alpha = 0.00002 # learning rate
        #policy_clip = 0.15 # epsilon
        #entropy_coeff = 0.00001 # adds randomness to the policy updates, based on the entropy of current policy
        gamma = 0.99
        gae_lambda = 0.96
        
        agent = Agent(n_actions=env.action_space.n, batch_size=batch_size, 
                        alpha=alpha, n_epochs=n_epochs, gamma=gamma, gae_lambda=gae_lambda, entropy_coeff=entropy_coeff,
                        input_dims=env.observation_space.shape, policy_clip=policy_clip)
        
        if not start_new_model:            
            try:
                agent.load_models()
            except:
                print("\nCould not load models\n")
                return


        score_history = all_runs_score_history

        learn_iters = 0
        avg_score = 0
        n_steps = 0

        n_eps_since_plt_update = 0
        for i in range(n_training_episodes):
            if n_steps >= max_training_steps:
                break
            n_eps_since_plt_update += 1
            
            observation = env.reset()[0]
            done = False
            score = 0
            pipes = 0
            while not done:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, terminated, truncated , info = env.step(action)
                done = terminated or truncated
                
                #reward = reward if not terminated else -2.0 # increase punishment for hitting pipe
                
                n_steps += 1
                score += reward
                
                agent.remember(observation, action, prob, val, reward, done)
                
                observation = observation_
                
                if n_steps % N == 0: # update policy after N timesteps
                    agent.learn()
                    learn_iters += 1
                
                if n_steps % max_episode_len == 0: break # cap so it won't only train in a single episode
                
                
                if not done:
                    pipes = info["score"]
                    
            score_history.append(score)
            avg_score = np.mean(score_history[-rolling_avg_episodes:])
            
            print('episode', i, 'pipes %.0f' % pipes, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
            
            if avg_score < 0.95 * best_score: # break early if large drop in performance
                break
            
            if avg_score > best_score:
                # skip if recent scores were bad compared to best rolling avg (prevent saving on bad update)
                if(np.mean(score_history[-10:]) < best_score):
                    print("\nbad last few scores, no update\n")
                    print(f"last few scores: {score_history[-10:]} \n")
                    continue
                if len(score_history) < rolling_avg_episodes: # skip until enough data for rolling avg
                    continue
                
                agent.save_models()
                models_updated = True
                    
                best_score = avg_score
                
                # Update saved history and plot
                current_run_start_ep = 0 if (len(runs_start_eps) == 0) else runs_start_eps[-1] + runs_n_episodes[-1]
                
                runs_start_eps.append(current_run_start_ep)
                runs_n_episodes.append(n_eps_since_plt_update)
                
                print("... saving plot ...")
                print(f"episodes since last update: {n_eps_since_plt_update}")
                save_plot_data(best_score, score_history, runs_start_eps, runs_n_episodes, plt_file_path)
                
                n_eps_since_plt_update = 0
                
                #print(runs_start_eps)
                #print(runs_n_episodes)
                
                x = [i+1 for i in range(len(score_history))]
                plot_learning_curve(x, score_history, figure_file)
        
        if backup_dir_name and models_updated:
            agent.load_models()
            agent.save_models_backup(backup_dir_name, best_avg_score=best_score)
            
        
        env.close()
        return models_updated


def demo_current_model(n_eps = 1):
    with gym.make("FlappyBird-v0", render_mode="human", use_lidar=False) as env:
        agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape)
        agent.load_models()
        
        for ep in range(n_eps):
            obs, _ = env.reset()
            timesteps = 0
            score = 0
            done = False
            while not done:
                timesteps += 1

                # Next action:
                # (feed the observation to your agent here)
                action = agent.choose_action(obs)[0]

                # Processing:
                obs, reward, terminated, _, info = env.step(action)
                score += reward

                # Checking if the player is still alive
                done = terminated


        print("time steps: ", timesteps)
        print("score: ", score)
    
def eval_current_model(n_episodes):
    with gym.make("FlappyBird-v0", use_lidar=False) as env:
        agent = Agent(n_actions=env.action_space.n, input_dims=env.observation_space.shape, entropy_coeff=0)
        agent.load_models()
        
        max_pipes = 0
        episode_scores = []
        for ep in range(n_episodes):
            obs, _ = env.reset()
            timesteps = 0
            score = 0
            env_score = 0
            done = False
            while not done:
                timesteps += 1

                # Next action:
                # (feed the observation to your agent here)
                action = agent.choose_action(obs, training=False)

                # Processing:
                obs, reward, terminated, _, info = env.step(action)
                score += reward

                # Checking if the player is still alive
                done = terminated
                
                if not done:
                    env_score = info["score"]
                
            
            print("env score: ", env_score)
            print("total reward: ", score)
            episode_scores.append(score)
            
            
            max_pipes = max(env_score, max_pipes)

        mean_score = np.mean(episode_scores)
        print(f"highest number of pipes passed in {n_episodes} eps: {max_pipes}")
        print(f"mean score: {mean_score}")
        return (max_pipes, mean_score, episode_scores)


if __name__ == '__main__':
    n_training_runs = 1
    max_episodes = 1000
    timesteps_per_run = 200000
    start_new_model = False
    
    backup_dir_char = "e"
    backup_files = None
    try:
        backup_files = os.listdir("PPO_v1/checkpoints/ppo/backups/" + backup_dir_char)   
    except:
        backup_files = None
        
    prev_model_runs = 0 if not backup_files else len(backup_files)
        
     
    plot_dir_path = "PPO_v1/plots/" + str(backup_dir_char)
    plots_dir = Path(plot_dir_path).mkdir(parents=True, exist_ok=True)
    
    figure_file = os.path.join(plot_dir_path, "flappy_bird.png")
    plt_data_file_path = os.path.join(plot_dir_path, "ppo_flappy_bird_plot_data")
    

    
    # Hyperparams
    alpha = 0.000004
    policy_clip = 0.15
    entropy_coeff = 0.001
    
    update_count = 0
    for i in range(n_training_runs):
        backup_dir_name = backup_dir_char + str(prev_model_runs + update_count + 1)
        print(backup_dir_name)
        
        modelUpdated = train(max_training_steps=timesteps_per_run,
              n_training_episodes=max_episodes,
              backup_dir_name=backup_dir_name,
              start_new_model=start_new_model,
              figure_file = figure_file,
              plt_file_path=plt_data_file_path,
              alpha=alpha,
              policy_clip=policy_clip,
              entropy_coeff=entropy_coeff)
        start_new_model = False
        
        if not modelUpdated: # IF model average did not improve between runs
            print("\n\nNo model improvement\n")
            alpha = 0.75 * alpha # reduce lr
            #entropy_coeff *= 0.25
            print(f"new lr: {alpha} \n")
        else:
            try: # eval
                update_count += 1
                eval_dir_path = f"PPO_v1/checkpoints/ppo/backups/evals/{backup_dir_char}"
                Path(eval_dir_path).mkdir(parents=True, exist_ok=True)
                max_pipes, mean_score, eval_scores = eval_current_model(100)
                eval_file_path = f"{eval_dir_path}/{backup_dir_name}_eval_pipes_{max_pipes}_max_score_{max(eval_scores):.2f}_mean_score_{mean_score:.2f}"
            
                with open(eval_file_path, 'wb') as f:
                    pickle.dump(eval_scores, f)
            except:
                print("\n Could not eval \n")
        #demo_current_model(n_eps=2)
        
    
    