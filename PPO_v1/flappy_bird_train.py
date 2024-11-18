import flappy_bird_gymnasium
import os

import gymnasium as gym
import numpy as np

from utils.ppo_utils import plot_learning_curve, load_plot_data, save_plot_data
from discrete_ppo_v0 import Agent
from ppo_hyperparameters import PPOHyperparameters

def train(
        # Run settings
          n_training_episodes,
          max_training_steps,
          load_current_model = False,
        # Directory names and paths for saving models
          backup_dir_name = None,
          figure_file = os.path.join("PPO_v1", "plots", "flappy_bird.png"),
          plt_file_path = os.path.join("PPO_v1", "plots", "ppo_flappy_bird_plot_data"),
        # PPO hyperparameters
          ppo_params: PPOHyperparameters = None,
        # Implementation parameters 
          max_episode_len=2048,
          performance_degredation_cap = 0.10
          ):
    
        
    # Load data from prev training runs
    prev_best_score, all_runs_score_history, runs_start_eps, runs_n_episodes = load_plot_data(plt_file_path)
    
    # Number of recent episodes to average for updating models
    rolling_avg_episodes = 200
    
    # Recalculate rolling avg if number of episodes to use has changed
    best_score = np.mean(all_runs_score_history[-rolling_avg_episodes:])
    # Subtract small offset to enable further improvements if too many max-capped scores
    best_score -= 0.01
    
        
    # Backups are saved and evaluation is run in main if model is updated,
    # so need to keep track of and return if model was updated during run.
    models_updated = False
    
    print("Prev best avg score: ", best_score)
        
    with gym.make("FlappyBird-v0", use_lidar=False) as env:        
        agent = Agent(n_actions=env.action_space.n,
                      input_dims=env.observation_space.shape,
                      ppo_params=ppo_params)
        
        if load_current_model:
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
        
        # Training
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
                                
                n_steps += 1
                score += reward
                
                agent.remember(observation, action, prob, val, reward, done)
                
                observation = observation_
                
                if n_steps % ppo_params.horizon == 0: # update policy after N timesteps
                    agent.learn()
                    learn_iters += 1
                
                # Capped episode length to prevent overtraining on a single long episode
                if n_steps % max_episode_len == 0: break
                
                
                if not done:
                    pipes = info["score"]
                    
            score_history.append(score)
            avg_score = np.mean(score_history[-rolling_avg_episodes:])
            
            print('episode', i, 'pipes %.0f' % pipes, 'score %.1f' % score, 'avg score %.1f' % avg_score, 'time_steps', n_steps, 'learning_steps', learn_iters)
            
            # Abort training early if large drop in performance
            if avg_score < best_score * (1 - performance_degredation_cap):
                break
            
            tol = 0.001
            if avg_score > best_score + tol:
                # skip if recent scores were bad compared to best avg score (prevent saving on bad update)
                if(np.mean(score_history[-10:]) < best_score):
                    print("\nbad last few scores, no update")
                    continue
                if len(score_history) < rolling_avg_episodes: # No saving until enough stored data for rolling avg
                    continue
                
                agent.save_models()
                models_updated = True
                
                best_score = avg_score # Update best score
                
                # Update run control data
                current_run_start_ep = 0 if (len(runs_start_eps) == 0) else runs_start_eps[-1] + runs_n_episodes[-1]
                runs_start_eps.append(current_run_start_ep)
                runs_n_episodes.append(n_eps_since_plt_update)
                
                print("... saving plot ...")
                print(f"episodes since last update: {n_eps_since_plt_update}")
                
                # Update stored plot data
                save_plot_data(best_score, score_history, runs_start_eps, runs_n_episodes, plt_file_path)
                
                # Reset update-interval counter
                n_eps_since_plt_update = 0
                
                # Update saved plot figure
                x = [i+1 for i in range(len(score_history))]
                plot_learning_curve(x, score_history, figure_file)
        
        # Save backups if models were updated
        if backup_dir_name and models_updated:
            agent.load_models()
            agent.save_models_backup(backup_dir_name, best_avg_score=best_score)
            
        
        env.close()
        return models_updated