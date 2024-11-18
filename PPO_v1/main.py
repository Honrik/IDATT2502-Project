import os
import pickle
from pathlib import Path

from flappy_bird_train import train
from flappy_bird_eval import eval_current_model, demo_current_model
from utils.ppo_utils import get_next_model_version, save_eval_data, get_plot_paths

from ppo_hyperparameters import PPOHyperparameters


if __name__ == '__main__':
    # Training run setup
    start_new_model = False # Training function needs to know if models should be loaded or created
    
    # Per training-run toggles
    train_model = True # Run training
    eval_after_train = True   # Run and save eval (after each training run)
    demo_after_train = True    # Run demo          (after each training run)
    
    n_eval_episodes = 100 # Number of eval episodes to run and save if model improves in training run
    n_demo_episodes = 1 # Number of demo episodes to run after each training run
    
    
    # Training run duration settings
    n_training_runs = 10
    max_episodes = 1000
    timesteps_per_run = 200000
    
    # Set name of current model (create new model if changing training methods)
    model_name = "e"

    # Get plot figure and data paths for model
    plt_fig_path, plt_data_path = get_plot_paths(model_name)
    

    # PPO hyperparameters
    ppo_hyperparams = PPOHyperparameters(        
        # Tuneable params
        alpha = 0.000004,      # Reduce for stability, increase for faster learning
        policy_clip = 0.15,    # Reduce for stability, increase for faster learning
        entropy_coeff = 0.001, # Try increase if slow model progress, reduce if instability 
        horizon = 1024,        # Reduce for more frequent policy updates, increase if instability
        n_epochs = 2,          # Increase for faster learning, reduce if instability or frequent performance degradation
        
        # Fixed params
        gamma = 0.99,        # Keep at 0.99
        gae_lambda = 0.96,   # Keep at 0.96
        minibatch_size = 32  # Keep at 32. 64 can also work if horizon >= 4096
    )
    
    # Implementation-specific parameters
    max_episode_len = 2048 # Truncate episodes to prevent overfitting on long episodes
    performance_degredation_cap = 0.05 # Abort training early if performance drops beyond this percentage

    
    # Run training runs
    for i in range(n_training_runs):
        # Get next version number of current model
        next_model_version = get_next_model_version(model_name)
        model_name_version = model_name + str(next_model_version)
        
        print(f"\n{model_name_version}")

        # Run a training iteration        
        model_improved = train(
            # Training run settings
              max_training_steps=timesteps_per_run,
              n_training_episodes=max_episodes,
              backup_dir_name=model_name_version,
              load_current_model= not start_new_model,
            # Plot file paths
              figure_file = plt_fig_path,
              plt_file_path=plt_data_path,
            # PPO parameters
              ppo_params = ppo_hyperparams,
            # Implementation parameters
              max_episode_len=max_episode_len,
              performance_degredation_cap = performance_degredation_cap)
        
        
        if not model_improved:
            # Reduce learning rate if previous run did not improve model
            ppo_hyperparams.alpha = 0.75 * ppo_hyperparams.alpha
            
            print("\nNo model improvement")
            print(f"New learning rate: {ppo_hyperparams.alpha} \n") 
        elif eval_after_train:
            # Run evaluation
            eval_data = eval_current_model(n_eval_episodes)
            
            # Save evalution data to file
            if eval_data:
                save_eval_data(model_name, model_name_version, eval_data)
            else:
                print("\n Could not run eval \n")
                                
        # Render demo episodes of the current model
        if demo_after_train:    
            demo_current_model(n_eps=n_demo_episodes)
        
        # When starting a new model only the first training run needs to perform setup
        start_new_model = False