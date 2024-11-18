import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import pickle

def plot_learning_curve(x, scores, figure_file, n_eps_for_avg: int = 100):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-n_eps_for_avg):(i+1)])
    plt.plot(x, running_avg)
    plt.title(f"Running average of previous {n_eps_for_avg} scores")
    plt.savefig(figure_file)
    
    
def save_plot_data(best_avg_score, score_history, run_start_ep_arr, run_n_steps_arr, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump((best_avg_score, score_history, run_start_ep_arr, run_n_steps_arr), f)


def load_plot_data(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return data[0], data[1], data[2], data[3]
    except (FileNotFoundError, EOFError):
        return -100.0, [], [], []
    
    
def get_next_model_version(model_name):
    backup_files = None
    try:
        model_backups_path = os.path.join("PPO_v1", "checkpoints", "ppo", "backups", model_name)
        backup_files = os.listdir(model_backups_path)   
    except:
        backup_files = None
        
    prev_model_runs = 0
    if backup_files:
        prev_model_runs = len(backup_files)
    
    next_model_version = prev_model_runs + 1
    
    return next_model_version


def save_eval_data(model_name, model_name_version, eval_data):
    try:
        max_pipes, mean_score, eval_scores = eval_data
        eval_dir_path = os.path.join("PPO_v1", "checkpoints", "ppo", "backups", "evals", model_name)
        
        # Make model eval directory if not exists
        Path(eval_dir_path).mkdir(parents=True, exist_ok=True)
        eval_file_name = f"{model_name_version}_eval_pipes_{max_pipes}_max_score_{max(eval_scores):.2f}_mean_score_{mean_score:.2f}"
        
        eval_file_path = os.path.join(eval_dir_path, eval_file_name)
        
        with open(eval_file_path, 'wb') as f:
            pickle.dump(eval_scores, f)
    except:
        print("\n Could not save eval \n")
        print(f"eval dir path: {eval_dir_path}")
        print(f"eval file path: {eval_file_path}")
        
def get_plot_paths(model_name,
                   fig_file_name = "flappy_bird.png",
                   data_file_name = "ppo_flappy_bird_plot_data"):
    
    plot_dir_path = os.path.join("PPO_v1", "plots", model_name)
    
    # Make plot directory for model if not exists
    Path(plot_dir_path).mkdir(parents=True, exist_ok=True)
    
    plt_fig_path = os.path.join(plot_dir_path, fig_file_name)
    plt_data_path = os.path.join(plot_dir_path, data_file_name)
    
    return (plt_fig_path, plt_data_path)