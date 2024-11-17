import numpy as np
import matplotlib.pyplot as plt
import pickle

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
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