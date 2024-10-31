import pickle
import numpy as np
from os.path import isfile

pkl_file_path = "demo_env/leaderboard.pkl"
txt_file_path = "demo_env/leaderboard.txt"

def saveLeaderboardEntry(agent_name: str, score: float):
    
    if isfile(pkl_file_path):
        with open(pkl_file_path, "rb") as f:
            agent_names, scores = pickle.load(f)
    else: # Init leaderboard structure if there is no file
        agent_names = np.array([], dtype='U20')
        scores = np.array([], dtype=float)
        

    
    if agent_name in agent_names:
        idx = np.where(agent_names == agent_name)[0][0]
        if scores[idx] < score:
            scores[idx] = score  # Update agent score if new score is higher
    else:
        # Add new agent entry
        agent_names = np.append(agent_names, agent_name)
        scores = np.append(scores, score)
    
    # Save updated leaderboard as a pickle file
    with open(pkl_file_path, "wb") as f:
        pickle.dump((agent_names, scores), f)
    
    # Write the leaderboard to a text file for viewing
    writeLeaderboardToTextFile(agent_names, scores)
        
        
def writeLeaderboardToTextFile(agent_names, scores):
    # Writes leaderboard data to text file
    if not isfile(txt_file_path):
        print(f"Text file did not exist: {txt_file_path}")
        with open(txt_file_path, 'a'): # Create the file
            pass
    
    # Text-formatting data
    agent_column_width = 20
    score_column_width = 8
    
    head = f"{'Agent':<{agent_column_width}}|{'Score':>{score_column_width}}\n"
    lines = [head, '*' * (agent_column_width + score_column_width + 3) + '\n']
    
    # Sort the data on score
    sorted_indices = np.argsort(scores)[::-1]
    for idx in sorted_indices:
        agent = agent_names[idx]
        agent_score = scores[idx]
        lines.append(f"{agent:<{agent_column_width}}| {agent_score:>{score_column_width}.2f}\n")
    
    # Write to file
    with open(txt_file_path, "w") as f:
        f.writelines(lines)

def removeLeaderboardEntry(agentname: str):
    # Deletes agent leaderboard entry if name exists in leaderboard
    try:
        with open(pkl_file_path, "rb") as f:
            agent_names, scores = pickle.load(f)
    except Exception as e:
        print(f"Error loading leaderboard: {e}")
        return
    
    if agentname in agent_names:
        idx = np.where(agent_names == agentname)[0][0]
        agent_names = np.delete(agent_names, idx)
        scores = np.delete(scores, idx)
        print(f"Removed {agentname} from leaderboard.")
    else:
        print(f"Agent {agentname} not found in leaderboard.")
        return
    
    # Write the new leaderboard state to pkl and txt files
    with open(pkl_file_path, "wb") as f:
        pickle.dump((agent_names, scores), f)
    
    writeLeaderboardToTextFile(agent_names, scores)

