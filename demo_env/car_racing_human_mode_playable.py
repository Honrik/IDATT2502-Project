from car_racing_human_mode_env import play
from leaderboard_util import saveLeaderboardEntry

# Human mode of car-racing with leaderboard
if __name__ == "__main__":

    agent_name = "Olav" #  <-- set agent name here

    toggle_input_name = True   
    if toggle_input_name:
        agent_name = input("Write the agent name:  ").strip()
    
    # Esc to exit, enter to restart
    # Car is controlled with the arrow keys
    score = play()
    
    saveLeaderboardEntry(agent_name=agent_name, score=score)