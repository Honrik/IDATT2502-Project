# IDATT2502-Project
The project of Henrik Gulbrandsen Nilsen and Olav Asprem where they explore different reinforcement learning algorithms within this external [flappy bird gymnasium environment](https://github.com/markub3327/flappy-bird-gymnasium). We highly recommend going through the action and state space for this repo, aswell as how the rewards are calculated. To view the report written for the class, check it out here: [Exploring Different Reinforcement Learning Algorithms with Flappy Bird](https://github.com/user-attachments/files/17803332/Dokument.maskinlaering.pdf).


## Sources
The DQN and DDQN code is based on this [repo](https://github.com/johnnycode8/dqn_pytorch/tree/main) by Johnny Code. He also has an amazing youtube series explaining the basic theory behind DQN and DDQN. 
The PPO code is based on this [repo](https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch) by Phil Tabor.

Using the existing code as a good baseline, countless of tests and tweaks have been made in an attempt to optimize the result.


## Algorithms
There are 3 main algorithms implemented:
* DQN
* DDQN
* PPO


## Requirements
We recommend running this in a conda environment. <br>Installing the gymnasium environment can potentially overwrite your own package versions for numpy etc.


**Working conda environment setup:**
```console
conda install python=3.11
pip install flappy-bird-gymnasium
pip install torch
pip install PyYAML
```


## Run demos
#### DQN
python DQN/agent.py flappybird_dqn 
#### DDQN
python DQN/agent.py flappybird_ddqn
#### PPO
python PPO_v1/flappy_bird_eval.py



## Results
### DQN
The first results of the dqn model:

![old_dqn](https://github.com/user-attachments/assets/771baa6f-d438-48c6-bc9e-c2b22e1433c7)

After tweaking code and hyperparameters, the result got better:

![dqn_flappybird](https://github.com/user-attachments/assets/33d1564f-1911-4701-9743-98a29d10f7d2)


### DDQN
Same story as in the DQN model, where the inital results were not great:

![old_ddqn](https://github.com/user-attachments/assets/6a4d4c02-9ba5-41bd-8180-eb0c738746b7)

After tweaking the code and hyperparameters, the results also got better for the ddqn model:

![ddqn_flappybird](https://github.com/user-attachments/assets/251b6120-b88c-48ff-ab45-759d833b2271)


### PPO
PPO was trained with an incremental approach using models with different methods. This was necessary due to issues running parallel environments, meaning no mechanism for decorrelation in patterns for longer episodes.

**Model b** was trained without episode truncation, which caused many catastrophic performance drops when encountering episodes longer than the fixed length training trajectory (horizon).<br>
![image](https://github.com/user-attachments/assets/926cb80d-7386-4ca3-a69c-04767b2144b2)

**Model e** resumed training of model b using additional methods such as episode truncation. The full methods are shown in the PPO training and main files.<br>
![image](https://github.com/user-attachments/assets/1a71bc12-8756-4c7e-8229-a7aeeb9fd8a6)

Final results show how incremental training of PPO, resulted in a maximum of **1610 pipes!** This is due to PPO having inherent exploration properties from the stochastic policy approach, as opposed methods like DQN which uses hyperparameter-based exploration, using epsilon.

![image](https://github.com/user-attachments/assets/f4488ff9-ae65-4543-ab65-b089cb062f7f)

### Comparison
To compare the algorithms, we preformed a 10-run test. This is the result:
![results](https://github.com/user-attachments/assets/3f18a1fb-846a-46c6-aa8d-2826a8eaa35e)
