import os
from pathlib import Path
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# This PPO implementation is based on an implementation used for
# solving the cartpole-environment in the old version of OpenAI gym.
# Source: https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/PolicyGradient/PPO/torch/utils.py

# Note:
# My first attempts had some bugs that I wasn't fully able to solve. Learning was unstable and slow,
# even after it was rewritten with identical logic and param-values to this implementation.

# To try to identify a cause I have tested incremental changes in the implementation for this model.
# In doing this, the only noticable change came from using nn.Sequential. This seemingly leads to noticable
# changes in learning performance, even though it logically does the same as manually passing the outputs between layers.
# PPO is inherently unpredictable between runs, so it is however difficult to make definite conclusions.

class PPOMemory:
    """Data structure for storing transitions for training PPO.
    Usage:
        remember(args): saves a transition, probability distribution from the current policy (for the state), and the critic value of the state.
        The data structure internally then holds a trajectory of memories.
        
        generate_batches(): returns the state and a list of minibatches containing shuffled indexes for stored transitions.
        Updating the critic-nn and policy-nn is done per minibatch.
        
        clear_memory(): clears the datastructure by resetting everything to empty lists.
    
    """
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.critic_vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        
        # shuffled indicies (for accessing transitions randomly, instead of shuffling the actual data)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        
        # divide into minibatches of randomly sampled transitions
        # batches is a list of minibatches (lists of transition indices)
        batch_start = np.arange(0, n_states, self.batch_size)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        # Convert lists to NumPy arrays (way more efficient for pytorch)
        states = np.array(self.states)
        actions = np.array(self.actions)
        probs = np.array(self.probs)
        critic_vals = np.array(self.critic_vals)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        return states, actions, probs, critic_vals, rewards, dones, batches

    def store_memory(self, state, action, probs, critic_val, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.critic_vals.append(critic_val)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.critic_vals = []

class ActorNetwork(nn.Module):
    """Actor network for training policy.
    
    Forward method defines the current policy:
    (state) -> probability distribution over actions.
    Because this implementation assumes discrete action space, the output is simply an array with
    a probability of taking each action (policy). As this gets trained, the best actions should
    get higher probabilites.
    """
    def __init__(self, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='PPO_v1\checkpoints\ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.actor = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self, file_path=None):
        test = 0
        if not file_path:            
            T.save(self.state_dict(), self.checkpoint_file)
        else:
            T.save(self.state_dict(), file_path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    """Critic network to evaluate state.
    The returns from this are used to calculate the advantage (if an action is better than the average availabe action).
    This advantage is then used to update the policy network.
    """
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='PPO_v1\checkpoints\ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
                nn.Linear(*input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)

        return value

    def save_checkpoint(self, file_path=None):
        if not file_path:            
            T.save(self.state_dict(), self.checkpoint_file)
        else:
            T.save(self.state_dict(), file_path)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class Agent:
    def __init__(self, n_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.96,
            policy_clip=0.2, batch_size=64, n_epochs=10, entropy_coeff=0, chkpt_dir='PPO_v1\checkpoints\ppo'):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff # testing this

        # make directories and files for models if not exists
        self.actor = ActorNetwork(n_actions, input_dims, alpha, chkpt_dir=chkpt_dir)
        self.critic = CriticNetwork(input_dims, alpha, chkpt_dir=chkpt_dir)
        self.memory = PPOMemory(batch_size)
    
    def remember(self, state, action, probs, val, reward, done):
        self.memory.store_memory(state, action, probs, val, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
    
    def save_models_backup(self, dir_name: str = None, best_avg_score: float = 0):
        if not dir_name:
            print("could not save. No directory name was passed")
            return

            
        print('... saving backup of models ...')
        #backup_dir_path = "PPO_v1/checkpoints/ppo/backups"
        backup_dir_path = os.path.join("PPO_v1", "checkpoints", "ppo", "backups")
        backup_dir_path = os.path.join(backup_dir_path, dir_name[0])
        backup_dir_path = os.path.join(backup_dir_path, f"{dir_name}_{best_avg_score:.2f}")
        
        try:
            Path(backup_dir_path).mkdir(parents=True, exist_ok=True)
            file_path_critic = os.path.join(backup_dir_path, "critic_torch_ppo")
            file_path_actor = os.path.join(backup_dir_path, "actor_torch_ppo")
            self.actor.save_checkpoint(file_path=file_path_actor)
            self.critic.save_checkpoint(file_path=file_path_critic)
        except:
            print("Backup could not be saved at:", backup_dir_path)

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    # Train the agent on the stored trajectory.
    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, critic_vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*critic_vals_arr[k+1]*\
                            (1-int(dones_arr[k])) - critic_vals_arr[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            critic_vals_arr = T.tensor(critic_vals_arr).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + critic_vals_arr[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()
                
                # try entropy loss to encourage exploration
                entropy_loss = dist.entropy().mean()
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * entropy_loss
                
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()