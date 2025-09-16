import flappy_bird_gymnasium
from neural_network import DQN
import gymnasium
import torch 
from experience import experience_replay
import itertools
import yaml
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gymnasium.make("FlappyBird-v0", render_mode="human", use_lidar=True)

class Agent:

    def __init__(self, hyperparameter_set): 
        with open('hyperparameters.yaml', 'r') as file:
            all_hyperparameters_sets =yaml.safe_load(file)
            hyperparameters = all_hyperparameters_sets[hyperparameter_set]

        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size= hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['replay_epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']


    def run(self,is_training = True, render = False):

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        rewardsPerEpisode = []
        epsilon_history = []

        policy_dqn = DQN_NN(num_states, num_actions).to_device(device)

        if is_training:
            memory = ReplayMemory(10000)

            epsilon = self.epsilon_init

        for episode in itertools.count():
            state, _ = env.reset()

            state = torch.tensor(state, dtype=torch.float, device = device)


            terminated = False
            episode_reward = 0.0

            while not terminated:

                if is_training and random.random() < epsilon: 
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.float, device = device)

                else: 
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(dim=0)).argmax()

                # Next action:
                # (feed the observation to your agent here)
                action = env.action_space.sample() # the actions are 0(do nothing) and 1(flap): sample will give back 0 or 1

                # Processing:
                new_state, reward, terminated, _, info = env.step(action)

                episode_reward += reward

                new_state = torch.tensor(new_state, dtype=torch.float, device = device)
                reward = torch.tensor(reward, dtype=torch.float, device = device)






                if is_training: 
                    memory.append((state, action, new_state, reward, terminated))
                
                
                state = new_state

            rewardsPerEpisode.append(episode_reward)

            epsilon = max(epsilon*self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)



if __name__ == '__main__':
    agent = Agent('cartpole1')
    agent.run(is_training=True, render = True)