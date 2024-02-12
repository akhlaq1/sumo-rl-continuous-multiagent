import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def mapping_to_target_range( x, target_min=0.05, target_max=1.5 ) :
    x02 = T.tanh(x) + 1 # x in range(0,2)
    scale = ( target_max-target_min )/2.
    return  x02 * scale + target_min


class CriticNetwork(nn.Module):
    '''
    This Network takes in state and action and outputs q value
    
    input_dims = No. of observations
    beta = learning rate of the descent function
    
    This CRITIC NETWORK gets observations of all the agents
    
    '''
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_agents,
                 n_actions, name, chkpt_dir):
        super(CriticNetwork, self).__init__()

        self.chkpt_file = os.path.join(chkpt_dir, name)

        # input_dims = No. of observations

        self.fc1 = nn.Linear(input_dims + n_agents * n_actions, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.q = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        x = F.relu(self.fc1(T.cat([state, action], dim=1)))
        x = F.relu(self.fc2(x))
        q = self.q(x)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))


class ActorNetwork(nn.Module):
    '''
    This function takes in State (observation) of the environment
    and outputs an action using the estimated policy.
    
    We can say this class is a policy estimator.
    
    The activation funciton used is **Softmax**, which outputs a 
    probability distribution for each action.
    
    We can try "Sigmoid" that has range 0 to 1
    '''
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir,action_space_high):
        super(ActorNetwork, self).__init__()
        self.action_space_high = action_space_high

        self.chkpt_file = os.path.join(chkpt_dir, name)

        self.fc1 = nn.Linear(input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.pi = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # pi = T.tanh(self.pi(x))
        # pi = T.softmax(self.pi(x), dim=1)
        out = mapping_to_target_range(self.pi(x))
        return out

    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))