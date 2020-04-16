#This file has the classes for DQN training
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple

#-------------------class Neural Network-----------------------------
class Net(nn.Module):

  def __init__(self, n_features, n_actions, n_hidden):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, n_hidden)
    self.fc2 = nn.Linear(n_hidden, n_actions)

  def forward(self, x):
    #x = F.relu(self.fc1(x))
    x = torch.sigmoid(self.fc1(x))
    x = self.fc2(x)
    return(x)

#--------------------Replay Memory--------------------------------

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN():
  def __init__(self, n_features, n_actions, n_hidden=15, gamma=0.9, alpha=0.1, target_update=10):
    self.gamma = gamma
    #how often to update target network
    self.target_update = target_update
    self.learning_counter = 0
    #mem size and batch size of replay memory
    self.memory_size = 5000
    #self.batch_size = 2
    self.batch_size = 128
    self.memory = ReplayMemory(self.memory_size)

    self.policy_net = Net(n_features, n_actions, n_hidden)
    self.policy_net = self.policy_net.float()
    self.target_net = Net(n_features, n_actions, n_hidden)
    self.target_net = self.target_net.float()
    self.target_net.load_state_dict(self.policy_net.state_dict())
    self.target_net.eval()

    self.optimizer = optim.SGD(self.policy_net.parameters(), lr=alpha)
    self.criterion = nn.MSELoss()

  def train(self):
    if len(self.memory) < self.batch_size:
        return(0)
    transitions = self.memory.sample(self.batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(self.batch_size)
    next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0].detach()
    #next_state_values[non_final_mask] = self.policy_net(non_final_next_states.float()).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Huber loss
    #loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    #for param in policy_net.parameters():
    #    param.grad.data.clamp_(-1, 1)
    self.optimizer.step() 

    #update the target network every self.target_update
    self.learning_counter += 1
    if self.learning_counter % self.target_update == 0:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    return(loss.item())

