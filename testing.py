# Test the training saved in the QL and DQL files and compare them to DynamicProgramming
import numpy as np
import os.path

import torch
from gridnet import GridNet
from DQN import DQN
from QLearn import QLearn

#------------------main----------------------------------------
PATH='./nn.pth'
QPATH='./q.npy'

gnet = GridNet()
Q_solv = gnet.PopulateQ(gamma=0.9)

qlearn = QLearn(gnet.n_size, gnet.n_BS, gnet.n_actions, gamma=0.9) 
#load training Q_matrix
if os.path.isfile(QPATH):
  qlearn.Q_matrix = np.load(QPATH)
else:
  print("Error: no trained data available", QPATH)
  exit()

dqn = DQN(gnet.n_features, gnet.n_actions, gamma=0.9)
#Load trained parameters if exists.
if os.path.isfile(PATH):
  dqn.policy_net.load_state_dict(torch.load(PATH))
else:
   print("Error: no trained data available", PATH)
   exit()

print('Dynamic Programming Testing ...')
total_reward = 0.0
#source is [0, 0] with no prev_BS
obs = [0, 0, -1]
while True:
  [x, y, prev_BS] = obs
  z = prev_BS + 1
  action = np.argmax(Q_solv[x, y, z, :])
  obs_, reward = gnet.step(obs, action)
  #total_reward = 0.9*total_reward + reward
  total_reward = total_reward + reward
  cur_BS = action % gnet.n_BS
  data_rate=gnet.complete_bitrates[x, y, cur_BS]
  print(obs, cur_BS, reward, data_rate)
  if obs_ == None:
    break
  obs = obs_
print("Total reward=", total_reward)

print('Q Learning Testing ...')
total_reward = 0.0
#source is [0, 0] with no prev_BS
obs = [0, 0, -1]
while True:
  [x, y, prev_BS] = obs
  z = prev_BS + 1
  action = np.argmax(qlearn.Q_matrix[x, y, z, :])
  obs_, reward = gnet.step(obs, action)
  #total_reward = 0.9*total_reward + reward
  total_reward = total_reward + reward
  cur_BS = action % gnet.n_BS
  data_rate=gnet.complete_bitrates[x, y, cur_BS]
  print(obs, cur_BS, reward, data_rate)
  if obs_ == None:
    break
  obs = obs_
print("Total reward=", total_reward)

print('DQN Testing ...')
total_reward = 0.0
#source is [0, 0] with no prev_BS
obs = [0, 0, -1]
while True:
  action = dqn.policy_net(torch.tensor([obs]).float()).max(1)[1].item()
  obs_, reward = gnet.step(obs, action)
  #total_reward = 0.9*total_reward + reward
  total_reward = total_reward + reward
  [x, y, prev_BS] = obs
  cur_BS = action % gnet.n_BS
  data_rate=gnet.complete_bitrates[x, y, cur_BS]
  print(obs, cur_BS, reward, data_rate)
  if obs_ == None:
    break
  obs = obs_
print("Total reward=", total_reward)
