# RUN THIS to train using DQN
import numpy as np
import os.path

import torch
from gridnet import GridNet
from DQN import DQN

#------------------main----------------------------------------
PATH='./nn.pth'
Training_steps = 50000
#Training_steps = 5

gnet = GridNet()
dqn = DQN(gnet.n_features, gnet.n_actions)
#Load trained parameters if exists.
if os.path.isfile(PATH):
  dqn.policy_net.load_state_dict(torch.load(PATH))
  dqn.target_net.load_state_dict(torch.load(PATH))

print('Training ...')
total_score = 0
for i in range (Training_steps):
    #random x, y, prev_BS and action
    x = np.random.randint(gnet.n_size)
    y = np.random.randint(gnet.n_size)
    #prev_BS = -1 means no prev BS. Use for source.
    prev_BS = np.random.randint(-1, gnet.n_BS)
    obs = [x, y, prev_BS] 
    action = np.random.randint(gnet.n_actions)
    obs_, reward = gnet.step(obs, action)
    if obs_ == None:
      dqn.memory.push(torch.tensor([obs]), torch.tensor([[action]]), None, torch.tensor([reward]))
    else:
      dqn.memory.push(torch.tensor([obs]), torch.tensor([[action]]), torch.tensor([obs_]), torch.tensor([reward]))
    score = dqn.train()
    total_score = total_score + score
    if (i % 1000 == 0): #caculate loss every 1000 steps
      loss = total_score/1000.0
      loss = np.sqrt(loss)
      print(loss)
      total_score = 0

#save trained parameters
torch.save(dqn.policy_net.state_dict(), PATH)

