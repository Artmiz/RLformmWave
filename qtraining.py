# RUN THIS to train using QL
import numpy as np
import os.path

from gridnet import GridNet
from QLearn import QLearn

#------------------main----------------------------------------
QPATH='./q.npy'
Training_steps = 50000
#Training_steps = 5

gnet = GridNet()
qlearn = QLearn(gnet.n_size, gnet.n_BS, gnet.n_actions, gamma=0.9)

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
    score = qlearn.train(obs, action, obs_, reward)
    total_score = total_score + score
    if (i % 1000 == 0): #caculate loss every 1000 steps
      loss = total_score/1000.0
      loss = np.sqrt(loss)
      print(loss)
      total_score = 0

#save trained parameters
np.save(QPATH, qlearn.Q_matrix)
