#defind the gridnet enviroment

import numpy as np

class GridNet:
  
  def __init__(self):
    #no of features for state, which is x, y, prev_BS
    self.n_features = 3 
    #size of network 5x5
    self.n_size = 5
    #no. of directions  2: up/right only
    self.n_dir = 2
    #no. of basestations
    self.n_BS = 5
    #no of actions. 10 = 2 directions * 5 BSs
    self.n_actions = self.n_dir * self.n_BS
    #switching cost
    self.switching_cost = 0.8
    
    #read in data rates at each x,y
    self.complete_bitrates = self.bitrates()

  #read in the bit rates
  def bitrates(self):
    complete_bitrates = np.zeros([self.n_size, self.n_size, self.n_BS])
    #open file contain data rate. each line is the data rate. first x, then y, then BS 
    finput = open('25steps_rand.txt', 'r')
    for bs in range(self.n_BS):
      for x in range(self.n_size):
        for y in range(self.n_size):
          complete_bitrates[x, y, bs] = float(finput.readline())
    finput.close()
    return(complete_bitrates)

  #for a given obs (or state), action, return obs_ (next_state) and reward
  def step(self, obs, action):
    [x, y, prev_BS] = obs
    cur_BS = action % self.n_BS
    dirs = action // self.n_BS

    reward = self.complete_bitrates[x, y, cur_BS]
    #prev_BS = -1 means no prev BS. So no sw cost
    if(prev_BS != -1 and cur_BS != prev_BS): 
      reward = reward - self.switching_cost

    if(x==self.n_size-1 and y==self.n_size-1): #it reach the goal
      obs_ = None
    else:
      if(x==self.n_size-1): #is the right bound
        x1 = x
        y1 = y+1
      elif(y==self.n_size-1): #it is the top bound 
        x1 = x+1
        y1 = y
      elif(dirs==0): #goes up
        x1 = x
        y1 = y+1
      else: #goes right
        x1 = x+1
        y1 = y
      obs_ = [x1, y1, cur_BS]
    return obs_, reward

  #solv Q matrix using DP.
  #optional, only works for certain special networks.
  def PopulateQ(self, gamma):
    #num of BS is n_BS+1 since we have a virtual BS -1 means no prev_BS
    Q_matrix = np.zeros([self.n_size, self.n_size, self.n_BS+1, self.n_actions])
    #For 5x5 network, total_steps is 8. It is also the max distance to goal.
    total_steps = 2*(self.n_size-1)
    for i in range(total_steps+1):
      #state_x for give distance i.
      x0=self.n_size-1-i
      if(x0<0): x0=0
      for x in range(x0, self.n_size):
        y = total_steps-i-x
        if(y<0): break
        for prev_BS in range(-1, self.n_BS):
          z = prev_BS + 1  #shift the index of BS since array index start from 0
          obs = [x, y, prev_BS]
          for action in range(self.n_actions):
            obs_, reward = self.step(obs, action)
            if ( obs_ == None ): #goal, no next state
              next_value = 0
            else:
              [nx, ny, nz] = obs_
              #find the value of state obs_, again, shift index of BS by 1 
              next_value = Q_matrix[nx, ny, nz+1, :].max()
            Q_matrix[x, y, z, action] = reward + gamma*next_value
    return Q_matrix

#---------main----------------
#gridnet = GridNet()
#print(gridnet.complete_bitrates)

"""
obs = [0, 0, 1]
while True:
  action = np.random.randint(gridnet.n_actions)
  obs_, reward = gridnet.step(obs, action)
  print(obs, obs_, reward)
  if obs_ == None:
    break
  obs = obs_
"""

#to print the R_matrix
"""
for state in range(gridnet.n_size*gridnet.n_size*gridnet.n_BS):
  prev_BS = state % gridnet.n_BS
  location = state // gridnet.n_BS
  y = location % gridnet.n_size
  x = location // gridnet.n_size
  obs = [x, y, prev_BS]
  state_r = []
  for action in range(gridnet.n_actions):
    obs_, reward = gridnet.step(obs, action)
    state_r.append(reward)
  print(obs, state_r)
"""
