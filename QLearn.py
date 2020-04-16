# This file has the Qlearning class definitions and can return Qlearning loss
import numpy as np

class QLearn():
  def __init__(self, n_size, n_BS, n_actions, gamma=0.9, alpha=1.0): 
    self.gamma = gamma
    self.alpha = alpha
    self.Q_matrix = np.zeros([n_size, n_size, n_BS+1, n_actions])

  def train(self, obs, action, obs_, reward):
    [x, y, prev_BS] = obs
    z = prev_BS + 1  #shift index of BS by 1 since it starts from -1.
    if ( obs_ == None ): #goal, no next state
      next_value = 0
    else:
      [nx, ny, nz] = obs_
      #find the value of state obs_, again, shift index of BS by 1
      next_value = self.Q_matrix[nx, ny, nz+1, :].max()
    #not use alpha yet. or alpha=1
    target = reward + self.gamma*next_value
    loss = self.Q_matrix[x, y, z, action] - target
    loss = loss*loss
    self.Q_matrix[x, y, z, action] = target
    return(loss) 

