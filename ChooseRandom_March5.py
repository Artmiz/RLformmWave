# This programming choose a random BS at every given user step to service the user

import numpy as np
import random

BS0 = open('9steps_datarates_BS0.txt')

linecount = 0
for line4 in BS0:
    linecount += 1
datarates_list = np.zeros([linecount,8])
BS0.close()

BS0 = open('9steps_datarates_BS0.txt', 'r')
BS1 = open('9steps_datarates_BS1.txt', 'r')
BS2 = open('9steps_datarates_BS2.txt', 'r')
BS3 = open('9steps_datarates_BS3.txt', 'r')
BS4 = open('9steps_datarates_BS4.txt', 'r')

linecount = 0
for line5 in BS0:
    datarates_list[linecount][0] = line5
    datarates_list[linecount][0] = float(datarates_list[linecount][0])
    linecount += 1

linecount = 0
for line6 in BS1:
    datarates_list[linecount][1] = line6
    datarates_list[linecount][1] = float(datarates_list[linecount][1])
    linecount += 1

linecount = 0
for line7 in BS2:
    datarates_list[linecount][2] = line7
    datarates_list[linecount][2] = float(datarates_list[linecount][2])
    linecount += 1

linecount = 0
for line8 in BS3:
    datarates_list[linecount][3] = line8
    datarates_list[linecount][3] = float(datarates_list[linecount][3])
    linecount += 1

linecount = 0
for line9 in BS4:
    datarates_list[linecount][4] = line9
    datarates_list[linecount][4] = float(datarates_list[linecount][4])
    linecount += 1

BS0.close()
BS1.close()
BS2.close()
BS3.close()
BS4.close()

penalty = 2.0 # cost of switching from one BS to another

total_reward = 0
total_steps = len(datarates_list)
#print('Total number of steps: ', total_steps)
BS_list = np.zeros(total_steps)


for step in range (total_steps):
    BS = random.randint(0, 4) # have 5 BS, so choose a BS between 0 and 4
    BS_list[step] = BS
    if (step == 0):
        total_reward = datarates_list[step][BS]
    else:
        if (BS == BS_list[step-1]):
            switch_cost = 0
        else:
            switch_cost = penalty
        total_reward += datarates_list[step][BS] - switch_cost

file_name = 'RandomBSselectionResults_%ssteps_penalty%s.txt' %(total_steps, penalty)
Result = open(file_name,'a' )  # This file has the results of loss  during training stored
Result.write('BS list: ' + repr(BS_list) + '\n')
Result.write('reward: ' + repr(total_reward) + '\n')

Result.close()