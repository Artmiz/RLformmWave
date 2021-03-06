# This program chooses the BS with the highest data rate for each step for the user to connect to

import numpy as np

n_BS = 5 

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


number_BS = len(datarates_list[0])
penalty = 3.0 # switching cost constant
total_reward = 0
total_steps = len(datarates_list)
BS_list = np.zeros(total_steps)
immediate_reward = np.zeros(number_BS)

for step in range (total_steps):
    datarate = datarates_list[step]
    BS = np.argmax(datarate)
    BS_list[step] = BS
    if (step == 0):
        switching_cost = 0
    else:
        if (BS_list[step] == BS_list[step-1]):
            switching_cost = 0
        else:
            switching_cost = penalty
    immediate_reward = datarates_list[step][BS] - switching_cost
    total_reward += immediate_reward
    

file_name = 'MaxDatarateResults_%ssteps_penalty%s.txt' %(total_steps, penalty)
Result = open(file_name,'a' )  # This file has the results choosing highest data rate BS at every user step
Result.write('BS list: ' + repr(BS_list) + '\n')
Result.write('reward: ' + repr(total_reward) + '\n')

Result.close()
        
