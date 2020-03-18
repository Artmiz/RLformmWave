#This program will predict the basestation lists to connect to, given the user's location and bitrates for all locations given 
# the coordinates of the basestations.


import numpy as np

# Read in data rate files
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

#DP variables 
total_steps = len(datarates_list)
number_BS = len(datarates_list[0])
penalty = 2.0 # switching cost constant
#The best solution (associated BSs) upto step m
base_stations = np.zeros((total_steps, total_steps), np.int8)
#The best reward upto step m
total_reward = np.zeros(total_steps)

#function to copy solutions of step n to step m as the first n steps.
def copy_solution(n, m):
    for i in range(n+1):
        base_stations[m][i]=base_stations[n][i]

#Find the solution for step m=0 (first step)
steps=0
for (BS) in range (number_BS):
    if(datarates_list[steps][BS] > total_reward[steps]):
        total_reward[steps]=datarates_list[steps][BS]
        base_stations[steps][steps]=BS

#Find the solution for step m (m=1 ... total_steps-1)
for steps in range(1, total_steps):
    #find BS0, the BS with highest rate at step m
    BS0=0
    for (BS) in range (1, number_BS):
        if(datarates_list[steps][BS] > datarates_list[steps][BS0]):
            BS0=BS
    
    #BS0 is the same as the BS of step m-1. Just add BS0 as solution of step m
    if(base_stations[steps-1, steps-1]==BS0):
        copy_solution(steps-1, steps)
        base_stations[steps][steps]=BS0
        total_reward[steps] = total_reward[steps-1] + datarates_list[steps][BS0]
    #BS0 is not the same as the BS of step m-1. Two cases: there is a switch or there is no switch 
    else:
        #there is a switch. Must be sol of step m-1 then switch to BS0
        copy_solution(steps-1, steps)
        base_stations[steps][steps]=BS0
        total_reward[steps] = total_reward[steps-1] + datarates_list[steps][BS0] - penalty

        #if there is no switch at step m. Any BS with daterate >= BS1 can be candidates.
        BS1 = base_stations[steps-1][steps-1]
        for BS in range(number_BS):
            #If daterate of BS < BS1, no need to consider
            if(datarates_list[steps][BS] < datarates_list[steps][BS1]):
               continue 
            
            #If step m is BS, back track to step n
            BS_total_rate = datarates_list[steps][BS]
            for n in range(steps-1, -1, -1):
                BS_total_rate = BS_total_rate + datarates_list[n][BS]
                #find the total reward if back track to step n
                reward = 0
                if(n!=0): 
                    reward=reward+total_reward[n-1]  #add reward upto step n-1
                    #if there is a switch, minus the penalty
                    if(BS!=base_stations[n-1][n-1]):
                        reward = reward - penalty
                reward = reward+BS_total_rate  #add the data rates from step n to step m

                if(reward > total_reward[steps]):
                    total_reward[steps] = reward
                    if(n!=0): copy_solution(n-1, steps)
                    for i in range(n, steps+1): base_stations[steps][i]=BS

                #if the last basesation of step n-1 is BS, no need to continue back trak. we already have the highest reward 
                if(BS==base_stations[n-1][n-1]):
                    break

file_name = 'DynamicProgrammingResults_%ssteps_penalty%s.txt' %(total_steps, penalty)
Result = open(file_name,'a' )  
#print results
for steps in range(total_steps):
    Result.write('steps =  ' + repr(steps) + '\n')
    #print("steps=", steps, "\nbase stations: ")
    Result.write('Base stations: ')
    for i in range(steps+1): 
        Result.write(repr(base_stations[steps][i]) + ' ')
        #print(base_stations[steps][i], " ")
    Result.write('reward: ' + repr(total_reward[steps]) + '\n')
    #print("reward: ", total_reward[steps])

Result.close()
