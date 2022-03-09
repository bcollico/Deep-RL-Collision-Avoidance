# initialization training params
BATCH_SIZE = 100
LR = 0.01
MOMENTUM=0.9
STEP_SIZE = 150

N_EPISODES = 800
C          = 1
M          = 10
GAMMA      = 0.8
VMAX       = 1.0 #??
DT         = 0.5 # for CADRL
EPS_GREEDY_MAX = 0.5 # probability with which a random action is chosen
EPS_GREEDY_MIN = 0.1
# epsilon greedy should decay from 0.5 to 0.1 linearly
NUM_RL_EPOCHS = 30
RL_BATCH_FRAC = 0.2

KINEMATIC=True
GOAL_EPS = 0.5
MAX_TIME = 25
RADIUS = 1

import os
try:
    USER = os.getlogin() #
except:
    USER = 'bcollico'
print(USER)

if USER == 'torstein':
    FOLDER = "/home/torstein/Stanford/aa277/aa277_project/data"
elif USER == 'bdobkowski':
    FOLDER  = "/home/bdobkowski/Stanford/AA277/aa277_project/data"
elif USER == 'bcollico':
    FOLDER  = "/home/bcollico/github/aa277_project/data"
else:
    raise Exception('Need to list a folder in on your local machine to store data')
