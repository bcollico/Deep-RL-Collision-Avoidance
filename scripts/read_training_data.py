import numpy as np
import csv
import os

# USE IF RUNNING SCRIPT AS STANDALONE
# cd = os.getcwd()
# head_dir = cd.split(os.path.sep)[-1]
#
# if head_dir == "aa277_project":
#     filename = os.path.join("data","training_data.csv")
# elif head_dir == "scripts":
#     filename = os.path.join("..","data","training_data.csv")
# else:
#     raise Exception("File path uncertain: Run script from project directory.")

class TrainingData():
    """Class for storing high-level simulation parameters and trajectories."""
    def __init__(self, param_dict:dict):
        # header values extracted from CSV
        self.n_episodes = param_dict["n_episodes"]
        self.n_agents   = param_dict["n_agents"]
        self.dt         = param_dict["dt"]

        # preallocate for data from each episode
        self.traj       = [OrcaTraj(self.n_agents)]*self.n_episodes


class OrcaTraj():
    """Class for organizing trajectory data by agent and state."""

    def __init__(self, n_agents):

        # access data e.g. self.Px[agent][time]
        self.Px = [np.array([])]*n_agents # x position for each agent
        self.Py = [np.array([])]*n_agents # y position for each agent
        self.Vx = [np.array([])]*n_agents # x velocity for each agent
        self.Vy = [np.array([])]*n_agents # y velocity for each agent

        # access data eg self.X[agent][time,state_index]
        self.X = [np.array([])]*n_agents # full state of each robot

def read_training_data(filename:str)->TrainingData:
    """Parsing script for training data CSV written from ORCA C++ code. For
    n robots and N simulations/episodes, CSV file has format:
    
        <# robots> robots
        <# episodes> episodes
        <time_step> Delta-t
        <episode #>, (<px_1(t1)> <py_1(t1)>) (<vx_1(t1)> <vy_1(t1)>), ... ,
                     (<px_n(t1)> <py_n(t1)>) (<vx_n(t1)> <vy_n(t1)>), ... ,
                     (<px_1(tf)> <py_1(tf)>) (<vx_1(tf)> <vy_1(tf)>), ... ,
                     (<px_n(tf)> <py_n(tf)>) (<vx_n(tf)> <vy_n(tf)>) 
                     
        where an episode is entirely contained in one row of the CSV file."""

    param_dict = dict()

    with open(filename) as datafile:
        datareader = csv.reader(datafile, delimiter=',')

        n_episodes = datareader.__next__()[0][0:-7] # remove the chars " robots"
        n_agents   = datareader.__next__()[0][0:-9] # remove the chars " episodes"
        dt         = datareader.__next__()[0][0:-8] # remove the chars " Delta-t"

        param_dict["n_episodes"] = int(n_episodes)
        param_dict["n_agents"]   = int(n_agents)
        param_dict["dt"]         = float(dt)

        # create dict to map episode to trajectories
        data = TrainingData(param_dict)

        # trajectory data is accessed as:
        # data.traj[episode].Px[agent][time_index])

        for ep in range(data.n_episodes):
            row = datareader.__next__()
            for t in range(1,len(row)-1,data.n_agents):
                for ag in range(data.n_agents):
                    # strip new line character, remove parens, split at space
                    r = row[t+ag].rstrip("\n").replace("(","").replace(")","").split(" ")

                    # assign floats to traj struct
                    data.traj[ep].Px[ag] = np.append(data.traj[ep].Px[ag], float(r[1]))
                    data.traj[ep].Py[ag] = np.append(data.traj[ep].Py[ag], float(r[2]))
                    data.traj[ep].Vx[ag] = np.append(data.traj[ep].Vx[ag], float(r[3]))
                    data.traj[ep].Vy[ag] = np.append(data.traj[ep].Vy[ag], float(r[4]))
                    try:
                        data.traj[ep].X[ag]  = np.vstack((data.traj[ep].X[ag], \
                            np.array([float(r[1]), float(r[2]), float(r[3]), float(r[4])])))
                    except:
                        # can't vstack an empty numpy array
                        data.traj[ep].X[ag]  = np.hstack((data.traj[ep].X[ag], \
                            np.array([float(r[1]), float(r[2]), float(r[3]), float(r[4])])))

    return data
