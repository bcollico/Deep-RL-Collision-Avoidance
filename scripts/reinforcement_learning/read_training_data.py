import numpy as np
import csv
import os

# USE IF RUNNING SCRIPT AS STANDALONE
# cd = os.getcwd()
# head_dir = cd.split(os.path.sep)[-1]
# #
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

        self.traj = []

        # preallocate for data from each episode
        for _ in range(self.n_episodes):
            self.traj.append(OrcaTraj(self.n_agents))


class OrcaTraj():
    """Class for organizing trajectory data by agent and state."""

    def __init__(self, n_agents):

        # access vector data e.g. self.Px[agent][time]
        self.Px = [] # x position for each agent
        self.Py = [] # y position for each agent
        self.Vx = [] # x velocity for each agent
        self.Vy = [] # y velocity for each agent

        # access vector data e.g. self.X[agent][time,state_index]
        self.X  = [] # full state of each robot
        self.Pg = []  # Goal state for each robot

        # access scalar data e.g. self.R[agent]
        self.R    = np.zeros(n_agents)
        self.Vmax = np.zeros(n_agents)

        for _ in range(n_agents):
            # access vector data e.g. self.Px[agent][time]
            self.Px.append(np.array([])) # x position for each agent
            self.Py.append(np.array([])) # y position for each agent
            self.Vx.append(np.array([])) # x velocity for each agent
            self.Vy.append(np.array([])) # y velocity for each agent

            # access vector data e.g. self.X[agent][time,state_index]
            self.X.append(np.array([])) # full state of each robot
            self.Pg.append(np.zeros(2)) # Goal state for each robot


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
    
        n_agents = datareader.__next__()[1] # Get the number of episodes
        n_episodes = datareader.__next__()[1] # Get the number of agents
        dt         = datareader.__next__()[1] # Get the simulation timestep


        param_dict["n_episodes"] = int(n_episodes)
        param_dict["n_agents"]   = int(n_agents)
        param_dict["dt"]         = float(dt)

        # create dict to map episode to trajectories
        data = TrainingData(param_dict)

        # trajectory data is accessed as:
        # data.traj[episode].Px[agent][time_index])

        for ep in range(data.n_episodes):
            header_row = datareader.__next__()
            radius_row = datareader.__next__()
            vmax_row   = datareader.__next__()
            goal_row   = datareader.__next__()
            traj_row   = datareader.__next__()

            print(header_row)
            for t in range(1,len(traj_row)-1,data.n_agents):
                for ag in range(data.n_agents):
                    # strip new line character, remove parens, split at space
                    r = traj_row[t+ag].rstrip("\n").replace("(","").replace(")","").split(" ")

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

            for idx in range(data.n_agents):
                data.traj[ep].R[idx]    = float(radius_row[idx+1])
                data.traj[ep].Vmax[idx] = float(vmax_row[idx+1])

                g = goal_row[idx+1].replace("(","").replace(")","").split(" ")

                data.traj[ep].Pg[idx][0]   = float(g[1])
                data.traj[ep].Pg[idx][1]   = float(g[2])

    return data


## Use for testing/checking outputs
# data_out = read_training_data(filename)

# print("EP 1")
# print(data_out.traj[0].Px[0][0:2])
# print(data_out.traj[0].R)
# print(data_out.traj[0].Pg[0])
# print(data_out.traj[0].Pg[1])
# print(data_out.traj[0].Vmax)

# print("EP 2")
# print(data_out.traj[1].Px[0][0:2])
# print(data_out.traj[1].R)
# print(data_out.traj[1].Pg[0])
# print(data_out.traj[1].Pg[1])
# print(data_out.traj[1].Vmax)
