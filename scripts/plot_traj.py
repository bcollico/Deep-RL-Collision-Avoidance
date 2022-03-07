from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from random import random
from reinforcement_learning import read_training_data

USER = 'Bradley'

if USER == 'Brian':
	folder  = '/home/bdobkowski/Stanford/AA277/aa277_project/data'
elif USER == 'Torstein':
	folder  = '/home/torstein/Stanford/aa277/aa277_project/data'
elif USER == 'Bradley':
    folder  = '/home/bcollico/github/aa277_project/data'
else:
	raise Exception('Need to set user folder')

def plot_traj(states:dict, goals:dict, radius:dict, dt=0.1):

    n_agents = len(states.keys())
    if n_agents < 5:
        colors = ['b', 'g', 'r', 'k', 'y']
    else:
        colors = [(random(), random(), random()) for _ in range(len)]

    ax = plt.axes()
    ax.set_title("CADRL Trajectories")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_xlim([-5.5,5.5])
    ax.set_ylim([-3.5,3.5])
    ax.grid(True)
    ax.set_aspect(1)

    theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
    circ_points = np.vstack((np.cos(theta), np.sin(theta)))

    min_traj = np.inf
    for key in states.keys():
        min_traj = np.min([min_traj, len(states[key][0,:])])        

    circle_idx = np.int64((np.linspace(0,min_traj-1,4)))
    print(circle_idx)

    for key in states.keys():

        i_traj   = states[key]
        i_radius = radius[key]
        i_goal   = goals[key]

        ax.plot(i_traj[0,:],
                i_traj[1,:], 
                linestyle='--',
                color=colors[int(key)], 
                label="Robot "+key)

        ax.plot(i_goal[0],
                i_goal[1],
                color=colors[int(key)],
                 marker='*',
                 markersize=12)

        for i in circle_idx[1:-1]:
            ax.plot(i_radius*circ_points[0,:]+i_traj[0,i],
                    i_radius*circ_points[1,:]+i_traj[1,i],
                    color=colors[int(key)])

            ax.plot(i_traj[0,i],
                    i_traj[1,i],
                    color=colors[int(key)],
                    marker='.')

    plt.legend()
    plt.show()


if __name__=='__main__':
    # path = folder+"/training_data_1000sim_no_overlap.csv"
    path = folder+"/training_data_1000sim_no_overlap.csv"
    data = read_training_data.read_training_data(path)

    ep_list = np.int64(np.linspace(1,999,10))
    # ep_list = [0]
    for ep in ep_list:
        states = dict()
        goals  = dict()
        radius = dict()

        episode = data.traj[ep]
        for i in range(data.n_agents):
            states[str(i)] = episode.X[i][:,0:2].T
            goals[str(i)]  = episode.Pg[i]
            radius[str(i)] = episode.R[i]

        plot_traj(states, goals, radius, dt=data.dt)

