from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from random import random
import read_training_data

USER = 'Bradley'

if USER == 'Brian':
	folder  = '/home/bdobkowski/Stanford/AA277/aa277_project/data'
elif USER == 'Torstein':
	folder  = '/home/torstein/Stanford/aa277/aa277_project/data'
elif USER == 'Bradley':
    folder  = '/home/bcollico/github/aa277_project/data'
else:
	raise Exception('Need to set user folder')

def plot_traj(states:dict(), goals:dict, radius:dict(), dt=0.1):

    # x_ep_dict[1][2][3] is 10-dimensional, and contains the state from get_state() in model.py
    # this function assumes that x_ep_dict[i_ep] is passed as the input

    n_agents = len(states.keys())
    if n_agents < 5:
        colors = ['b', 'g', 'r', 'k', 'y']
    else:
        colors = [(random(), random(), random()) for _ in range(len)]

    ax = plt.axes()
    ax.set_title("Collision Avoidance Trajectories")
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

    circle_idx = np.int64((np.linspace(0,min_traj-1,2)))

    for key in states.keys():

        i_traj   = states[key]
        i_radius = radius[key]
        i_goal   = goals[key]

        steps_to_goal = np.sum([np.linalg.norm(i_traj[2:4,:], axis=0)>0.05])
        # steps_to_goal = len(i_traj) - np.sum([np.linalg.norm(i_traj[0:2,:]-i_goal.reshape(-1,1), axis=0)<0.1])

        ax.plot(i_traj[0,:steps_to_goal],
                i_traj[1,:steps_to_goal], 
                linestyle='--',
                color=colors[int(key)], 
                label="Robot "+key)

        ax.plot(i_goal[0],
                i_goal[1],
                color=colors[int(key)],
                 marker='*',
                 markersize=12)

        circle_idx_i = circle_idx
        circle_idx_i[-1] = steps_to_goal

        for i in circle_idx_i:
            
            ax.plot(i_radius*circ_points[0,:]+i_traj[0,i],
                    i_radius*circ_points[1,:]+i_traj[1,i],
                    color=colors[int(key)])

            ax.plot(i_traj[0,i],
                    i_traj[1,i],
                    color=colors[int(key)],
                    marker='.')

            ax.text(i_traj[0,i]-i_radius/5, 
                    i_traj[1,i]+i_radius/2*(-1)**int(key), 
                    "t={:g}".format(i*dt),
                    color=colors[int(key)])

    plt.legend()
    plt.show()


if __name__=='__main__':
    path = folder+"/training_data_100sim.csv"
    # path = folder+"/static_tests.csv"
    data = read_training_data.read_training_data(path)

    # ep_list = np.int64(np.linspace(1,999,10))
    # ep_list = range(2)
    ep_list = [53]
    for ep in ep_list:
        states = dict()
        goals  = dict()
        radius = dict()

        episode = data.traj[ep]
        for i in range(data.n_agents):
            states[str(i)] = episode.X[i][:,0:4].T
            goals[str(i)]  = episode.Pg[i]
            radius[str(i)] = episode.R[i]

        plot_traj(states, goals, radius, dt=data.dt)

