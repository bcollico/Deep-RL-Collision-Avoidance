import numpy as np

import itertools

from state_definitions import  get_state
from rl_utils import get_vel, get_heading, get_vpref
from configs import *


def plot_action_space(kinematic, state):
    '''
    25 precomputed actions and 10 random actions to choose from in lookahead function

    Ref: cadrl online repo: https://github.com/ChanganVR/CADRL/blob/master/train.py
    '''

    # actions array is in velocity mag, rotation angle

    if kinematic:
        velocities = [(i + 1) / 5 * get_vpref(state) for i in range(5)]
        rotations  = [i/4*np.pi/3 - np.pi/6 for i in range(5)]
        actions    = np.array([x for x in itertools.product(velocities, rotations)])
        a = np.random.uniform(low=0.0, high=get_vpref(state), size=(25, 1)) # TODO: how should we sample actions?
        b = np.random.uniform(low=-np.pi/6, high=np.pi/6, size=(25, 1)) # TODO: how should we sample actions?
        actions = np.append(actions, np.concatenate((a,b), axis=1), axis=0) # adding option to do nothing (if robot is at goal)
        actions = np.append(actions, np.array([0, 0]).reshape(1,2), axis=0)
    else:
        velocities = [(i + 1) / 5 * get_vpref(state) for i in range(5)]
        rotations  = [i / 4 * 2 * np.pi for i in range(5)]
        actions    = np.array([x for x in itertools.product(velocities, rotations)])
        a = np.random.uniform(low=0.0, high=get_vpref(state), size=(25, 1)) # TODO: how should we sample actions?
        b = np.random.uniform(low=-np.pi, high=np.pi, size=(25, 1)) # TODO: how should we sample actions?
        actions = np.append(actions, np.concatenate((a,b), axis=1), axis=0) # adding option to do nothing (if robot is at goal)
        actions = np.append(actions, np.array([0, 0]).reshape(1,2), axis=0)

    # transforming to x, y cartesian velocities
    if len(state)>1:
        heading = get_heading(state)
    else:
        heading = get_heading(state)

    A      = np.zeros_like(actions)
    A[:,0] = actions[:,0]*np.cos(heading + actions[:,1])
    A[:,1] = actions[:,0]*np.sin(heading + actions[:,1])

    a, b = A.shape

    assert a == 51
    assert b == 2

    theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
    circ_points = np.vstack((np.cos(theta), np.sin(theta)))

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(0,0,color='k',marker='.',markersize=10,linestyle='none')
    ax.quiver(0,0,get_vel(state[:-1])[0], get_vel(state[:-1])[1], color='k', angles='xy', 
        scale_units='xy', 
        scale=1,label='Current Vel')

    for i in range(5):
        ax.plot(velocities[i]*circ_points[0,:],
            velocities[i]*circ_points[1,:],
            color='k',linestyle='--', linewidth=0.5)
    # for a in A:
    ax.plot(A[:,0], A[:,1], color='r', marker='.',markersize=10,label='Action Space',linestyle='none')
    ax.set_title("Lookahead Action Space")
    ax.set_xlabel("X Direction")
    ax.set_ylabel("Y Direction")
    ax.axis([0,1.05,0,1.05])
    ax.set_aspect(1)
    plt.legend()
    plt.show()

    if len(state)>1:
        print(get_vel(state[:-1]))


if __name__ == '__main__':
    s = get_state(np.array([0,0,0.5,0.5]), 1,0,0,1)
    kinematic=True

    plot_action_space(kinematic, s)
