import numpy as np
import tensorflow as tf
import os
import model
import json
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from visualize_traj import animate, plot_animation
import random

from state_definitions import  get_joint_state, get_joint_state_vectorized, get_rotated_state, get_state
from nn_utils import get_nn_input, load_traj_data
from rl_utils import get_goal, get_pos, get_vel, get_current_state, get_radius, get_vpref, propagate_dynamics, \
robots_intersect, close_to_goal
from model import LR, USER, FOLDER
from configs import *

def CADRL(value_model, initial_state_1, initial_state_2, epsilon, dt, episode=0):
    '''
        Algorithm 1: CADRL (Collision Avoidance with Deep RL)

        INCOMPLETE

        Doesn't work yet. Also haven't implemented epsilon-greedy exploration yet
    '''
    t  = 0

    # robot trajectories will be Tx9, where T is the total timesteps in the traj
    x_1 = initial_state_1.reshape(1, -1) # robot 1 state px, py, vx, vy, ...
    x_2 = initial_state_2.reshape(1, -1) # robot 2 state px, py, vx, vy, ...

    state_dim = initial_state_1.shape[0]

    # x[0:2] - position
    # x[2:4] - velocity
    # x[5:7] - goal position

    # while distance between robots and there goals is greater than eps
    R1s, R2s, x1s_rot, x2s_rot = [], [], [], []
    while (not close_to_goal(x_1)) or (not close_to_goal(x_2)):
        t += dt

        n_timesteps_x1 = len(x_1)
        n_timesteps_x2 = len(x_2)

        v_filtered_1 = np.ma.average(x_1[:-1, 2:4], axis=0, weights = np.exp(range(len(x_1[:-1])))) # weights the more recent scores more
        v_filtered_2 = np.ma.average(x_2[:-1, 2:4], axis=0, weights = np.exp(range(len(x_2[:-1])))) # weights the more recent scores more

        # Question: how do we set the velocities for the ~ robot in CADRL?
        # My feeling is we do both robots in CADRL simultaneously
        # To do this, I think we'd need to save and load states for individual robots 
        # rather than loading joint states/trajectories
        x1_o_nxt = propagate_dynamics(get_current_state(x_1), v_filtered_1, dt)
        x2_o_nxt = propagate_dynamics(get_current_state(x_2), v_filtered_2, dt)

        num_sampled_actions = 50

        A = np.random.uniform(low=-VMAX, high=VMAX, size=(num_sampled_actions-1, 2)) # TODO: how should we sample actions?
        A = np.append(A, np.array([[0, 0]]), axis=0) # adding option to do nothing (if robot is at goal)

        gamma_bar_x1 = GAMMA**(dt*get_vpref(x_1))
        gamma_bar_x2 = GAMMA**(dt*get_vpref(x_2))

        x1_nxt = np.zeros((num_sampled_actions, state_dim))
        x2_nxt = np.zeros((num_sampled_actions, state_dim))

        # attempt at vectorizing this A loop

        curr_state_1 = np.repeat(get_current_state(x_1).reshape(1,-1), repeats=num_sampled_actions, axis=0)
        curr_state_2 = np.repeat(get_current_state(x_2).reshape(1,-1), repeats=num_sampled_actions, axis=0)

        x1_nxt = propagate_dynamics(curr_state_1, A, dt)
        x2_nxt = propagate_dynamics(curr_state_2, A, dt)

        x1_o_nxt_all = np.repeat(x1_o_nxt.reshape(1,-1), repeats=num_sampled_actions, axis=0)
        x2_o_nxt_all = np.repeat(x2_o_nxt.reshape(1,-1), repeats=num_sampled_actions, axis=0)

        x1_joint = get_joint_state_vectorized(x1_nxt, x2_o_nxt_all)
        x2_joint = get_joint_state_vectorized(x2_nxt, x1_o_nxt_all)

        x1_joint_rotated = np.apply_along_axis(get_rotated_state, 1, x1_joint)
        x2_joint_rotated = np.apply_along_axis(get_rotated_state, 1, x2_joint)

        R1 = reward_vectorized(curr_state_1, curr_state_2, A, dt)
        R2 = reward_vectorized(curr_state_2, curr_state_1, A, dt)


        lookahead_1 = R1 + gamma_bar_x1 * value_model(x1_joint_rotated)
        lookahead_2 = R2 + gamma_bar_x2 * value_model(x2_joint_rotated)

        try:
            assert np.all(value_model(x1_joint_rotated)<1.0)
        except:
            import pdb;pdb.set_trace()

        #####################################################################################
        # Brian: I will delete this eventually, but I still use it to test that my vectorized reward  
        # function gives me the same results as the regular reward function

        # x1_nxt__ = np.zeros((num_sampled_actions, state_dim))
        # x2_nxt__ = np.zeros((num_sampled_actions, state_dim))

        # lookahead_1__ = np.zeros(num_sampled_actions)
        # lookahead_2__ = np.zeros(num_sampled_actions)

        # for idx, a in enumerate(A):
        #     x1_nxt__[idx] = propagate_dynamics(get_current_state(x_1), a, dt)
        #     x2_nxt__[idx] = propagate_dynamics(get_current_state(x_2), a, dt)

        #     x1_joint__ = model.get_joint_state2(x1_nxt__[idx], x2_o_nxt)
        #     x2_joint__ = model.get_joint_state2(x2_nxt__[idx], x1_o_nxt)

        #     x1_joint_rotated__ = model.get_rotated_state(x1_joint__)
        #     x2_joint_rotated__ = model.get_rotated_state(x2_joint__)

        #     assert np.linalg.norm(x1_joint_rotated__ - x1_joint_rotated[idx]) < 1e-4
        #     assert np.linalg.norm(x2_joint_rotated__ - x2_joint_rotated[idx]) < 1e-4

        #     # does our action have no impact on reward here? should we feed it the propagated state?
        #     R1__ = reward(get_current_state(x_1), get_current_state(x_2), a, dt)
        #     R2__ = reward(get_current_state(x_2), get_current_state(x_1), a, dt)

        #     try:
        #         assert np.linalg.norm(R1[idx] - R1__) < 1e-4
        #         assert np.linalg.norm(R2[idx] - R2__) < 1e-4
        #     except:
        #         import pdb;pdb.set_trace()

        #     lookahead_1__[idx] = R1__ + gamma_bar_x1 * value_model(x1_joint_rotated__.reshape(1,-1))
        #     lookahead_2__[idx] = R2__ + gamma_bar_x2 * value_model(x2_joint_rotated__.reshape(1,-1))

        # try:
        #     assert np.linalg.norm(lookahead_1 - lookahead_1__.reshape(-1, 1)) < 1e-3
        #     assert np.linalg.norm(lookahead_2 - lookahead_2__.reshape(-1, 1)) < 1e-3
        # except:
        #     import pdb;pdb.set_trace()

        ######################################################################################################
        if random.random() < epsilon:
            idx = np.random.randint(0, len(A)-1)
            opt_action_1 = random.choice(A)
        else:
            idx = np.argmax(lookahead_1)
            opt_action_1 = A[idx]
        R1s.append(R1[idx])
        x1s_rot.append(x1_joint_rotated[idx])
        if random.random() < epsilon:
            idx = np.random.randint(0, len(A)-1)
            opt_action_2 = A[idx]
        else:
            idx =  np.argmax(lookahead_2)
            opt_action_2 = A[np.argmax(lookahead_2)]
        R2s.append(R2[idx])
        x2s_rot.append(x2_joint_rotated[idx])

        if not close_to_goal(x_1):  
            x_1_new = propagate_dynamics(get_current_state(x_1), opt_action_1, dt).reshape(1, -1)
            # Q: how should we fill out the velocity in the states? previous timestep?
            x_1[-1, 2:4] = opt_action_1 # velocity
            x_1[-1, 8] = np.arctan2(opt_action_1[1], opt_action_1[0]) # heading, theta
            x_1 = np.append(x_1, x_1_new, axis=0)      
        if not close_to_goal(x_2):
            x_2_new = propagate_dynamics(get_current_state(x_2), opt_action_2, dt).reshape(1, -1)
            x_2[-1, 2:4] = opt_action_2 # velocity
            x_2[-1, 8] = np.arctan2(opt_action_2[1], opt_action_2[0]) # heading, theta
            x_2 = np.append(x_2, x_2_new, axis=0)

        # print(np.linalg.norm(get_pos(x_1) - get_goal(x_1)))
        # print(np.linalg.norm(get_pos(x_2) - get_goal(x_2)))

        if n_timesteps_x1*dt > 25 or n_timesteps_x2*dt > 25:

            # plot_animation(get_goal(x_1),
            #                get_goal(x_2),
            #                x_1[:, 0:2],
            #                x_2[:, 0:2],
            #                get_radius(x_1),
            #                get_radius(x_2))

            return x_1, x_2, False, np.array(R1s), np.array(R2s), np.array(x1s_rot), np.array(x2s_rot)

    if robots_intersect(x_1, x_2):
        print('Reached goal but intersected')
        if False and episode>=7:
            plot_animation(get_goal(x_1),
                        get_goal(x_2),
                        x_1[:, 0:2],
                        x_2[:, 0:2],
                        get_radius(x_1),
                        get_radius(x_2))

        return x_1, x_2, False, np.array(R1s), np.array(R2s), np.array(x1s_rot), np.array(x2s_rot)
        # TODO - if a robot intersects another but still reaches the goal, should this be counted in training?
    else:
        if episode >= 7:
            plot_animation(get_goal(x_1),
                    get_goal(x_2),
                    x_1[:, 0:2],
                    x_2[:, 0:2],
                    get_radius(x_1),
                    get_radius(x_2))
        return x_1, x_2, True, np.array(R1s), np.array(R2s), np.array(x1s_rot), np.array(x2s_rot)

def reward(x1, x2, a, dt):
    '''
    Reward function for robot 1 only, given joint state (parametrized by 2 individual states)
    '''
    x2_const_vel = get_vel(x2)

    # below adapted from public CADRL repo
    dmin = float('inf')
    dmin_time = 1
    for frac in np.linspace(0, 1, num=5):
        time = frac*dt
        x1_pos = get_pos(propagate_dynamics(x1, a, time))
        x2_pos = get_pos(propagate_dynamics(x2, x2_const_vel, time))
        distance = np.linalg.norm(x1_pos - x2_pos)
        if distance < dmin:
            dmin = distance
            dmin_time = time

    x1_nxt = propagate_dynamics(x1, a, dt)

    if dmin < get_radius(x1) + get_radius(x2):
        R = -0.25
        end_time = dmin_time
    else:
        end_time = 1
        if dmin < get_radius(x1) + get_radius(x2) + 0.2:
            R = -0.1 - dmin/2
        elif close_to_goal(x1_nxt):
            R = 1
        else:
            R = 0

    return R

def reward_vectorized(x1, x2, a, dt):
    '''
    Reward function for robot 1 only, given joint state (parametrized by 2 individual states)
    '''
    x2_const_vel = x2[:, 2:4]

    # below adapted from public CADRL repo
    num_interps = 4
    dist = 10000*np.ones((len(a), num_interps))
    for idx, frac in enumerate(np.linspace(0, 1, num=num_interps)):
        time = frac*dt
        x1_states = propagate_dynamics(x1, a, time)
        x2_states = propagate_dynamics(x2, x2_const_vel, time)
        dist[:, idx] = np.linalg.norm(x1_states[:, 0:2] - x2_states[:, 0:2], axis=1)

    dmin = np.min(dist, axis=1)

    x1_nxt = propagate_dynamics(x1, a, dt)

    R = np.zeros(len(a))
    goal_close = np.linalg.norm(x1_nxt[:, 0:2] - x1_nxt[:, 5:7], axis=1) < 1e-1
    R[goal_close] = 1
    dmin_idx = dmin < get_radius(x1.T) + get_radius(x2.T) + 0.2
    R[dmin_idx] = np.multiply(-0.1 - dmin[dmin_idx]/2, np.ones(len(dmin[dmin_idx])))
    dmin_idx_2 = dmin < get_radius(x1.T) + get_radius(x2.T)
    R[dmin_idx_2] = -0.25

    return R.reshape(-1, 1)