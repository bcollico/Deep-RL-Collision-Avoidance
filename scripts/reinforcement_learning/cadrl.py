import numpy as np
import tensorflow as tf

import random
import itertools

from state_definitions import  get_joint_state_vectorized, get_rotated_state
from rl_utils import get_goal, get_pos, get_vel, get_current_state, get_radius, get_vpref, propagate_dynamics, \
robots_intersect, close_to_goal, get_heading, fill_vel_heading
from configs import *
from visualize_traj import plot_animation

def CADRL(value_model, initial_state_1, initial_state_2, epsilon, episode=0, test=False, reward_key='original'):
    '''
        Algorithm 1: CADRL (Collision Avoidance with Deep RL)

    '''
    dt = DT
    t  = 0

    # robot trajectories will be Tx9, where T is the total timesteps in the traj
    x_1 = initial_state_1.reshape(1, -1) # robot 1 state px, py, vx, vy, ...
    x_2 = initial_state_2.reshape(1, -1) # robot 2 state px, py, vx, vy, ...

    state_dim = initial_state_1.shape[0]

    # set heading towards goal state
    x_1[0, 8] = np.arctan2(get_goal(x_1)[1] - get_pos(x_1)[1], get_goal(x_1)[0] - get_pos(x_1)[0])
    x_2[0, 8] = np.arctan2(get_goal(x_2)[1] - get_pos(x_2)[1], get_goal(x_2)[0] - get_pos(x_2)[0])

    # x[0:2] - position
    # x[2:4] - velocity
    # x[5:7] - goal position

    # while distance between robots and there goals is greater than eps
    R1s, R2s, x1s_rot, x2s_rot = [], [], [], []

    done_1 = False
    done_2 = False

    while (not done_1) or (not done_2):
        t += dt

        n_timesteps_x1 = len(x_1)
        n_timesteps_x2 = len(x_2)

        v_filtered_1 = get_filtered_velocity(x_1)
        v_filtered_2 = get_filtered_velocity(x_2)
       
        # Question: how do we set the velocities for the these next states?
        x1_o_nxt = propagate_dynamics(get_current_state(x_1), v_filtered_1, dt)
        x2_o_nxt = propagate_dynamics(get_current_state(x_2), v_filtered_2, dt)

        # if t>1.0 and (np.linalg.norm(v_filtered_1 - np.array([0,0])) < 1e-3 or np.linalg.norm(v_filtered_1 - np.array([0,0])) < 1e-3) :
        #     print(v_filtered_1)
        #     print(v_filtered_2)
        #     import pdb;pdb.set_trace()

        x1_o_nxt = fill_vel_heading(x1_o_nxt, v_filtered_1)
        x2_o_nxt = fill_vel_heading(x2_o_nxt, v_filtered_2)

        num_sampled_actions = 51

        # A1 = build_action_space_simple(num_sampled_actions, VMAX)   
        # A2 = build_action_space_simple(num_sampled_actions, VMAX) 

        A1 = build_action_space(KINEMATIC, x_1)     
        A2 = build_action_space(KINEMATIC, x_2)     

        gamma_bar_x1 = GAMMA**(dt*get_vpref(x_1))
        gamma_bar_x2 = GAMMA**(dt*get_vpref(x_2))

        x1_nxt = np.zeros((num_sampled_actions, state_dim))
        x2_nxt = np.zeros((num_sampled_actions, state_dim))

        # attempt at vectorizing this A loop

        curr_state_1 = np.repeat(get_current_state(x_1).reshape(1,-1), repeats=num_sampled_actions, axis=0)
        curr_state_2 = np.repeat(get_current_state(x_2).reshape(1,-1), repeats=num_sampled_actions, axis=0)

        x1_nxt = propagate_dynamics(curr_state_1, A1, dt)
        x2_nxt = propagate_dynamics(curr_state_2, A2, dt)

        x1_nxt = fill_vel_heading(x1_nxt, A1)
        x2_nxt = fill_vel_heading(x2_nxt, A2)

        x1_o_nxt_all = np.repeat(x1_o_nxt.reshape(1,-1), repeats=num_sampled_actions, axis=0)
        x2_o_nxt_all = np.repeat(x2_o_nxt.reshape(1,-1), repeats=num_sampled_actions, axis=0)

        x1_joint = get_joint_state_vectorized(x1_nxt, x2_o_nxt_all)
        x2_joint = get_joint_state_vectorized(x2_nxt, x1_o_nxt_all)

        x1_joint_rotated = np.apply_along_axis(get_rotated_state, 1, x1_joint)
        x2_joint_rotated = np.apply_along_axis(get_rotated_state, 1, x2_joint)

        if reward_key == 'original':
            R1 = reward_vectorized(curr_state_1, curr_state_2, A1, dt)
            R2 = reward_vectorized(curr_state_2, curr_state_1, A2, dt)
        elif reward_key == 'reward_1':
            R1 = reward_vectorized(curr_state_1, curr_state_2, A1, dt, reward_type=1)
            R2 = reward_vectorized(curr_state_2, curr_state_1, A2, dt, reward_type=1)
        elif reward_key == 'reward_2':
            R1 = reward_vectorized(curr_state_1, curr_state_2, A1, dt, reward_type=2)
            R2 = reward_vectorized(curr_state_2, curr_state_1, A2, dt, reward_type=2)

        lookahead_1 = R1 + gamma_bar_x1 * value_model(x1_joint_rotated)
        lookahead_2 = R2 + gamma_bar_x2 * value_model(x2_joint_rotated)

        

        if not test and random.random() < epsilon:
            idx1 = np.random.randint(0, len(A1)-1)
            opt_action_1 = A1[idx1]
        else:
            idx1 = np.argmax(lookahead_1)
            opt_action_1 = A1[idx1]

        if not test and random.random() < epsilon:
            idx2 = np.random.randint(0, len(A2)-1)
            opt_action_2 = A2[idx2]
        else:
            idx2 =  np.argmax(lookahead_2)
            opt_action_2 = A2[idx2]

        if not done_1:  
            x_1_new = propagate_dynamics(get_current_state(x_1), opt_action_1, dt).reshape(1, -1)
            # Q: how should we fill out the velocity in the states? previous timestep?
            x_1[-1, 2:4] = opt_action_1 # velocity
            x_1[-1, 8]   = np.arctan2(opt_action_1[1], opt_action_1[0]) # heading, theta
            x_1 = np.append(x_1, x_1_new, axis=0) 
            R1s.append(R1[idx1])
            x1s_rot.append(x1_joint_rotated[idx1])     
        if not done_2:
            x_2_new = propagate_dynamics(get_current_state(x_2), opt_action_2, dt).reshape(1, -1)
            x_2[-1, 2:4] = opt_action_2 # velocity
            x_2[-1, 8]   = np.arctan2(opt_action_2[1], opt_action_2[0]) # heading, theta
            x_2 = np.append(x_2, x_2_new, axis=0)
            R2s.append(R2[idx2])
            x2s_rot.append(x2_joint_rotated[idx2])       
        # print(np.linalg.norm(get_pos(x_1) - get_goal(x_1)))
        # print(np.linalg.norm(get_pos(x_2) - get_goal(x_2)))

        if close_to_goal(x_1):
            done_1 = True

        if close_to_goal(x_2):
            done_2 = True

        if robots_intersect(x_1, x_2):
            done_1 = True
            done_2 = True

        if n_timesteps_x1*dt > MAX_TIME or n_timesteps_x2*dt > MAX_TIME:
            #print('Robots did not reach goal')

            # plot_animation(get_goal(x_1),
            #                get_goal(x_2),
            #                x_1[:, 0:2],
            #                x_2[:, 0:2],
            #                get_radius(x_1),
            #                get_radius(x_2))
            R2s.append(R2[idx2])
            x2s_rot.append(x2_joint_rotated[idx2]) 
            R1s.append(R1[idx1])
            x1s_rot.append(x1_joint_rotated[idx1])       
            return x_1, x_2, False, np.array(R1s), np.array(R2s), np.array(x1s_rot), np.array(x2s_rot), False

    if robots_intersect(x_1, x_2):
        #print('Reached goal but intersected')
        # if True:
        # # if False and episode>=7:
        #     plot_animation(get_goal(x_1),
        #                 get_goal(x_2),
        #                 x_1[:, 0:2],
        #                 x_2[:, 0:2],
        #                 get_radius(x_1),
        #                 get_radius(x_2))

        R2s.append(R2[idx2])
        x2s_rot.append(x2_joint_rotated[idx2]) 
        R1s.append(R1[idx1])
        x1s_rot.append(x1_joint_rotated[idx1]) 
        return x_1, x_2, False, np.array(R1s), np.array(R2s), np.array(x1s_rot), np.array(x2s_rot), True
        # TODO - if a robot intersects another but still reaches the goal, should this be counted in training?
    else:
        # if episode >= 7:
        #     plot_animation(get_goal(x_1),
        #             get_goal(x_2),
        #             x_1[:, 0:2],
        #             x_2[:, 0:2],
        #             get_radius(x_1),
        #             get_radius(x_2))
        R2s.append(R2[idx2])
        x2s_rot.append(x2_joint_rotated[idx2]) 
        R1s.append(R1[idx1])
        x1s_rot.append(x1_joint_rotated[idx1])      
        return x_1, x_2, True, np.array(R1s), np.array(R2s), np.array(x1s_rot), np.array(x2s_rot), False

def reward_vectorized(x1, x2, a, dt, test1=False, test2=False, reward_type=0):
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

    dmin = np.min(dist, axis=1) - (get_radius(x1) + get_radius(x2))

    x1_nxt = propagate_dynamics(x1, a, dt)

    R             = np.zeros(len(a))
    goal_close    = np.linalg.norm(x1_nxt[:, 0:2] - x1_nxt[:, 5:7], axis=1) < GOAL_EPS
    R[goal_close] = 1
    dmin_idx      = dmin < 0.2
    R[dmin_idx]   = -0.1 - dmin[dmin_idx]/2
    dmin_idx_2    = dmin < 0.0
    R[dmin_idx_2] = -0.25

    if reward_type == 1:
        dmin_idx      = dmin < 0.5
        R[dmin_idx]   = -0.5 - dmin[dmin_idx]/2
        dmin_idx_2    = dmin < 0.0
        R[dmin_idx_2] = -0.75

    if reward_type == 2:
        pass

    return R.reshape(-1, 1)

def build_action_space(kinematic, state):
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
        heading = get_heading(state[:-1])
    else:
        heading = get_heading(state)

    A      = np.zeros_like(actions)
    A[:,0] = actions[:,0]*np.cos(heading + actions[:,1])
    A[:,1] = actions[:,0]*np.sin(heading + actions[:,1])

    a, b = A.shape

    assert a == 51
    assert b == 2

    # if len(state)>1:
    #     from matplotlib import pyplot as plt
    #     plt.figure()
    #     plt.scatter(get_vel(state[:-1])[0], get_vel(state[:-1])[1], c='k', s=200)
    #     for a in A:
    #         plt.scatter(a[0], a[1], c='r')
    #     plt.show()

    # if len(state)>1:
    #     print(get_vel(state[:-1]))

    # import pdb;pdb.set_trace()

    return A

def build_action_space_simple(num_actions, vmax):
    A = np.random.uniform(low=-vmax, high=vmax, size=(num_actions-1, 2)) # TODO: how should we sample actions?
    A = np.append(A, np.array([[0, 0]]), axis=0) # adding option to do nothing (if robot is at goal)

    return A

def get_filtered_velocity(x):
    # v_filtered = np.ma.average(x[:-1, 2:4], axis=0, weights = np.exp(range(len(x[:-1])))) # weights the more recent scores more

    # if len(x) <= 4 and len(x) > 1:
    #     v_filtered = np.ma.average(x[:-1, 2:4], axis=0, weights = np.exp(range(len(x[:-1]))))
    # elif len(x) > 4:
    #     v_filtered = np.ma.average(x[-5:-1, 2:4], axis=0, weights = np.exp(range(4)))
    # else:
    #     v_filtered = np.array([0, 0])

    if len(x) <= 3 and len(x) > 1:
        v_filtered = np.mean(x[:-1, 2:4], axis=0)
    elif len(x) > 3:
        v_filtered = np.mean(x[-4:-1, 2:4], axis=0)
    else:
        v_filtered = np.array([0, 0])

    # if len(x) > 1:
    #     v_filtered = x[-2, 2:4]
    # else:
    #     v_filtered = np.array([0, 0])

    return v_filtered
