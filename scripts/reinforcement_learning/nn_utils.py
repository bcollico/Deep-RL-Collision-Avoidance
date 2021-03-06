import numpy as np
from read_training_data import read_training_data
import json
import os
from state_definitions import  get_joint_state, get_rotated_state, get_state, get_joint_state_vectorized
from configs import *
GAMMA = 0.8

def load_traj_data(folder, data_type='train'):

    if data_type == 'train': 
        data = read_training_data(os.path.join(folder, 'train_data.csv'))
    else:
        data = read_training_data(os.path.join(folder, 'test_data.csv'))

    radius = 1.0
    x_ep_dict = {}

    episodes_count = len(data.traj)

    for ep in range(episodes_count):

        x_robot_dict = {}

        traj = data.traj[ep]
        v_pref = max(traj.Vmax[0], traj.Vmax[1])
        robot_count = len(traj.Pg)
        
        for i in range(robot_count):
            pgx_i = traj.Pg[i][0]
            pgy_i = traj.Pg[i][1]

            xs = []


            N = len(data.traj[ep].X[i])

            for timestep in range(N):
                
                s_robo_i = data.traj[ep].X[i][timestep]
                state_i = get_state(s_robo_i, radius, pgx_i, pgy_i, v_pref)
                xs.append(state_i)

            x_robot_dict[i] = np.array(xs)

        x_ep_dict[ep] = x_robot_dict

    return x_ep_dict, v_pref, data.dt


def get_nn_input(x_dict, dt, v_pref):
 
    #v_pref = VMAX
    gamma = GAMMA
    #dt = DT


    
    def get_y(v_pref, gamma, tg):
        return gamma**(tg*v_pref)


    x_rotated = {}
    y_out = {}
    for ep, x_ep_dict in x_dict.items():
        
        y_robot_dict, x_robot_dict_rotated = {}, {}

        for i, i_dict in x_ep_dict.items():
      
            #Only add states when we have trajectories from both robots

            ys ,xs_rotated = [], []


            for j, j_dict in x_ep_dict.items():

                if i == j :
                    continue

                N = min(len(i_dict),len(j_dict))

                for timestep in range(N):
                    
                    state_i = i_dict[timestep]
                    state_j = j_dict[timestep]

                    joint = get_joint_state(state_i, state_j)

                    xs_rotated.append(get_rotated_state(joint).tolist())

                    y = get_y(v_pref, gamma, (N-timestep+1)*(dt))
                    ys.append((y).tolist())





            ##obs need two robots
            x_robot_dict_rotated[i] = xs_rotated
            y_robot_dict[i] = ys

        x_rotated[ep] = x_robot_dict_rotated
        y_out[ep] = y_robot_dict

   
    return x_rotated, y_out


def load_nn_data(x_dict, y_dict):

    x_dim = 15
    xs = np.empty((0, x_dim))
    ys = np.array([])

    for xk_ep, yk_ep in zip(x_dict.values(), y_dict.values()):
        for xk, yk in zip(xk_ep.values(), yk_ep.values()):
        
            xs = np.vstack([xs, np.asarray(xk)])
            ys = np.concatenate((ys, np.asarray(yk)))

    split = 2*len(ys)//3

    assert(xs.shape[0] == ys.shape[0]), "xs and ys shape different"
    shuffler = np.random.permutation(len(xs))
    xs, ys = xs[shuffler], ys[shuffler]

    x_train = xs[:split]
    y_train = ys[:split]

    x_test = xs[split:]
    y_test = ys[split:]

    return x_train, y_train, x_test, y_test

def create_train_set_from_dict(x_dict, y_dict):

    dim_rot_joint_state = 15
    xs = np.array([]).reshape((0, dim_rot_joint_state))
    ys = np.array([])

    for xk, yk in zip(x_dict.keys(), y_dict.keys()):

        x1_joint = get_joint_state_vectorized(x_dict[xk][0], x_dict[xk][1])
        x2_joint = get_joint_state_vectorized(x_dict[xk][1], x_dict[xk][0])

        x1_joint_rotated = np.apply_along_axis(get_rotated_state, 1, x1_joint)
        x2_joint_rotated = np.apply_along_axis(get_rotated_state, 1, x2_joint)

        xs = np.vstack([xs, x1_joint_rotated])
        xs = np.vstack([xs, x2_joint_rotated])

        try:
            ys = np.concatenate((ys, y_dict[yk][0]))
            ys = np.concatenate((ys, y_dict[yk][1]))
        except:
            ys = np.concatenate((ys, y_dict[yk][0].numpy().flatten()))
            ys = np.concatenate((ys, y_dict[yk][1].numpy().flatten()))

    return xs, ys

