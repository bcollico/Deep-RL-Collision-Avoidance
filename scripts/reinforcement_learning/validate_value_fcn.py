from asyncore import read
import numpy as np
from read_training_data import read_training_data
from plot_traj import plot_traj
from configs import *
from nn_utils import load_traj_data
from state_definitions import get_state, get_joint_state, get_rotated_state
import tensorflow as tf
import os
from cadrl import CADRL

def evaluate_value_fcn_propagate(value_fnc, s_initial_1, s_initial_2, visualize):
    
    
    xs1, xs2, cadrl_successful, Rs1, Rs2, x1s_rot, x2s_rot, collision = CADRL(value_fnc, s_initial_1, s_initial_2, 100)

    goals = [xs1[0, 5:6],  xs2[0, 5:6]]

    dt = DT
    gamma = GAMMA
    rotated_states = [x1s_rot, x2s_rot]
    avg_value_diff = np.array([])
    avg_vel_diff = np.array([])
    avg_extra_time = np.array([])
    for i, xs in enumerate([xs1, xs2]):

        i_radius = RADIUS
        i_goal   = goals[i]
        i_vpref  = xs[0, 7]

        i_dg     = np.linalg.norm(xs[:, 0:2]-i_goal, axis=0)

        steps_to_goal = np.sum([np.linalg.norm(xs[:, 2:4], axis=0)>0.05])

        output_value = np.zeros((1,steps_to_goal))
        true_value = np.zeros((1,steps_to_goal))
        extra_time = np.zeros((1,steps_to_goal))

        
        for j, xs_other in enumerate([xs1, xs2]):
            if j is not i:
                j_radius = RADIUS
                min_idx = min(len(xs), len(xs_other))
                collision = not np.all(np.linalg.norm(xs[:min_idx, 0:2]-xs_other[:min_idx, 0:2],axis=0)>(i_radius+j_radius))
                # print(np.linalg.norm(i_traj[0:2,:]-j_traj[0:2,:],axis=0))
                if collision:
                    print(f"Collision between agents {i} and {j}")


        for step in range(steps_to_goal):

            dg      = i_dg[step]
            t       = step*dt
            tg      = (steps_to_goal-step)*dt



            s_12 = rotated_states[i][step]

            # print(s_12)

            output_value[0,step] = value_fnc(np.array([s_12]))
            true_value[0,step]   = gamma**(tg*i_vpref)
            extra_time[0,step]   = tg - dg/i_vpref

    

        avg_value_diff = np.append(avg_value_diff, np.mean(np.abs(output_value-true_value)))
        avg_vel_diff = np.append(avg_vel_diff, np.mean(i_vpref - np.linalg.norm(xs[:steps_to_goal, 2:4], axis=0)))
        avg_extra_time = np.append(avg_extra_time, np.mean(extra_time))
    
    #print("Average Value Difference from Truth: ", avg_value_diff)
    #print("Average Velocity Difference from Pref: ", avg_vel_diff)
    #print("Average Extra Time from Ideal Path: ", avg_extra_time)
    if visualize:
        pass
        #plot_traj(, goals, radius)
    return avg_value_diff, avg_vel_diff, avg_extra_time
def evaluate_value_fcn(value_fcn, states:dict, goals:dict, radius:dict, 
                        vprefs:dict, dt=0.1, gamma=0.8, visualize=True):

    output_value_dict = dict()
    true_value_dict   = dict()
    extra_time_dict   = dict()
    collision         = False

    avg_value_diff = np.array([])
    avg_vel_diff = np.array([])
    avg_extra_time = np.array([])

    for key in states.keys():

        i_traj   = states[key]
        i_radius = radius[key]
        i_goal   = goals[key]
        i_vpref  = vprefs[key]

        i_dg     = np.linalg.norm(i_traj[0:2,:]-i_goal.reshape(-1,1), axis=0)

        steps_to_goal = np.sum([np.linalg.norm(i_traj[2:4,:], axis=0)>0.05])

        output_value = np.zeros((1,steps_to_goal))
        true_value = np.zeros((1,steps_to_goal))
        extra_time = np.zeros((1,steps_to_goal))

        for key2 in states.keys():
            if key2 is not key:
                j_traj = states[key2]
                j_radius = radius[key2]
                j_goal   = goals[key2]
                j_vpref  = vprefs[key]
                collision = not np.all(np.linalg.norm(i_traj[0:2,:]-j_traj[0:2,:],axis=0)>(i_radius+j_radius))
                # print(np.linalg.norm(i_traj[0:2,:]-j_traj[0:2,:],axis=0))
                if collision:
                    print("Collision between agents {:s} and {:s}".format(key, key2))


        for step in range(steps_to_goal):

            i_state = i_traj[:,step]
            dg      = i_dg[step]
            t       = step*dt
            tg      = (steps_to_goal-step)*dt

            j_state = j_traj[:,step]

            s1 = get_state(i_state[0:4], i_radius, i_goal[0], i_goal[1], i_vpref)
            s2 = get_state(j_state[0:4], j_radius, j_goal[0], j_goal[1], j_vpref)

            s_12 = get_rotated_state(get_joint_state(s1,s2))

            # print(s_12)

            output_value[0,step] = value_fcn(s_12.reshape(1,-1))
            true_value[0,step]   = gamma**(tg*i_vpref)
            extra_time[0,step]   = tg - dg/i_vpref

        if collision == 0:
            output_value_dict[key] = output_value
            true_value_dict[key]   = true_value
            extra_time_dict[key]   = extra_time
        else:
            output_value_dict[key] = None
            true_value_dict[key] = None
            extra_time_dict[key] = None

        avg_value_diff = np.append(avg_value_diff, np.mean(np.abs(output_value-true_value)))
        avg_vel_diff = np.append(avg_vel_diff, np.mean(i_vpref - np.linalg.norm(i_traj[2:4,:steps_to_goal], axis=0)))
        avg_extra_time = np.append(avg_extra_time, np.mean(extra_time))
    
    #print("Average Value Difference from Truth: ", avg_value_diff)
    #print("Average Velocity Difference from Pref: ", avg_vel_diff)
    #print("Average Extra Time from Ideal Path: ", avg_extra_time)
    if visualize:
        plot_traj(states, goals, radius)
    return avg_value_diff, avg_vel_diff, avg_extra_time

    # return collision, output_value_dict, true_value_dict, extra_time_dict

def evaluate(value_fnc, visualize, num_episodes, data_path=FOLDER+"/static_tests.csv"):
    data = read_training_data(data_path)



    ep_list = range(num_episodes)
    robots_count = data.n_agents
    # ep_list = [50]
    avg_val_diffs = np.zeros((0, robots_count))
    avg_vel_diffs = np.zeros((0, robots_count))
    avg_extra_times = np.zeros((0, robots_count))
    for ep in ep_list:

        episode = data.traj[ep]
        s_robo1 = episode.X[0][0,0:4].T
        s_robo2 = episode.X[1][0,0:4].T

        s_initial_1 = get_state(s_robo1, RADIUS, episode.Pg[0][0], episode.Pg[0][1], episode.Vmax[0])
        s_initial_2 = get_state(s_robo2, RADIUS, episode.Pg[1][0], episode.Pg[1][1], episode.Vmax[1])
    

        res = evaluate_value_fcn_propagate(value_fnc, s_initial_1, s_initial_2, visualize=visualize)
        avg_val_diff, avg_vel_diff, avg_extra_time = res

        avg_val_diffs = np.vstack((avg_val_diffs, avg_val_diff))
        avg_vel_diffs = np.vstack((avg_vel_diffs, avg_vel_diff))
        avg_extra_times = np.vstack((avg_extra_times, avg_extra_time))
    print("Val diff:", np.mean(avg_val_diffs, axis=0 ))
    print("Vel diff:", np.mean(avg_vel_diffs, axis=0 ))
    print("Extra time", np.mean(avg_extra_times, axis=0 ))


if __name__=='__main__':
    # path = folder+"/training_data_100sim.csv"
    initial_model_path = FOLDER+"/initial_value_model/"
    post_rl_model_path = FOLDER+"/post_RL_value_model/"
    data_path  = FOLDER+"/test_data.csv"

    initial_value_fnc = tf.keras.models.load_model(initial_model_path)
    post_rl_fnc = tf.keras.models.load_model(post_rl_model_path)

    print("Initial value model evaluation")
    evaluate(value_fnc=initial_value_fnc, visualize=False, num_episodes=100,  data_path=data_path)
    print("Post RL value model evaluation")
    evaluate(value_fnc=post_rl_fnc, visualize=False, num_episodes=100,  data_path=data_path)
    