import numpy as np
from read_training_data import read_training_data
from plot_traj import plot_traj
from configs import *
from state_definitions import get_state, get_joint_state, get_rotated_state
import tensorflow as tf
import os
from cadrl import CADRL
from rl_utils import get_goal, get_radius
from cadrl import plot_animation
def evaluate_value_fcn_propagate(value_fnc, s_initial_1, s_initial_2, visualize, dt=DT, animate=False):
    
    xs1, xs2, cadrl_successful, Rs1, Rs2, x1s_rot, x2s_rot, collision = CADRL(value_fnc, s_initial_1, s_initial_2, 0.0, test=True)
    
   
    if animate:
        plot_animation(get_goal(xs1),
                    get_goal(xs2),
                    xs1[:, 0:2],
                    xs2[:, 0:2],
                    get_radius(xs1),
                    get_radius(xs2))
    if not cadrl_successful : 
        return np.zeros(2), np.zeros(2), np.zeros(2), cadrl_successful, collision
    goals = [xs1[0, 5:7],  xs2[0, 5:7]]

    gamma = GAMMA
    rotated_states = [x1s_rot, x2s_rot]
    avg_value_diff = np.array([])
    avg_vel_diff   = np.array([])
    avg_extra_time = np.array([])
    for i, xs in enumerate([xs1, xs2]):

        i_radius = RADIUS
        i_goal   = goals[i]
        i_vpref  = xs[0, 7]

        i_dg     = np.linalg.norm(xs[:, 0:2]-i_goal, axis=1)

        steps_to_goal = len(xs)
        output_value = np.zeros((1,steps_to_goal))
        true_value = np.zeros((1,steps_to_goal))
        extra_time = np.zeros((1,steps_to_goal))

        for j, xs_other in enumerate([xs1, xs2]):
            if j is not i:
                j_radius = RADIUS
                min_idx = min(len(xs), len(xs_other))
                collision = not np.all(np.linalg.norm(xs[:min_idx, 0:2]-xs_other[:min_idx, 0:2],axis=1)>(i_radius+j_radius))
                # print(np.linalg.norm(i_traj[0:2,:]-j_traj[0:2,:],axis=0))
                if collision:
                    print(f"Collision between agents {i} and {j}")


        for step in range(steps_to_goal):

            dg      = i_dg[step]
            t       = step*dt
            tg      = (steps_to_goal-step+1)*dt

            s_12 = rotated_states[i][step]

            output_value[0,step] = value_fnc(np.array([s_12]))
            true_value[0,step]   = gamma**(tg*i_vpref)
            extra_time[0,step]   = tg - dg/i_vpref
            try:
                assert(extra_time[0, step] >=0), "negative extra time"
            except:
                return

        avg_value_diff = np.append(avg_value_diff, np.mean(np.abs(output_value-true_value)))
        avg_vel_diff = np.append(avg_vel_diff, np.mean(i_vpref - np.linalg.norm(xs[:steps_to_goal, 2:4], axis=1)))
        avg_extra_time = np.append(avg_extra_time, np.mean(extra_time))
    
    #print("Average Value Difference from Truth: ", avg_value_diff)
    #print("Average Velocity Difference from Pref: ", avg_vel_diff)
    #print("Average Extra Time from Ideal Path: ", avg_extra_time)
    if visualize:
        plot_traj(xs1, xs2, dt=dt)
    return avg_value_diff, avg_vel_diff, avg_extra_time, cadrl_successful, collision

def evaluate(value_fnc, visualize, num_episodes, data=None, data_path=FOLDER+"/test_data.csv", animate=False):
    if data is None:
        data = read_training_data(data_path)
    
    ep_list = range(num_episodes)
    robots_count = data.n_agents
    dt = DT
    # ep_list = [50]
    avg_val_diffs = np.zeros((0, robots_count))
    avg_vel_diffs = np.zeros((0, robots_count))
    avg_extra_times = np.zeros((0, robots_count))

    successes  = 0
    collisions = 0
    failures   = 0

    for ep in ep_list[:num_episodes]:

        episode = data.traj[ep]
        s_robo1 = episode.X[0][0,0:4].T
        s_robo2 = episode.X[1][0,0:4].T

        s_initial_1 = get_state(s_robo1, RADIUS, episode.Pg[0][0], episode.Pg[0][1], episode.Vmax[0])
        s_initial_2 = get_state(s_robo2, RADIUS, episode.Pg[1][0], episode.Pg[1][1], episode.Vmax[1])
    

        res = evaluate_value_fcn_propagate(value_fnc, s_initial_1, s_initial_2, visualize=visualize, animate=animate)
        avg_val_diff, avg_vel_diff, avg_extra_time, success, collision = res

        if success:
            successes+=1
        
            avg_val_diffs = np.vstack((avg_val_diffs, avg_val_diff))
            avg_vel_diffs = np.vstack((avg_vel_diffs, avg_vel_diff))
            avg_extra_times = np.vstack((avg_extra_times, avg_extra_time))
        elif collision:
            collisions+=1
        else:
            failures+=1
    mean_val =  np.mean(avg_val_diffs, axis=0 )
    mean_vel = np.mean(avg_vel_diffs, axis=0 )
    mean_extra_time = np.mean(np.abs(avg_extra_times[:, 0]-avg_extra_times[:, 1]), axis=0 )
    print(f"Successes: {successes}, Collisions: {collisions}, failures: {failures}")
    print("Val diff:", mean_val)
    print("Vel diff:", mean_vel)
    print("Extra time", mean_extra_time)
    return mean_val, mean_vel, mean_extra_time, successes, collisions, failures


def pass_evaluation(res_new, res_old):
    mean_val_new, mean_vel_new, mean_extra_time_new = res_new
    mean_val_old, mean_vel_old, mean_extra_time_old = res_old

    return np.all(mean_val_new<=mean_val_old)
    

if __name__=='__main__':
    # path = folder+"/training_data_100sim.csv"
    initial_model_path = FOLDER+"/initial_value_model/"
    post_rl_model_path = FOLDER+"/post_RL_value_model/"
    data_path  = FOLDER+"/test_data.csv"

    data = read_training_data(data_path)

    initial_value_fnc = tf.keras.models.load_model(initial_model_path)
    post_rl_fnc = tf.keras.models.load_model(post_rl_model_path)

    # print("Initial value model evaluation")
    evaluate(value_fnc=initial_value_fnc, visualize=False, num_episodes=100, data=data, data_path=None, animate=False)
    print("Post RL value model evaluation")
    evaluate(value_fnc=post_rl_fnc, visualize=False, num_episodes=100,  data=data, data_path=None, animate=False)
    