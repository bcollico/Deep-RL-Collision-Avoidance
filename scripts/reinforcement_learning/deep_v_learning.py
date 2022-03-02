import numpy as np
import tensorflow as tf
import os
import model
import json

N_EPISODES = 10
M          = 10
GAMMA      = 0.99
VMAX       = 1.0 #??

USER = 'Brian'
# USER = 'Torstein'
# USER = 'Valentin'
# USER = 'Bradley'

if USER == 'Torstein':
    folder = "/home/torstein/Stanford/aa277/aa277_project/data"
elif USER == 'Brian':
    folder  = "/home/bdobkowski/Stanford/AA277/aa277_project/data"
else:
    raise Exception('Need to list a folder in on your local machine to store data')

def load_training_test_data(folder):
    with open(os.path.join(folder, 'x_dict.json'), 'r') as j:
        x_dict = json.loads(j.read())
    with open(os.path.join(folder, 'y_dict.json'), 'r') as j:
        y_dict = json.loads(j.read())
    
    for xk, yk in zip(x_dict.keys(), y_dict.keys()):
        x_dict[xk] = np.asarray(x_dict[xk])
        y_dict[yk] = np.asarray(y_dict[yk])

    return x_dict, y_dict

def propagate_dynamics(x, v, dt):
    '''
       Single integrator robot dynamics
    '''

    # TODO: incorporate kinematic constraints?

    x[0:2] = x[0:2] + dt* v
    # replace x[2:4] with filtered velocity v? what is observed?

    return x

def reward(x1, x2, a, dt):
    '''
    Reward function for robot 1 only, given joint state (parametrized by 2 individual states)
    '''
    d = np.linalg.norm(x1[0:2] - x2[0:2]) # is this how to calculate d? 
    #Doesn't seem so. How can dmin be less than 0 like in the paper?

    eps = 1e-2

    x1_nxt = propagate_dynamics(np.copy(x1), a, dt)

    if d < 0:
        R = -0.25
    elif d<0.2:
        R = -0.1-d/2
    elif np.linalg.norm(x1[5:7] - x1[0:2]) < eps:
        R = 1
    else:
    	R = 0.0

    # experimental - trying to get robot to approach the goal
    # if np.linalg.norm(x1_nxt[0:2] - x1[5:7]) < np.linalg.norm(x1[0:2] - x1[5:7]):
    # 	R += 20.
    # else:
    #     R += -0.02

    return R

def close_to_goal(x):
	'''
	condition for exiting the while loop in CADRL
	'''
	return np.linalg.norm(x[0:2, -1] - x[5:7, -1]) < 1e-1


def CADRL(value_model, initial_state_1, initial_state_2):
    '''
        Algorithm 1: CADRL (Collision Avoidance with Deep RL)

        INCOMPLETE

        Doesn't work yet. Also haven't implemented epsilon-greedy exploration yet
    '''
    dt = 0.1 # uncertain
    t  = 0

    # robot trajectories will be 10xT, where T is the total timesteps in the traj
    x_1 = initial_state_1.reshape(-1, 1) # robot 1 state px, py, vx, vy, ...
    x_2 = initial_state_2.reshape(-1, 1) # robot 2 state px, py, vx, vy, ...

    # x[0:2] - position
    # x[2:4] - velocity
    # x[5:7] - goal position

    # while distance between robots and there goals is greater than eps
    while not close_to_goal(x_1) and not close_to_goal(x_2):
        t += dt

        v_filtered_1 = np.ma.average(x_1[2:4, :], axis=1, weights = np.exp(range(len(x_1[0])))) # weights the more recent scores more
        v_filtered_2 = np.ma.average(x_2[2:4, :], axis=1, weights = np.exp(range(len(x_2[0])))) # weights the more recent scores more

        # Question: how do we set the velocities for the ~ robot in CADRL?
        # My feeling is we do both robots in CADRL simultaneously
        # To do this, I think we'd need to save and load states for individual robots 
        # rather than loading joint states/trajectories
        x1_o_nxt = propagate_dynamics(x_1[:,-1], v_filtered_1, dt)
        x2_o_nxt = propagate_dynamics(x_2[:,-1], v_filtered_2, dt)

        A = np.random.uniform(low=-VMAX, high=VMAX, size=(100, 2)) # TODO: how should we sample actions?
        # what should the size of these sampled actions be?

        gamma_bar_x1 = GAMMA**dt*x_1[7, -1]
        gamma_bar_x2 = GAMMA**dt*x_2[7, -1]

        x1_nxt = np.zeros((len(A), x_1.shape[0]))
        x2_nxt = np.zeros((len(A), x_2.shape[0]))

        bellman_1 = np.zeros(len(A))
        bellman_2 = np.zeros(len(A))

        for idx, a in enumerate(A):
            x1_nxt[idx] = propagate_dynamics(x_1[:, -1], a, dt)
            x2_nxt[idx] = propagate_dynamics(x_2[:, -1], a, dt)

            x1_joint = model.get_joint_state2(x1_nxt[idx], x2_o_nxt)
            x2_joint = model.get_joint_state2(x2_nxt[idx], x1_o_nxt)

            x1_joint_rotated = model.get_rotated_state(x1_joint)
            x2_joint_rotated = model.get_rotated_state(x2_joint)

            # does our action have no impact on reward here? should we feed it the propagated state?
            R1 = reward(x_1[:, -1], x_2[:, -1], a, dt)
            R2 = reward(x_2[:, -1], x_1[:, -1], a, dt)

            bellman_1[idx] = R1 + gamma_bar_x1 * value_model(x1_joint_rotated.reshape(1,-1))
            bellman_2[idx] = R2 + gamma_bar_x2 * value_model(x2_joint_rotated.reshape(1,-1))

        opt_action_1 = A[np.argmax(bellman_1)]
        opt_action_2 = A[np.argmax(bellman_2)]

        if not close_to_goal(x_1):        
	        x_1 = np.append(x_1, propagate_dynamics(np.copy(x_1[:, -1]), opt_action_1, dt).reshape(-1, 1), axis=1)
        if not close_to_goal(x_2):
	        x_2 = np.append(x_2, propagate_dynamics(np.copy(x_2[:, -1]), opt_action_2, dt).reshape(-1, 1), axis=1)

        print(np.linalg.norm(x_1[0:2, -1] - x_1[5:7, -1]))
        # print(x_1[5:7, -1])
        # print(x_1[0:2, -1])

        # print(t)

    return x_1, x_2


if __name__ == '__main__':

    # algorithm 2 line 4
    value_model = tf.keras.models.load_model(folder)

    # algorithm 2 line 5
    x_dict, y_dict = load_training_test_data(folder)
    x_ep_dict, y_ep_dict = model.load_traj_generate_data_not_joint(folder)

    # x_ep_dict is a dictionary with following pattern:
    # x_ep_dict[1] is all robot traj data for episode 1
    # x_ep_dict[1][2] is robot 2 traj data for episode 1
    # x_ep_dict[1][2][3] is timestep 3 for robot 2 in episode 1
    # x_ep_dict[1][2][3] is 10-dimensional, and contains the state from get_state() in model.py

    # y_ep_dict is constructed the same way

    for ep in range(N_EPISODES):
        for m in range(M):
            
            # algorithm 2 line 8: random test case (by index)
            rand_idx_1 = np.random.randint(0, high=len(x_ep_dict[ep].keys()))
            rand_idx_2 = np.random.randint(0, high=len(x_ep_dict[ep].keys()))
            while rand_idx_2 == rand_idx_1:
                rand_idx_2 = np.random.randint(0, high=len(x_ep_dict[ep].keys()))

            # algorithm 2 line 9
            s_1, s_2 = CADRL(value_model, x_ep_dict[ep][rand_idx_1][0], x_ep_dict[ep][rand_idx_2][0])

    import pdb;pdb.set_trace()