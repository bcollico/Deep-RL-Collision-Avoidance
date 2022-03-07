import numpy as np
from configs import *
from state_definitions import get_rotated_state, get_joint_state

# def load_training_test_data(folder):
#     with open(os.path.join(folder, 'x_dict.json'), 'r') as j:
#         x_dict = json.loads(j.read())
#     with open(os.path.join(folder, 'y_dict.json'), 'r') as j:
#         y_dict = json.loads(j.read())
    
#     for xk, yk in zip(x_dict.keys(), y_dict.keys()):
#         x_dict[xk] = np.asarray(x_dict[xk])
#         y_dict[yk] = np.asarray(y_dict[yk])

#     return x_dict, y_dict

def get_goal(x):
    try:
        a, b = x.shape
        return x[-1, 5:7]
    except:
        return x[5:7]

def get_pos(x):
    try:
        a, b = x.shape
        return x[-1, 0:2]
    except:
        return x[0:2]

def get_vel(x):
    try:
        a, b = x.shape
        return x[-1, 2:4]
    except:
        return x[2:4]

def get_vpref(x):
    try:
        a, b = x.shape
        return x[-1, 7]
    except:
        return x[7]

def get_radius(x):
    try:
        a, b = x.shape
        return x[-1, 4]
    except:
        return x[4]

def get_current_state(x):
    return np.copy(x[-1, :])

def close_to_goal(x):
    '''
    condition for exiting the while loop in CADRL
    '''
    tol = 1e-1
    return get_goal_distance(x) < tol
    #return np.linalg.norm(get_pos(x) - get_goal(x)) < 1e-1

def get_goal_distance(x):
    goal = get_goal(x)
    pos = get_pos(x)
    return np.linalg.norm(pos-goal)

def robots_intersect(x1, x2):
    '''
    Do the robots intersect during their trajectory?
    '''
    r1 = get_radius(x1)
    r2 = get_radius(x2)
    min_length = np.minimum(len(x1), len(x2))
    distances = np.linalg.norm(x1[0:min_length,0:2] - x2[0:min_length,0:2], axis=1)
    return np.any(distances<=r1+r2)

def propagate_dynamics(xx, v, dt):
    '''
       Single integrator robot dynamics
    '''

    # TODO: incorporate kinematic constraints?

    x = np.copy(xx)

    try:
        a, b = x.shape
        x[:, 0:2] = x[:, 0:2] + dt* v
        return x
    except:
        x[0:2] = x[0:2] + dt* v
        return x

def find_y_values(V_prime, state_sequence0, state_sequence1, reward_sequence0, gamma):
    """
    Estimate state values of finished episode and update the memory pool
    """
    
    tg0 = len(state_sequence0)
    tg1 = len(state_sequence1)
    xs, ys = [], []
    for step in range(tg0-1):
        steps_left = tg0-step-1
        state0 = state_sequence0[step]
        next_state0 = state_sequence0[step+1]

        reward0 = reward_sequence0[step].item()
        # approximate the value with TD prediction based on the next state
        value = reward0 + gamma * V_prime(np.array([next_state0])).numpy().item()

        # penalize non-cooperating behaviors
        #state1 = state_sequence1[step]
        #if state0 is None:
        #    te0 = 0
        dg0 =  state0[0].item()
        v_pref = state0[1].item()
        te0 = steps_left*DT - dg0/v_pref
        if step>len(state_sequence1)-1:
            te1 = 0
        else:
            state1 = state_sequence1[step]
            dg1 = state1[0].item()
            steps_left1 = dg1-step-1
            te1 = (steps_left1)*DT - dg1/v_pref
        if te0 < 1 and te1 > 6:
            value -= 0.1
        xs.append(state0)
        ys.append(value)
    
    return np.array(xs), np.array(ys)
