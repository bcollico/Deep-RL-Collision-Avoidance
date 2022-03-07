import numpy as np

def load_training_test_data(folder):
    with open(os.path.join(folder, 'x_dict.json'), 'r') as j:
        x_dict = json.loads(j.read())
    with open(os.path.join(folder, 'y_dict.json'), 'r') as j:
        y_dict = json.loads(j.read())
    
    for xk, yk in zip(x_dict.keys(), y_dict.keys()):
        x_dict[xk] = np.asarray(x_dict[xk])
        y_dict[yk] = np.asarray(y_dict[yk])

    return x_dict, y_dict

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
    return np.linalg.norm(get_pos(x) - get_goal(x)) < 1e-1

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