import numpy as np

def get_joint_state_vectorized(traj_1, traj_2):
    dim = 14
    len_1 = traj_1.shape[0]
    len_2 = traj_2.shape[0]
    x = np.zeros((len_1, dim))
    x[:,:4] = traj_1[:,:4]
    x[:,4] = traj_1[:,4] 
    x[:,5] = traj_1[:,5] 
    x[:,6] = traj_1[:,6] 
    x[:,7] = traj_1[:,7] 
    x[:,8] = traj_1[:,8] 
    if len_1 <= len_2:
        x[:,9:13] = traj_2[:len_1,:4]
        x[:,13] = traj_2[:len_1,4] #radius1
    else:
        x[:len_2,9:13] = traj_2[:,:4]
        x[len_2:,9:13] = traj_2[-1,:4] # repeating last state
        x[:len_2,13] = traj_2[:,4] #radius1
        x[len_2:,13] = traj_2[-1,4] # repeating last state

    return x
def get_joint_state(s_robo1, s_robo2):
    dim = 14
    x = np.zeros((dim))
    x[:4] = s_robo1[:4]
    x[4] = s_robo1[4] 
    x[5] = s_robo1[5] 
    x[6] = s_robo1[6] 
    x[7] = s_robo1[7] 
    x[8] = s_robo1[8] 
    x[9:13] = s_robo2[:4]
    x[13] = s_robo2[4] #radius1
    return x

def get_state(s_robo1, radius, pgx, pgy, v_pref):
    dim = 9
    vx = s_robo1[2]
    vy = s_robo1[3]
    theta = np.arctan2(vy,vx)
    x = np.zeros((dim))
    x[:4] = s_robo1[:4]
    x[4] = radius
    x[5] = pgx 
    x[6] = pgy 
    x[7] = v_pref
    x[8] = theta
    return x

def angle_diff(a, b):
    a = a % (2. * np.pi)
    b = b % (2. * np.pi)
    diff = a - b
    if np.size(diff) == 1:
        if np.abs(a - b) > np.pi:
            sign = 2. * (diff < 0.) - 1.
            diff += sign * 2. * np.pi
    else:
        idx = np.abs(diff) > np.pi
        sign = 2. * (diff[idx] < 0.) - 1.
        diff[idx] += sign * 2. * np.pi
    return diff

def get_rotated_state(x):
    Pg = x[5:7]
    P = x[:2]

    #alpha is angle from original x to new x
    alpha = np.arctan2(Pg[1]-P[1], Pg[0]-P[0])
    R = np.array([
        [np.cos(alpha), np.sin(alpha)],
        [-np.sin(alpha), np.cos(alpha)]
    ])
    v_p = R@x[2:4]
    theta = x[8]
    theta_p = angle_diff(theta, alpha)
    # if not np.linalg.norm(theta - alpha - theta_p) < 1e-4:
    #     import pdb;pdb.set_trace()
    rot = np.zeros(shape=(x.shape[0]+1))
    rot[0] = np.linalg.norm(x[:2]-x[5:7], 2) #d_g
    rot[1] = x[7] #v_pref
    rot[2:4] = v_p #v_prime
    rot[4] = x[4] # radius
    rot[5] = theta_p #theta_prime
    rot[6:8] = R@x[11:13] #v_tilde_prime ###OBS Uncertain###
    rot[8:10] = R@(x[9:11]-x[:2]) #p_tilde_prime
    rot[10] = x[13] #r_tilde
    rot[11] = x[4] + x[13] #r +r_tilde
    rot[12] = np.cos(theta_p)
    rot[13] = np.sin(theta_p)
    rot[14] = np.linalg.norm(x[:2]-x[9:11], 2)
    return rot

