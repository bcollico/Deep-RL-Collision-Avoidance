import numpy as np
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
    theta_p = theta-alpha
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

