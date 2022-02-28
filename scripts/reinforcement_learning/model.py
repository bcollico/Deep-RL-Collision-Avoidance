"""
File name: model.py

Creation Date: Wed 09 Feb 2022

Description:

"""

# Standard Python Libraries
# -----------------------------------------------------------------------------
from asyncore import read
from gettext import pgettext
import numpy as np
from sklearn.neighbors import radius_neighbors_graph
import tensorflow as tf
from read_training_data import read_training_data
import os
import json
# from utils import JointState
#import tf.keras as keras
# Local Application Modules
# -----------------------------------------------------------------------------

USER = 'Brian'
# USER = 'Torstein'
# USER = 'Valentin'
# USER = 'Bradley'

def state_value_pair(s_jn, t_g, v_pref, gamma):
    y = gamma**(t_g*v_pref)
    joint = np.append(s_jn, y)
    return joint




def create_model(input_shape=15):
    output_shape = 1
    hidden_neurons = 30
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(hidden_neurons, activation = "relu", name="layer1"), 
        tf.keras.layers.Dense(hidden_neurons, activation = "relu", name="layer2"), 
        tf.keras.layers.Dense(output_shape, name="final_layer", 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-7))

    ])
    model.summary()

    return model

def train_model(model, x, y, epochs = 1000):

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    loss = tf.keras.losses.MeanSquaredError()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="loss", patience=30)]


    model.compile(optimizer = optimizer, loss = loss,  
            steps_per_execution=10)
    model.fit(x, y, epochs = epochs, callbacks = callbacks)

    return model

def generate_random_training_data(input_shape, output_shape, N):
    inp = (N, ) + input_shape
    out = (N, ) + output_shape
    x = np.random.randn(*inp)
    y = np.sum(x, axis=-1)
    print(x, y)
    return x, y


def get_joint_state(s_robo1, s_robo2, radius, pgx, pgy, v_pref):
    dim = 14
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
    x[9:13] = s_robo2[:4]
    x[13] = radius #radius1
    return x

def get_joint_state2(s_robo1, s_robo2):
    dim = 14
    x = np.zeros((dim))
    x[:4] = s_robo1[:4]
    x[4] = s_robo1[9] 
    x[5] = s_robo1[5] 
    x[6] = s_robo1[6] 
    x[7] = s_robo1[7] 
    x[8] = s_robo1[8] 
    x[9:13] = s_robo2[:4]
    x[13] = s_robo1[9] #radius1
    return x

def get_state(s_robo1, radius, pgx, pgy, v_pref):
    dim = 10
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
    x[9] = radius #radius1
    return x

def get_rotated_state(x):
    Pg = x[5:7]
    P = x[:2]

    #alpha is angle from original x to new x
    alpha = np.arctan2(Pg[1]-Pg[0], P[1]-P[0])
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

def load_traj_generate_data_not_joint(folder):
 
    output_shape = (1,)
    # x, y = generate_random_training_data(input_shape, output_shape, N)
    data = read_training_data(os.path.join(folder, 'training_data_2sim_example.csv'))
    # data = read_training_data("/home/torstein/Stanford/aa277/aa277_project/data/training_data.csv")
    
    robo1 = 0
    robo2 = 1

    episodes_count = len(data.traj)
    dt = data.dt

    radius = 0.1

    gamma = 0.999

    x_ep_dict = {}
    y_ep_dict = {}

    #generating for robot1 first: 
    #after, we  can do the same for robot2
    for ep in range(episodes_count):

        x_robot_dict = {}
        y_robot_dict = {}

        traj = data.traj[ep]
        for robot in range(len(traj.Pg)):
            pgx = traj.Pg[robot][0]
            pgy = traj.Pg[robot][1]
            Vmax = max(traj.Vmax[0], traj.Vmax[1])
            v_pref = Vmax
            N = len(data.traj[ep].X[robot])

            xs = []
            ys = []
            for i in range(N):

                s_robo1 = data.traj[ep].X[robot][i]
                state = get_state(s_robo1, radius, pgx, pgy, v_pref)
                
                tg = i*dt
                y = gamma**(tg*v_pref)
                xs.append(state.tolist())
                ys.append(y.tolist())

            x_robot_dict[robot] = np.array(xs)
            y_robot_dict[robot] = np.array(ys)

        x_ep_dict[ep] = x_robot_dict
        y_ep_dict[ep] = y_robot_dict

    return x_ep_dict, y_ep_dict

    # y_json = json.dumps(y_ep_dict)
    # with open(os.path.join(folder, 'y_dict.json'),"w") as f:
    #     f.write(y_json)

    # x_json = json.dumps(x_ep_dict)
    # with open(os.path.join(folder, 'x_dict.json'),"w") as f:
    #     f.write(x_json)

def load_traj_generate_data(folder):
 
    output_shape = (1,)
    # x, y = generate_random_training_data(input_shape, output_shape, N)
    data = read_training_data(os.path.join(folder, 'training_data_2sim_example.csv'))
    # data = read_training_data("/home/torstein/Stanford/aa277/aa277_project/data/training_data.csv")
    
    robo1 = 0
    robo2 = 1

    episodes_count = len(data.traj)
    dt = data.dt

    radius = 0.1

    gamma = 0.999

    x_dict = {}
    y_dict = {}
    x_dict_rotated = {}

    #generating for robot1 first: 
    #after, we  can do the same for robot2
    for ep in range(episodes_count):

        xs_rotated = []
        xs = []
        ys = []

        traj = data.traj[ep]
        pgx = traj.Pg[robo1][0]
        pgy = traj.Pg[robo1][1]
        Vmax = max(traj.Vmax[0], traj.Vmax[1])
        v_pref = Vmax
        N = len(data.traj[ep].X[robo1])
        for i in range(N):

            s_robo1 = data.traj[ep].X[robo1][i]
            s_robo2 = data.traj[ep].X[robo2][i]
            state = get_joint_state(s_robo1, s_robo2, radius, pgx, pgy, v_pref)
            
            #should rotate state here...
            rotated_state = get_rotated_state(state)
            tg = i*dt
            y = gamma**(tg*v_pref)
            xs_rotated.append(rotated_state.tolist())
            xs.append(state.tolist())
            ys.append(y.tolist())

        x_dict_rotated[ep] = xs_rotated
        x_dict[ep] = xs
        y_dict[ep] = ys

    # all_xs, all_ys = np.array(xs), np.array(ys)

    # with open(f"{folder}/xs.npy", 'wb') as f:
    #     np.save(f, all_xs)
    # with open(f"{folder}/ys.npy", 'wb') as f:
    #     np.save(f, all_ys)

    x_json = json.dumps(x_dict_rotated)
    with open(os.path.join(folder, 'x_dict_rotated.json'),"w") as f:
        f.write(x_json)

    y_json = json.dumps(y_dict)
    with open(os.path.join(folder, 'y_dict.json'),"w") as f:
        f.write(y_json)

    x_json = json.dumps(x_dict)
    with open(os.path.join(folder, 'x_dict.json'),"w") as f:
        f.write(x_json)

def load_training_test_data(folder):
    # with open(f"{folder}/xs.npy", 'rb') as f:
    #     xs = np.load(f, allow_pickle=True)
    # with open(f"{folder}/ys.npy", 'rb') as f:
    #     ys = np.load(f, allow_pickle=True)

    with open(os.path.join(folder, 'x_dict_rotated.json'), 'r') as j:
        x_dict = json.loads(j.read())
    with open(os.path.join(folder, 'y_dict.json'), 'r') as j:
        y_dict = json.loads(j.read())

    xs = np.array([]).reshape((0, len(x_dict['0'][0])))
    ys = np.array([])

    for xk, yk in zip(x_dict.keys(), y_dict.keys()):
        xs = np.vstack([xs, np.asarray(x_dict[xk])])
        ys = np.concatenate((ys, np.asarray(y_dict[yk])))

    split = 2*len(ys)//3

    x_train = xs[:split]
    y_train = ys[:split]

    x_test = xs[split:]
    y_test = ys[split:]

    return x_train, y_train, x_test, y_test


def test_model(folder):

    x_train, y_train, x_test, y_test = load_training_test_data(folder)
    
    input_shape = (x_train[0].shape[0], )
    model = create_model(input_shape)
    
    model = train_model(model, x_train, y_train, epochs=1000)
    results = model.evaluate(x_test, y_test, batch_size=128)
    print(results)

    model.save(folder)

if __name__ == '__main__':
    if USER == 'Torstein':
        folder = "/home/torstein/Stanford/aa277/aa277_project/data"
    elif USER == 'Brian':
        folder  = "/home/bdobkowski/Stanford/AA277/aa277_project/data"
    else:
        raise Exception('Need to list a folder in on your local machine to store data')
    load_traj_generate_data(folder)
    test_model(folder=folder)

