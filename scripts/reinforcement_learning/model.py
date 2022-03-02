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
from utils import JointState
#import tf.keras as keras
# Local Application Modules
# -----------------------------------------------------------------------------


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

def load_traj_generate_data(folder):
 
    output_shape = (1,)
    #x, y = generate_random_training_data(input_shape, output_shape, N)
    data = read_training_data("/home/torstein/Stanford/aa277/aa277_project/data/training_data.csv")
    
    robo1 = 0
    robo2 = 1

    episodes_count = len(data.traj)
    dt = data.dt

    radius = 0.1

    gamma = 0.999

    xs = []
    ys = []

    #generating for robot1 first: 
    #after, we  can do the same for robot2
    for ep in range(episodes_count):
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
            tg = (N-i)*dt
            y = gamma**(tg*v_pref)
            xs.append(rotated_state)
            ys.append(y)


    xs, ys = np.array(xs), np.array(ys)

    with open(f"{folder}/xs.npy", 'wb') as f:
        np.save(f, xs)
    with open(f"{folder}/ys.npy", 'wb') as f:
        np.save(f, ys)


def get_training_data(folder):


    with open(f"{folder}/xs.npy", 'rb') as f:
        xs = np.load(f, allow_pickle=True)
    with open(f"{folder}/ys.npy", 'rb') as f:
        ys = np.load(f, allow_pickle=True)

    return xs, ys


if __name__=='__main__': 
    data_folder = "/home/torstein/Stanford/aa277/aa277_project/data"
    #load_traj_generate_data(folder)
    xs, ys = get_training_data(folder=data_folder)
    model = create_model(xs[0].shape[0], )

    split = 2*len(ys)//3
    x_train = xs[:split]
    y_train = ys[:split]

    x_test = xs[split:]
    y_test = ys[split:]
    model = train_model(model, x_train, y_train, epochs=10)
    
    results = model.evaluate(x_test, y_test, batch_size=128)
    print(results)

    model.save(f"{data_folder}/model/trained_1000")