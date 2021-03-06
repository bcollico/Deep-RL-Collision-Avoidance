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
import os
import json
import math
from tqdm.keras import TqdmCallback

# Local Application Modules
# -----------------------------------------------------------------------------
from nn_utils import load_nn_data, load_traj_data, get_nn_input
from configs import *

class Clip(tf.keras.layers.Layer):
  def __init__(self):
    super(Clip, self).__init__()

  def call(self, inputs):
    return tf.clip_by_value(inputs, clip_value_min=0.0, clip_value_max=0.99)

def state_value_pair(s_jn, t_g, v_pref, gamma):
    y = gamma**(t_g*v_pref)
    joint = np.append(s_jn, y)
    return joint

def create_model(input_shape=15):
    output_shape = 1
    
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Dense(150, activation = "relu", name="layer1"), 
        tf.keras.layers.Dense(100, activation = "relu", name="layer2"), 
        tf.keras.layers.Dense(100, activation = "relu", name="layer3"), 
        tf.keras.layers.Dense(output_shape, name="final_layer", 
                kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-8, l2=1e-7))
    ])
    model.summary()

    return model

def train_model(model, x, y, epochs = 250, verbose=1):

    optimizer = tf.keras.optimizers.SGD(learning_rate=LR, momentum=MOMENTUM)
    loss = tf.keras.losses.MeanSquaredError()

    def scheduler(epoch):
        init = LR
        gamma = 0.1
        return init* math.pow(gamma,  
           math.floor((1+epoch)/STEP_SIZE))
    lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)


    model.compile(optimizer = optimizer, loss = loss,  
            steps_per_execution=10)
    model.fit(x, y, epochs = epochs, batch_size=BATCH_SIZE,
    callbacks=[lr_scheduler, TqdmCallback(verbose=verbose)], verbose=0)

    return model



def nn_training(model_folder, epochs=1000, x_dict=None, y_dict=None, retrain=True):

    if not retrain:
        model = tf.keras.models.load_model(os.path.join(model_folder, 'initial_value_model'))
    else: 
        model = create_model()

    x_train, y_train, x_test, y_test = load_nn_data(x_dict, y_dict)
    
    model = train_model(model, x_train, y_train, epochs=epochs)
    #results = model.evaluate(x_test, y_test, batch_size=128)
    #print(results)

    model.save(os.path.join(model_folder, 'initial_value_model'))

def backprop(model, xs, ys, epochs, verbose=1):
    train_model(model, xs, ys, epochs=epochs,verbose=verbose)

if __name__ == '__main__':
    x_dict, v_pref, dt = load_traj_data(FOLDER)
    x_dict_rotated, y_dict = get_nn_input(x_dict, dt, v_pref)
    print("Test data loaded")
    nn_training(FOLDER, x_dict=x_dict_rotated, y_dict=y_dict, epochs=1000)

