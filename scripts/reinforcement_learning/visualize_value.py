from read_training_data import read_training_data
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
from model import USER, create_model
import os
from configs import *
from state_definitions import get_rotated_state, get_joint_state, get_state
from itertools import product

data = read_training_data(os.path.join(FOLDER, 'training_data_100sim.csv'))
episode = 19
robo1 = 0
robo2 = 1

traj = data.traj[episode]
Pg1 = traj.Pg[robo1]
Pg2 = traj.Pg[robo2]
X_robo1 = traj.X[robo1]
X_robo2 = traj.X[robo2]
radius1 = traj.R[robo1]
radius2 = traj.R[robo2]
# vpref1  = traj.Vmax[0]
# vpref2  = traj.Vmax[1]

# print(X_robo1)
def animate(i, 
            Pg1,
            Pg2,
            X_robo1,
            X_robo2,
            radius1,
            radius2,
            value_fcn,
            ax1,
            fig):

    FONTSIZE = 10

    if i >=len(X_robo1):
        return

    pos1 = np.array([X_robo1[i][0], X_robo1[i][1]])
    vel1 = np.array([X_robo1[i][2], X_robo1[i][3]])
    pos2 = np.array([X_robo2[i][0], X_robo2[i][1]])
    vel2 = np.array([X_robo2[i][2], X_robo2[i][3]])

    s1_in = np.array([pos1[0], pos1[1], vel1[0], vel1[1]])
    s2_in = np.array([pos2[0], pos2[1], vel2[0], vel2[1]])

    xmax = 5
    xmin = -xmax
    ymax = 3
    ymin = -ymax

    step = 0.1

    xrange = np.arange(xmin,xmax,step=step)
    yrange = np.arange(ymin,ymax,step=step)

    xrange_plot = np.arange(xmin,xmax+step,step=step)
    yrange_plot = np.arange(ymin,ymax+step,step=step)

    pos_map = product(xrange, yrange)

    s1 = get_state(s1_in  , radius1, Pg1[0], Pg1[1], 1)
    s2 = get_state(s2_in  , radius2, Pg2[0], Pg2[1], 1)
    output_value = np.array(list(map(lambda x: map_fcn(x, value_fcn, s1, s2), pos_map)))

    output_value = output_value[:,:,0]
    zmax = output_value.max()
    zmin = 0

    output_value = output_value.reshape(len(xrange),len(yrange))
    
    robo1 = plt.Circle( pos1,
                        radius1,
                        fill = False,
                        color = 'b',
                        label = 'Robot 1')

    robo2 = plt.Circle( pos2,
                        radius2,
                        fill = False,
                        color = 'g',
                        label = 'Robot 2')   
    ax1.clear()

    c = ax1.pcolormesh(xrange_plot, yrange_plot, output_value.T, cmap='hot', vmin=zmin, vmax=zmax)
    if i == 1:
        cbar = fig.colorbar(c, ax=ax1, shrink=0.6)
        cbar.ax.tick_params(labelsize=FONTSIZE) 
        cbar.set_label(label='Network Output Value', size=FONTSIZE)
    ax1.tick_params(labelsize=FONTSIZE)

    ax1.scatter(Pg1[0], Pg1[1], color='b', marker='*')
    ax1.scatter(Pg2[0], Pg2[1], color='g', marker='*')

    if i > 1:
        ax1.plot(X_robo1[0:i+1, 0], X_robo1[0:i+1, 1], linestyle='--', color='b')
        ax1.plot(X_robo2[0:i+1, 0], X_robo2[0:i+1, 1], linestyle='--', color='g')

    ax1.set_xlim([-5,5])
    ax1.set_ylim([-2.5, 2.5])
    ax1.set_aspect( 1 )
    ax1.add_artist( robo1 )
    ax1.add_artist( robo2 )

    plt.title( '2D Trajectory' )
    plt.legend()

def map_fcn(x, value_fcn, s1, s2):
    s1[0] = x[0]
    s1[1] = x[1]
    return value_fcn(get_rotated_state(get_joint_state(s1,s2)).reshape(1,-1))

def plot_animation(Pg1, Pg2, X_robo1, X_robo2, radius1, radius2, fig):
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    xlimits = [-5,5]
    ylimits = [-2.5, 2.5]
    ax1.set_xlim(xlimits)
    ax1.set_ylim(ylimits)

    if len(X_robo1) > len(X_robo2):
        x2 = np.zeros((len(X_robo1), 2))
        x2[0:len(X_robo2)] = X_robo2
        x2[len(X_robo2):] = X_robo2[-1]
        x1 = X_robo1
    elif len(X_robo2) > len(X_robo1):
        x1 = np.zeros((len(X_robo2), 2))
        x1[0:len(X_robo1)] = X_robo1
        x1[len(X_robo1):] = X_robo1[-1]
        x2 = X_robo2
    else:
        x1 = X_robo1
        x2 = X_robo2

    anim = FuncAnimation(fig, animate, interval=100, fargs=(Pg1,
            Pg2,
            x1,
            x2,
            radius1,
            radius2,
            value_fcn,
            ax1, fig))
    pth = os.path.join(FOLDER, 'dummy.mp4')
    writervideo = animation.FFMpegWriter(fps=20) 
    anim.save(pth, writer=writervideo)
    plt.show()

if __name__=='__main__':

    model_path = FOLDER+"/post_RL_value_model/"
    value_fcn = create_model()
    value_fcn.load_weights(model_path)

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    
    anim = FuncAnimation(fig, animate, interval=100, fargs=(Pg1,
            Pg2,
            X_robo1,
            X_robo2,
            radius1,
            radius2,
            value_fcn,
            ax1, fig))
    pth = os.path.join(FOLDER, 'dummy.mp4')
    writervideo = animation.FFMpegWriter(fps=20) 
    anim.save(pth, writer=writervideo)


    plt.show()
