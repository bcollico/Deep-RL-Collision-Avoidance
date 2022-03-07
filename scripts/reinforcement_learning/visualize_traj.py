from read_training_data import read_training_data
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
from model import USER
import os
from configs import *

data = read_training_data(os.path.join(FOLDER, 'training_data_1000sim.csv'))
episode = 199
robo1 = 0
robo2 = 1

traj = data.traj[episode]
Pg1 = traj.Pg[robo1]
Pg2 = traj.Pg[robo2]
X_robo1 = traj.X[robo1]
X_robo2 = traj.X[robo2]
radius1 = traj.R[robo1]
radius2 = traj.R[robo2]

# print(X_robo1)
def animate(i, 
            Pg1,
            Pg2,
            X_robo1,
            X_robo2,
            radius1,
            radius2,
            ax1):

    if i >=len(X_robo1):
        return
    pos1 = np.array([X_robo1[i][0], X_robo1[i][1]])
    pos2 = np.array([X_robo2[i][0], X_robo2[i][1]])
    
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

def plot_animation(Pg1, Pg2, X_robo1, X_robo2, radius1, radius2):
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
            ax1))
    pth = os.path.join(FOLDER, 'dummy.mp4')
    writervideo = animation.FFMpegWriter(fps=20) 
    anim.save(pth, writer=writervideo)
    plt.show()

if __name__=='__main__':

    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    
    anim = FuncAnimation(fig, animate, interval=100, fargs=(Pg1,
            Pg2,
            X_robo1,
            X_robo2,
            radius1,
            radius2,
            ax1))
    pth = os.path.join(FOLDER, 'dummy.mp4')
    writervideo = animation.FFMpegWriter(fps=20) 
    anim.save(pth, writer=writervideo)


    plt.show()
