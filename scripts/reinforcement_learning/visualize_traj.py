


from read_training_data import read_training_data
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import numpy as np
from model import USER
import os

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
limits = [-5,5]
ax1.set_xlim(limits)
ax1.set_ylim(limits)
if USER == 'Brian':
	folder = '/home/bdobkowski/Stanford/AA277/aa277_project/data'
elif USER == 'Torstein':
	folder  = '/home/torstein/Stanford/aa277/aa277_project/data'
else:
	raise Exception('Need to set user folder')
data = read_training_data(os.path.join(folder, 'training_data_2sim_example.csv'))
episode = 0
robo1 = 0
robo2 = 1

traj = data.traj[episode]
Pg1 = traj.Pg[robo1]
Pg2 = traj.Pg[robo2]
X_robo1 = traj.X[robo1]
X_robo2 = traj.X[robo2]
radius = 0.1

print(X_robo1)
def animate(i):
    if i >=len(X_robo1):
        return
    pos1 = np.array([X_robo1[i][0], X_robo1[i][1]])
    robo1 = plt.Circle( (pos1[0], pos1[1]),
                                      radius,
                                      fill = False )
    robo2 = plt.Circle( (X_robo2[i][0], X_robo2[i][1]),
                                      radius,
                                      fill = False )   
    ax1.clear()

    ax1.scatter(Pg1[0], Pg1[1])
    ax1.scatter(Pg2[0], Pg2[1])

    ax1.set_xlim(limits)
    ax1.set_ylim(limits)
    ax1.set_aspect( 1 )
    ax1.add_artist( robo1 )
    ax1.add_artist( robo2 )

    plt.title( 'Robots' )

if __name__=='__main__':
    
    


    anim = FuncAnimation(fig, animate,
                                interval=100)
    pth = os.path.join(folder, 'dummy.mp4')
    # f = r"/home/torstein/Stanford/aa277/aa277_project/data/dummy.mp4" 
    writervideo = animation.FFMpegWriter(fps=20) 
    anim.save(pth, writer=writervideo)


    plt.show()