import numpy as np
from read_training_data import read_training_data
from plot_traj import plot_traj
from model import create_model
from state_definitions import get_state, get_joint_state, get_rotated_state
from itertools import product
import matplotlib.pyplot as plt

USER = 'Bradley'

if USER == 'Brian':
	folder  = '/home/bdobkowski/Stanford/AA277/aa277_project/data'
elif USER == 'Torstein':
	folder  = '/home/torstein/Stanford/aa277/aa277_project/data'
elif USER == 'Bradley':
    folder  = '/home/bcollico/github/aa277_project/data'
else:
	raise Exception('Need to set user folder')

def value_heatmap(value_fcn):

    vpref = np.array([1,1])
    radius = np.array([1,1])
    goal = np.array([1.5,-1.5])

    xmax = 5
    xmin = -xmax
    ymax = 3
    ymin = -ymax

    step = 0.25

    xrange = np.arange(xmin,xmax,step=step)
    yrange = np.arange(ymin,ymax,step=step)


    xrange_plot = np.arange(xmin,xmax+step,step=step)
    yrange_plot = np.arange(ymin,ymax+step,step=step)

    ego_x = 2
    ego_y = 2
    ego_vx = 0
    ego_vy = -vpref[0]
    ego_state = np.array([ego_x, ego_y, ego_vx, ego_vy])

    pos_map = product(xrange, yrange)

    s1 = get_state(ego_state  , radius[0], goal[0], goal[1], vpref[0])
    s2 = get_state(np.zeros(4), radius[1], None   , None   , vpref[1])

    output_value = np.array(list(map(lambda x: map_fcn(x, value_fcn, s1, s2), pos_map)))
    # distance     = np.array(list(map(lambda x: np.linalg.norm([x[0][0]-x[1][0], x[0][1]-x[1][1]]), pos_map)))
    # rel_angle    = np.array(list(map(lambda x: np.arctan2(x[0][0]-x[1][0], x[0][1]-x[1][1]), pos_map)))
    # angle_to_goal= np.array(list(map(lambda x: np.arctan2(goal[0][0]-x[1][0], x[0][1]-x[1][1]), pos_map)))

    n = np.sqrt(len(output_value))
    # print(np.shape(output_value))
    zmax = output_value.max()
    zmin = 0 # output_value.min()
    output_value = output_value[:,:,0]
    output_value = output_value.reshape(len(xrange),len(yrange))

    fig, ax = plt.subplots()

    x,y = np.meshgrid(xrange_plot, yrange_plot)
    
    c = ax.pcolormesh(xrange_plot, yrange_plot, output_value.T, cmap='hot', vmin=zmin, vmax=zmax)
    # c = ax.pcolormesh(output_value, cmap='hot', vmin=zmin, vmax=zmax)
    fig.colorbar(c, ax=ax)

    theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
    circ_points = np.vstack((np.cos(theta), np.sin(theta)))

    ax.plot(radius[0]*circ_points[0,:]+ego_x,
            radius[0]*circ_points[1,:]+ego_y,
            color='k')

    ax.plot(ego_x,
            ego_y,
            color='k',
            marker='.',
            label='Robot 1')

    ax.plot(goal[0],
            goal[1],
            color='k',
            linestyle='none',
            marker='*',
            markersize=12,
            label='goal')

    ax.quiver(ego_x,
            ego_y,
            ego_vx, 
            ego_vy,
            color='k',
            angles='xy', 
            scale_units='xy', 
            scale=1)


    # rc('text',usetex=True)
    ax.set_aspect(1)
    ax.set_title('Value Network Output')
    ax.set_xlabel('Robot 2 X Position')
    ax.set_ylabel('Robot 2 Y Position')

    ax.axis([xmin, xmax, ymin, ymax])

    plt.legend()
    plt.show()




def map_fcn(x, value_fcn, s1, s2):

    # print(pair)s

    s2[0] = x[0]
    s2[1] = x[1]

    return value_fcn(get_rotated_state(get_joint_state(s1,s2)).reshape(1,-1))



if __name__=='__main__':
    model_path = folder+"/initial_value_model/"

    value_fcn = create_model()
    value_fcn.load_weights(model_path)

    value_heatmap(value_fcn)

