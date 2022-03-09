import numpy as np
from numpy.random import uniform
from read_training_data import read_training_data
from plot_traj import plot_traj
from model import create_model
from state_definitions import get_state, get_joint_state, get_rotated_state
from itertools import product
import matplotlib.pyplot as plt
from matplotlib import rc

USER = 'Bradley'
GAMMA = 0.8
VARIABLE='Robot_1'

if USER == 'Brian':
	folder  = '/home/bdobkowski/Stanford/AA277/aa277_project/data'
elif USER == 'Torstein':
	folder  = '/home/torstein/Stanford/aa277/aa277_project/data'
elif USER == 'Bradley':
    folder  = '/home/bcollico/github/aa277_project/data'
else:
	raise Exception('Need to set user folder')

def value_heatmap(value_fcn, s1_in=None, goal=None, save=False, idx=''):

    if s1_in is None:
        s1_in = np.array([-3,0,1,0])

    if goal is None:
        goal = np.array([1.5,0])

    if save:
        FONTSIZE = 10
    else:
        FONTSIZE = 16

    vpref = np.array([1,1])
    radius = np.array([1,1])

    xmax = 5
    xmin = -xmax
    ymax = 3
    ymin = -ymax

    step = 0.25

    xrange = np.arange(xmin,xmax,step=step)
    yrange = np.arange(ymin,ymax,step=step)

    xrange_plot = np.arange(xmin,xmax+step,step=step)
    yrange_plot = np.arange(ymin,ymax+step,step=step)

    pos_map = product(xrange, yrange)

    if VARIABLE=='Robot_2':
        s2_in = np.zeros(4)
        s1 = get_state(s1_in  , radius[0], goal[0], goal[1], vpref[0])
        s2 = get_state(s2_in  , radius[1], None   , None   , vpref[1])
        output_value = np.array(list(map(lambda x: map_fcn(x, value_fcn, s1, s2), pos_map)))
        other = 'Robot_1'
    elif VARIABLE=='Robot_1':
        s1 = get_state(s1_in  , radius[0], goal[0], goal[1], vpref[0])
        s2 = get_state(s1_in  , radius[1], None   , None   , vpref[1])
        output_value = np.array(list(map(lambda x: map_fcn(x, value_fcn, s1, s2), pos_map)))
        other = 'Robot_2'
    # distance     = np.array(list(map(lambda x: np.linalg.norm([x[0][0]-x[1][0], x[0][1]-x[1][1]]), pos_map)))
    # rel_angle    = np.array(list(map(lambda x: np.arctan2(x[0][0]-x[1][0], x[0][1]-x[1][1]), pos_map)))
    # angle_to_goal= np.array(list(map(lambda x: np.arctan2(goal[0][0]-x[1][0], x[0][1]-x[1][1]), pos_map)))

    n = np.sqrt(len(output_value))
    # print(np.shape(output_value))
    output_value = output_value[:,:,0]
    # output_value[output_value<=0] = 1e-8
    # output_value = np.log(output_value) / np.log(GAMMA)
    zmax = output_value.max()
    zmin = 0# output_value.min()
    output_value = output_value.reshape(len(xrange),len(yrange))

    fig, ax = plt.subplots()

    x,y = np.meshgrid(xrange_plot, yrange_plot)
    
    c = ax.pcolormesh(xrange_plot, yrange_plot, output_value.T, cmap='hot', vmin=zmin, vmax=zmax)
    # c = ax.pcolormesh(output_value, cmap='hot', vmin=zmin, vmax=zmax)
    cbar = fig.colorbar(c, ax=ax, shrink=0.6)
    cbar.ax.tick_params(labelsize=FONTSIZE) 
    cbar.set_label(label='Network Output Value', size=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE) 

    theta = np.linspace(0, 2*np.pi, num=100, endpoint=True)
    circ_points = np.vstack((np.cos(theta), np.sin(theta)))

    ax.plot(radius[0]*circ_points[0,:]+s1_in[0],
            radius[0]*circ_points[1,:]+s1_in[1],
            color='k')

    ax.plot(s1_in[0],
            s1_in[1],
            color='k',
            marker='.',
            label=other)

    ax.plot(goal[0],
            goal[1],
            color='k',
            linestyle='none',
            marker='*',
            markersize=12,
            label='goal')

    ax.quiver(s1_in[0],
            s1_in[1],
            s1_in[2], 
            s1_in[3],
            color='k',
            angles='xy', 
            scale_units='xy', 
            scale=1)


    # rc('text',usetex=True)
    ax.set_aspect(1)
    ax.set_title('Value Network Output', fontsize=FONTSIZE)
    ax.set_xlabel(VARIABLE+' X Position', fontsize=FONTSIZE)
    ax.set_ylabel(VARIABLE+' Y Position', fontsize=FONTSIZE)

    ax.axis([xmin, xmax, ymin, ymax])

    plt.legend(prop={"size":FONTSIZE})
    
    if not save:
        plt.show()
    elif save:
        plt.savefig(folder+"/heatmap_"+idx+'_'+VARIABLE+".png")

def map_fcn(x, value_fcn, s1, s2):

    if VARIABLE == 'Robot_2':
        s2[0] = x[0]
        s2[1] = x[1]
    elif VARIABLE == 'Robot_1':
        s1[0] = x[0]
        s1[1] = x[1]
        theta = np.arctan2(s1[6]-s1[1], s1[5]-s1[0])
        s1[2] = np.cos(theta)
        s1[3] = np.sin(theta)
        s1[8] = theta

    # print(get_joint_state(s1,s2))
    return value_fcn(get_rotated_state(get_joint_state(s1,s2)).reshape(1,-1))


if __name__=='__main__':
    model_path = folder+"/initial_value_model/"

    c45 = np.cos(np.pi/4)
    s45 = np.sin(np.pi/4)

    ego_state_list = [np.array([-3, 0, 1,  0]), # horizontal
                      np.array([ 0, 2, 0, -1]), # vertical
                      #np.array([ 1.5, 1.5, -c45, -s45]), #q1
                      #np.array([-1.5, 1.5,  c45, -s45]), #q2
                      #np.array([-1.5,-1.5,  c45,  s45]), #q3
                      #np.array([ 1.5,-1.5, -c45,  s45])  #q4
                      ]

    goal_list = [np.array([ 1.5, 0]),
                 np.array([   0, -1.5]),
                #  np.array([-1.5*c45, -1.5*s45]),
                #  np.array([ 1.5*c45, -1.5*s45]),
                #  np.array([ 1.5*c45,  1.5*s45]),
                #  np.array([-1.5*c45,  1.5*s45]),
                # np.array([ 1.5, 0]),
                # np.array([ 1.5, 0]),
                # np.array([ 1.5, 0]),
                # np.array([ 1.5, 0])
                 ]

    # n_runs = 10
    # ego_state_list = []
    # goal_list = []
    # for _ in range(n_runs):
    #     x   = uniform(-4,4)
    #     y   = uniform(-2,2)
    #     vel = uniform(-1,1,2)
    #     vel = vel/np.linalg.norm(vel)

    #     pgx  = uniform(-4,4)
    #     pgy   = uniform(-2,2)

    #     ego_state_list.append(np.array([x, y, vel[0], vel[1]]))
    #     goal_list.append(np.array([pgx, pgy]))

    variable_list = ['Robot_2']

    value_fcn = create_model()
    value_fcn.load_weights(model_path)

    for VARIABLE in variable_list:
        for i in range(len(ego_state_list)):
            value_heatmap(value_fcn, ego_state_list[i], goal_list[i], save=True, idx=str(i))

