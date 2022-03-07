import numpy as np
from read_training_data import read_training_data
from plot_traj import plot_traj
from model import create_model
from state_definitions import get_state, get_joint_state, get_rotated_state

USER = 'Bradley'

if USER == 'Brian':
	folder  = '/home/bdobkowski/Stanford/AA277/aa277_project/data'
elif USER == 'Torstein':
	folder  = '/home/torstein/Stanford/aa277/aa277_project/data'
elif USER == 'Bradley':
    folder  = '/home/bcollico/github/aa277_project/data'
else:
	raise Exception('Need to set user folder')

def evaluate_value_fcn(value_fcn, states:dict, goals:dict, radius:dict, 
                        vprefs:dict, dt=0.1, gamma=0.8):

    output_value_dict = dict()
    true_value_dict   = dict()
    extra_time_dict   = dict()
    collision         = False

    avg_value_diff = np.array([])
    avg_vel_diff = np.array([])
    avg_extra_time = np.array([])

    for key in states.keys():

        i_traj   = states[key]
        i_radius = radius[key]
        i_goal   = goals[key]
        i_vpref  = vprefs[key]

        i_dg     = np.linalg.norm(i_traj[0:2,:]-i_goal.reshape(-1,1), axis=0)

        steps_to_goal = np.sum([np.linalg.norm(i_traj[2:4,:], axis=0)>0.05])

        output_value = np.zeros((1,steps_to_goal))
        true_value = np.zeros((1,steps_to_goal))
        extra_time = np.zeros((1,steps_to_goal))

        for key2 in states.keys():
            if key2 is not key:
                j_traj = states[key2]
                j_radius = radius[key2]
                j_goal   = goals[key2]
                j_vpref  = vprefs[key]
                collision = not np.all(np.linalg.norm(i_traj[0:2,:]-j_traj[0:2,:],axis=0)>(i_radius+j_radius))
                # print(np.linalg.norm(i_traj[0:2,:]-j_traj[0:2,:],axis=0))
                if collision:
                    print("Collision between agents {:s} and {:s}".format(key, key2))


        for step in range(steps_to_goal):

            i_state = i_traj[:,step]
            dg      = i_dg[step]
            t       = step*dt
            tg      = (steps_to_goal-step)*dt

            j_state = j_traj[:,step]

            s1 = get_state(i_state[0:4], i_radius, i_goal[0], i_goal[1], i_vpref)
            s2 = get_state(j_state[0:4], j_radius, j_goal[0], j_goal[1], j_vpref)

            s_12 = get_rotated_state(get_joint_state(s1,s2))

            # print(s_12)

            output_value[0,step] = value_fcn(s_12.reshape(1,-1))
            true_value[0,step]   = gamma**(tg*i_vpref)
            extra_time[0,step]   = tg - dg/i_vpref

        if collision == 0:
            output_value_dict[key] = output_value
            true_value_dict[key]   = true_value
            extra_time_dict[key]   = extra_time
        else:
            output_value_dict[key] = None
            true_value_dict[key] = None
            extra_time_dict[key] = None

        avg_value_diff = np.append(avg_value_diff, np.mean(np.abs(output_value-true_value)))
        avg_vel_diff = np.append(avg_vel_diff, np.mean(i_vpref - np.linalg.norm(i_traj[2:4,:steps_to_goal], axis=0)))
        avg_extra_time = np.append(avg_extra_time, np.mean(extra_time))
    
    print("Average Value Difference from Truth: ", avg_value_diff)
    print("Average Velocity Difference from Pref: ", avg_vel_diff)
    print("Average Extra Time from Ideal Path: ", avg_extra_time)

    plot_traj(states, goals, radius)
    
    # return collision, output_value_dict, true_value_dict, extra_time_dict


if __name__=='__main__':
    # path = folder+"/training_data_100sim.csv"
    model_path = folder+"/initial_value_model/"
    data_path  = folder+"/static_tests.csv"
    data = read_training_data(data_path)

    value_fcn = create_model()
    value_fcn.load_weights(model_path)

    # ep_list = np.int64(np.linspace(1,999,10))
    ep_list = range(2)
    # ep_list = [50]
    for ep in ep_list:
        states = dict()
        goals  = dict()
        radius = dict()
        vprefs = dict()

        episode = data.traj[ep]
        for i in range(data.n_agents):
            states[str(i)] = episode.X[i][:,0:4].T
            goals[str(i)]  = episode.Pg[i]
            radius[str(i)] = episode.R[i]
            vprefs[str(i)] = episode.Vmax[i]

        evaluate_value_fcn(value_fcn, states, goals, radius, vprefs, dt=data.dt)

