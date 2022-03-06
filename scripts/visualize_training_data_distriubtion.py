from reinforcement_learning import read_training_data
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np

USER = 'Bradley'

if USER == 'Brian':
	folder  = '/home/bdobkowski/Stanford/AA277/aa277_project/data'
elif USER == 'Torstein':
	folder  = '/home/torstein/Stanford/aa277/aa277_project/data'
elif USER == 'Bradley':
    folder  = '/home/bcollico/github/aa277_project/data'
else:
	raise Exception('Need to set user folder')

def visualize_training_data(path_to_data):
    data = read_training_data.read_training_data(path_to_data)

    x_i = np.zeros((data.n_episodes, data.n_agents))
    y_i = np.zeros((data.n_episodes, data.n_agents))
    xg_i = np.zeros((data.n_episodes, data.n_agents))
    yg_i = np.zeros((data.n_episodes, data.n_agents))

    for i in tqdm(range(data.n_episodes)):
        for j in range(data.n_agents):
            x_i[i,j] = data.traj[i].Px[j][0]
            y_i[i,j] = data.traj[i].Py[j][0]
            xg_i[i,j] = data.traj[i].Pg[j][0]
            yg_i[i,j] = data.traj[i].Pg[j][1]

    colors = ['b', 'g', 'r', 'k', 'y']

    ax = plt.axes()
    ax.set_xlim([-5,5])
    ax.set_ylim([-2.5,2.5])
    ax.set_title("Initial States Distribution")
    ax.set_xlabel("X Initial")
    ax.set_ylabel("Y Initial")

    for j in range(data.n_agents):
        ax.scatter(x_i[:,j], y_i[:,j], color=colors[j], marker='.', label='Robot '+str(j))

    plt.legend()
    plt.show()

    ax2 = plt.axes()
    ax2.set_xlim([-5,5])
    ax2.set_ylim([-2.5,2.5])
    ax2.set_title("Initial Goals Distribution")
    ax2.set_xlabel("X Goal")
    ax2.set_ylabel("Y Goal")

    for j in range(data.n_agents):
        ax2.scatter(xg_i[:,j], yg_i[:,j], color=colors[j], marker='.', label='Robot '+str(j))

    plt.legend()
    plt.show()


if __name__=='__main__':
    path = folder+"/training_data_1000sim.csv"
    visualize_training_data(path)

