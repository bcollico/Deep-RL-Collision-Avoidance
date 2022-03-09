from configs import *
from matplotlib import pyplot as plt
import numpy as np

successes  = np.load(FOLDER + '/rl_successes.npy')
failures   = np.load(FOLDER + '/rl_failures.npy')
collisions = np.load(FOLDER + '/rl_collisions.npy')

plt.figure()
plt.plot(successes, 'g', label='successes')
plt.plot(failures, 'r', label='failures')
plt.plot(collisions, 'b', label='collisions')
plt.legend()
plt.show()