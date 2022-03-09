import numpy as np
import matplotlib.pyplot as plt
from state_definitions import get_rotated_state

class State:
    def __init__(self,x):
        self.px = x[0]
        self.py = x[1]
        self.vx = x[2]
        self.vy = x[3] 
        self.radius = x[4]
        self.pgx = x[5] 
        self.pgy = x[6]
        self.v_pref = x[7]
        self.theta = x[8]
        self.px1 = x[9]
        self.py1 = x[10]
        self.vx1 = x[11]
        self.vy1 = x[12] 
        self.radius1 = x[13]

def rotate_their( x, kinematic = True):
    # first translate the coordinate then rotate around the origin
    # 'px', 'py', 'vx', 'vy', 'radius', 'pgx', 'pgy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
    #  0     1      2     3      4        5     6         7        8       9      10     11    12       13
    
    state = State(x)
    #state = IndexTranslator(state.cpu().numpy())
    dx = state.pgx - state.px
    dy = state.pgy - state.py
    rot = np.arctan2(state.pgy-state.py, state.pgx-state.px)

    dg = np.linalg.norm(np.array([dx, dy]))
    v_pref = state.v_pref
    vx = state.vx * np.cos(rot) + state.vy * np.sin(rot)
    vy = state.vy * np.cos(rot) - state.vx * np.sin(rot)
    radius = state.radius
    if kinematic:
        theta = state.theta - rot
    else:
        theta = state.theta
    vx1 = state.vx1 * np.cos(rot) + state.vy1 * np.sin(rot)
    vy1 = state.vy1 * np.cos(rot) - state.vx1 * np.sin(rot)
    px1 = (state.px1 - state.px) * np.cos(rot) + (state.py1 - state.py) * np.sin(rot)
    py1 = (state.py1 - state.py) * np.cos(rot) - (state.px1 - state.px) * np.sin(rot)
    pgx = (state.pgx - state.px) * np.cos(rot) + (state.pgy - state.py) * np.sin(rot)
    pgy = (state.pgy - state.py) * np.cos(rot) - (state.pgx - state.px) * np.sin(rot)
    radius1 = state.radius1
    radius_sum = radius + radius1
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    da = np.linalg.norm(np.array([state.px - state.px1, state.py - state.py1]))

    ax = plt.subplot(1,2,1)
    ax1 = plt.subplot(1,2,2)

    ax.set_title('Original State', fontsize=16)
    ax.set_xlabel('X Position', fontsize=16)
    ax.set_ylabel('Y Position', fontsize=16)
    ax.grid(1)
    ax.plot([state.px,state.px1],[state.py,state.py1],color='k',linewidth=0.75,linestyle='--',label='Rel Pos')
    ax.plot(state.px, state.py,color='b',label='Robot 1',marker='.',markersize=12,linestyle='none')
    ax.quiver(state.px, state.py, state.vx, state.vy,color='b',angles='xy', scale_units='xy', scale=1)
    ax.plot(state.px1, state.py1,color='g',label='Robot 2',marker='.',markersize=12,linestyle='none')
    ax.quiver(state.px1, state.py1, state.vx1, state.vy1,color='g',angles='xy', scale_units='xy', scale=1)
    ax.plot(state.pgx, state.pgy,color='b',marker='*',markersize=12, label='Goal 1',linestyle='none')
    ax.quiver(state.px, state.py, np.cos(rot), np.sin(rot), color='k',angles='xy', scale_units='xy', scale=1)
    ax.quiver(state.px, state.py, -np.sin(rot), np.cos(rot), color='k',angles='xy', scale_units='xy', scale=1)
    ax.text((state.px+np.cos(rot))*1.05, (state.py+np.sin(rot))*1.05, 'x')
    ax.text((state.px-np.sin(rot))*1.05, (state.py+np.cos(rot))*1.05, 'y')
    ax.axis([-2,2,-2,2])
    ax.set_aspect(1)
    ax.legend()

    ax1.set_title('Rotated State', fontsize=16)
    ax1.set_xlabel('X Position', fontsize=16)
    ax1.set_ylabel('Y Position', fontsize=16)
    ax1.grid(1)
    ax1.plot([0,px1],[0,py1],color='k',linewidth=0.75,linestyle='--',label='Rel Pos')
    ax1.plot(0, 0,color='b',label='Robot 1',marker='.',markersize=12,linestyle='none')
    ax1.quiver(0, 0, vx, vy,color='b',angles='xy', scale_units='xy', scale=1)
    ax1.plot(px1, py1, color='g',label='Robot 2',marker='.',markersize=12,linestyle='none')
    ax1.quiver(px1, py1, vx1, vy1,color='g',angles='xy', scale_units='xy', scale=1)
    ax1.plot(pgx, pgy,color='b',marker='*',markersize=12, label='Goal 1',linestyle='none')
    ax1.quiver(0, 0, 1, 0, color='k',angles='xy', scale_units='xy', scale=1)
    ax1.quiver(0, 0, 0, 1, color='k',angles='xy', scale_units='xy', scale=1)
    ax1.text(1.05, -0.05, 'x')
    ax1.text(-0.05, 1.05, 'y')
    ax1.axis([-2,2,-2,2])
    ax1.set_aspect(1)
    ax1.legend()
    plt.show()

    new_state = np.array([dg, v_pref, vx, vy, radius, theta, vx1, vy1, px1, py1,
                                radius1, radius_sum, cos_theta, sin_theta, da])
    return new_state


def test_random_state(x):

    rotated_their = rotate_their(x)
    rotated_our = get_rotated_state(x)

    for i, (their, our) in enumerate(zip(rotated_their, rotated_our)):
        if abs(their-our)>1e-6:
            print(i, their, our)
    return np.allclose(rotated_their, rotated_our, rtol = 1e-3, atol = 1e-3)

if __name__ == '__main__':
    dim = 14
    for i in range(1000):
        x = np.random.randn(dim)

        if not test_random_state(x):
            print(rotate_their(x), get_rotated_state(x))
            
    pass
