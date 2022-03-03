import numpy as np

from model import get_rotated_state

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
    radius1 = state.radius1
    radius_sum = radius + radius1
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    da = np.linalg.norm(np.array([state.px - state.px1, state.py - state.py1]))

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