import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from collections import deque


def reward_fnCos(x, costheta, theta_dot=0):
    cost = 1+costheta-x**2/25#-abs(theta_dot)*0.05
    return cost
def _action_static_friction(action, threshold=0.083):
    if abs(action)<threshold:
        return 0
    if action>0:
        return action-threshold
    else:
        return action+threshold
length=0.48#center of mass
N_STEPS=800
# #TEST#
K1 = 18.8 #30#Guil  #dynamic friction
K2 = 0.099#0.15 #2 #0.1  #friction on theta
Mpoletest = 0.2
McartTest = 0.6
Mcart=0.45
# Mpole=0.03#0.05#s
Mpole=0.06#6#0.05#s
#Applied_force=5.6#REAL for 180pwm
Applied_force=5.6#5.6 #6 or 2.2(a*m)

class CartPoleCusBottom(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te=0.05,
                 randomReset=False,
                 f_a=-18.03005925191054,
                 f_b=0.965036433340654,
                 f_c=-0.8992003750802359,
                 seed=0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.g = 9.806
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.47  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        # self.v_max = 15
        # self.w_max = 30
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1  # 05
        self.wAngularIni = 4.488  # 4.488 #T=1.4285, w=
        self.reward = None
        self.randomReset = randomReset
        self.fa=f_a
        self.fb=f_b
        self.fc=f_c
        self.tensionMax=8.47
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def _calculate_force(self,action):
        return self.masscart*(self.fa*self.state[1]+self.fb*self.tensionMax*action[0]+self.fc*np.sign(self.state[1]))
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot]=self.state
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = (self._calculate_force(action) + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (self.masscart + self.masspole * sintheta ** 2)
                thetaacc = self.g / self.length * sintheta + xacc / self.length * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta+=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
        else:
            pass
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)
        # print(f'{self.state}')
        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}
    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        if not self.randomReset:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[1] = self.np_random.uniform(low=-0.2, high=0.2)
            self.state[4] = self.np_random.uniform(low=-0.2, high=0.2)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height / 2  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 0.2
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(.8, .6, .4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5, .5, .8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        theta = math.atan2(x[3], x[2])
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

