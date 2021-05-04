import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from scipy import signal
from scipy.fft import fft,fftfreq
from collections import deque
from scipy.integrate import odeint
import iir_filter
#PWM 180
[A,B,C,D]=[-7.794018686563599, 0.37538450501353504, -0.4891760779740128, -0.002568958116514183]
#pwm_offset=2
wFiltered=0 #filtered [21.222338184653342, 0.08719759445687768]
wAngular=4.85658326956131

def reward_fnCos(x, costheta, sintheta, theta_dot=0, sparse=False, Kx=0.5):
    if sparse:
        reward = 0.0
        if abs(np.arctan2(sintheta,costheta))<np.pi*30/180 and abs(x)<0.2:
            reward+=1
    else:
        reward = 1 + costheta - Kx * x ** 2
    return reward


length=0.416#center of mass
N_STEPS=800
# #TEST#
K1 = 18.8 #30#Guil  #dynamic friction
K2 = 0.11963736650935591#0.15 #2 #0.1  #friction on theta
# Mpoletest = 0.2
# McartTest = 0.6
Mcart=0.5
Mpole=0.075
class CartPoleContinous(gym.Env):
    def __init__(self,
                 Te=0.02,
                 discreteActions=False,
                 resetMode='random',
                 f_a=A,
                 f_b=B,
                 f_c=C,
                 f_d=D,
                 integrator="semi-euler",
                 tensionMax=12,#8.4706,
                 FILTER=False,
                 n=1,
                 Kp=0,
                 sparseReward=False,
                 Km=0,#1.2,
                 seed=0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.sparseReward=sparseReward
        self.Km=Km#bruit du mesure
        self.Kp=Kp#bruit du process
        self.g = 9.806
        self.masscart = Mcart
        self.masspoleIni = Mpole
        self.total_mass = (self.masspoleIni + self.masscart)
        self.length = 0.4611167818372032  # center of mass
        # self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = integrator  # 'rk'#
        self.theta_threshold_radians = math.pi
        self.x_threshold = 0.36
        # FOR DATA
        self.v_max = 15
        self.w_max = 100
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.discreteActions=discreteActions
        if self.discreteActions:
            self.action_space = spaces.Discrete(3)
        else:
            self.action_space= spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.tauMec = 0.1  # 05
        self.wAngularIni = wAngular  # 4.488 #T=1.4285, w=
        self.reward = None
        self.resetMode = resetMode
        self.fa=f_a
        self.fb=f_b
        self.fc=f_c
        self.fd=f_d
        self.tensionMax=tensionMax
        self.arr=[]
        self.n=n
        self.Kp=Kp
        self.FILTER=FILTER
        if self.FILTER:
            self.iirTheta_dot = iir_filter.IIR_filter(signal.butter(4, 0.9, 'lowpass', output='sos'))#2nd param 0.3
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def _calculate_force(self,action):
        f=self.masscart*(self.fa*self.state[1]+self.fb*self.tensionMax*action[0]+self.fc*np.sign(self.state[1])+self.fd)  #PWM 180 : 7.437548494321268
        return f
    def step(self, action):
        [x, x_dot, costheta, sintheta, theta_dot]=self.state
        if self.discreteActions:
            if action==0:
                action=[-1.0]
            elif action==1:
                action=[0.0]
            elif action==2:
                action=[1.0]
            else:
                raise Exception
        else:#continous
            pass
        self.wAngularIniUsed = np.random.normal(self.wAngularIni, 0.1, 1)[0]
        self.masspole=np.random.normal(self.masspoleIni, 0.01, 1)[0]
        self.polemass_length = (self.masspole * self.length)
        if self.kinematics_integrator=='semi-euler':
            for i in range(self.n):
                force = self._calculate_force(action)
                # force=np.random.normal(force,)
                xacc = (force + np.random.normal(0,scale=1.0) + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) / (
                                   self.masscart + self.masspole * sintheta ** 2)  # self.wAngularIni = np.random.normal(wAngular, 0.006, 1)[0]
                thetaacc = self.wAngularIniUsed ** 2 * sintheta + xacc / self.length * costheta - theta_dot * K2
                # xacc=np.random.normal(xacc, 0.03, 1)[0]
                # thetaacc=np.random.normal(thetaacc, 0.03, 1)[0]
                x_dot += self.tau / self.n * xacc
                x += x_dot * self.tau / self.n
                theta_dot += thetaacc * self.tau / self.n
                theta = math.atan2(sintheta, costheta)
                theta += theta_dot * self.tau / self.n
                costheta = np.cos(theta)
                sintheta = np.sin(theta)
        else:
            [x, x_dot, theta, theta_dot] = odeint(self.pend, [x, x_dot, math.atan2(sintheta, costheta), theta_dot], [0, 0.05],
                           args=(action, self.fa, self.fb, self.fc))[-1,:]
        #adding process noise (on x is negligible)
        if self.Kp!=0:
            theta_dot = np.random.normal(theta_dot, self.Kp * 0.2, 1)[0]
            x_dot = np.random.normal(x_dot, self.Kp * 1.2e-3, 1)[0]
            theta = np.random.normal(theta, 0.5 * self.Kp * 0.01, 1)[0]

        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        self.state = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32)
        done=False
        self.COUNTER+=1
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE or abs(theta_dot)>11:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        cost = reward_fnCos(x, costheta,sintheta=sintheta,theta_dot=theta_dot,sparse=self.sparseReward)
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        # # #adding noise on observed variables
        if self.Km!=0:
            theta= np.random.normal(theta, 1.5*self.Km * 0.01, 1)[0]
            theta_dot = np.random.normal(theta_dot, self.Km * 0.2, 1)[0]
            x_dot = np.random.normal(x_dot, self.Km * 1.2e-3, 1)[0]
        # # # filtering
        if self.FILTER:
            x_dot = self.iirX_dot.filter(x_dot)
            theta_dot = self.iirTheta_dot.filter(theta_dot)
        return np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot], dtype=np.float32), cost, done, {}

    def pend(self, state, t, action, fa, fb, fc):
        [x, x_dot, theta, theta_dot] = state
        dqdt = np.zeros_like(state)
        costheta, sintheta = [np.cos(theta), np.sin(theta)]
        dqdt[0] = state[1]
        force=self._calculate_force(action)#0.44*(fa*x_dot+fb*self.tensionMax*action[0]+fc*np.sign(x_dot))
        dqdt[1] = (self.masscart*(fa*x_dot+fb*8.47*action[0]+fc*np.sign(x_dot))#force
                   + self.masspole * self.g * sintheta * costheta - self.masspole * theta_dot ** 2 * sintheta * self.length) \
                  / (self.masscart + self.masspole * sintheta ** 2)
        dqdt[3] = self.g / self.length * sintheta# + dqdt[1] / self.length * costheta - theta_dot * K2
        dqdt[2] = state[3]
        # print(dqdt)
        return dqdt
    def reset(self, costheta=-1, sintheta=0, xIni=None,iniSpeed=0.0):
        if self.FILTER:
            # self.iirX_dot=iir_filter.IIR_filter(signal.butter(4, 0.5, 'lowpass', output='sos'))
            self.iirTheta_dot.reset()
        self.COUNTER=0
        self.steps_beyond_done = None
        if self.resetMode=='experimental':
            self.state = np.zeros(shape=(5,))
            if xIni!=None:
                self.state[0] = xIni
                print(f'x = {self.state[0]}')
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        elif self.resetMode=='goal':
            self.state = np.zeros(shape=(5,))
            self.state[2] = 1
            self.state[0] = self.np_random.uniform(low=-0.1, high=0.1)
        elif self.resetMode == 'random':
            self.state = np.zeros(shape=(5,))
            self.state[0] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[1] = self.np_random.uniform(low=-0.3, high=0.3)
            self.state[4] = self.np_random.uniform(low=-5, high=5)
            theta=self.np_random.uniform(-math.pi,math.pi)
            self.state[2] = np.cos(theta)
            self.state[3] = np.sin(theta)
        else:
            print('not defined, choose from experimental/goal/random')
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



