"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import csv
from collections import deque

# def reward_fnCos(x, costheta, theta_dot=0):
#     cost = 1+costheta-abs(x)/25 - theta_dot * 0.1
#     if costheta>0.9:
#         cost+=1
#     return cost

def reward_fnCos(x, costheta, theta_dot=0):
    cost = 1+costheta-abs(x)/25
    return cost
# def reward_fn(x,theta,action=0.0):
#     cost=2+np.cos(theta)-x**2/25
#     return cost
def _action_static_friction(action, threshold=0.083):
    if abs(action)<threshold:
        return 0
    if action>0:
        return action-threshold
    else:
        return action+threshold
N_STEPS=1000
# K1 = 18.8 #30#Guil  #dynamic friction
# K2 = 0.01#0.1  #friction on theta
# Mpoletest = 0.2
# McartTest = 0.6
# Mcart=1000#0.45#0.45
# Mpole=0.03#0.05#
# #Applied_force=5.6#REAL for 180pwm
# Applied_force=6#6

# #TEST#
K1 = 18.8 #30#Guil  #dynamic friction
K2 = 0.07#0.15 #2 #0.1  #friction on theta
Mpoletest = 0.2
McartTest = 0.6
Mcart=0.44#0.45#0.45
# Mpole=0.03#0.05#s
Mpole=0.06#6#0.05#s
#Applied_force=5.6#REAL for 180pwm
Applied_force=5.6#5.6 #6 or 2.2(a*m)

#TEST#
# K1 = 9.212 #30#Guil  #dynamic friction
# K2 = 0.3#0.15 #2 #0.1  #friction on theta
# Mpoletest = 0.2
# McartTest = 0.6
# Mcart=0.45#0.45#0.45
# # Mpole=0.03#0.05#s
# Mpole=0.062#0.05#s
# #Applied_force=5.6#REAL for 180pwm
# Applied_force=1.4#2.8 or 2.2(a*m)

DEBUG=False
#with U/force
class CartPoleCosSinDev(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # center of mass
        self.length = 0.45  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.37
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.thetas=[]

        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if DEBUG:
            print(self.state)
            print(action[0])
        assert self.observation_space.contains(self.state), 'obs_err'

        action[0]=_action_static_friction(action[0])
        self.COUNTER += 1
        force = action[0] * self.force_mag * self.total_mass
        n=2
        for i in range(n):
            x, x_dot, costheta, sintheta, theta_dot = self.state
            theta = math.atan2(sintheta, costheta)
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) - K2 * theta_dot

            if self.kinematics_integrator == 'euler':
                xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            elif self.kinematics_integrator == 'friction':
                xacc = -K1 * x_dot + temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            theta = self.rescale_angle(theta)
            self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
            done = False
            if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
                done = True
                x = np.clip(x, -self.x_threshold, self.x_threshold)
        # self.thetas.append(theta)
        cost = reward_fnCos(x, costheta)
        # cost = reward_fnCos(x, costheta, x_dot)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 4
        if DEBUG:
            print(cost)
            print(done)
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(5,))#
        # self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        if DEBUG:
            print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
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
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleCosSinWorking(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.37
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.thetas=[]

        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        # action=_action_static_friction(action)
        self.COUNTER += 1
        self.total_mass = (self.masspole + self.masscart)
        x, x_dot, costheta, sintheta, theta_dot = self.state
        if DEBUG:
            print(self.state)
        force = action[0] * self.force_mag
        theta = math.atan2(sintheta, costheta)
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) - K2 * theta_dot
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        elif self.kinematics_integrator == 'friction':
            xacc = -K1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot
        theta = self.rescale_angle(theta)
        self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
        done = False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        # self.thetas.append(theta)
        cost = reward_fnCos(x, costheta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 4
        if DEBUG:
            print(cost)
            print(done)
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        if DEBUG:
            print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
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
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleCosSinHistory(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 k_history_len=2,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        # self.length = 0.5  # center of mass
        self.length = 0.47  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.37
        # FOR DATA
        self.v_max = 100
        self.w_max = 100
        self.thetas=[]
        self.k_history_len = k_history_len
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])
        high=np.append(high,np.ones(shape=(k_history_len,)))
        self.action_history_buffer = deque(np.zeros(self.k_history_len), maxlen=self.k_history_len)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if DEBUG:
            print(self.state)
            print(action[0])
        assert self.observation_space.contains(self.state), 'obs_err'
        self.action_history_buffer.append(action[0])
        action[0]=_action_static_friction(action[0])
        self.COUNTER += 1
        force = action[0] * self.force_mag * self.total_mass

        n=2
        for i in range(n):

            theta = math.atan2(sintheta, costheta)
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)) - K2 * theta_dot

            if self.kinematics_integrator == 'euler':
                xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            elif self.kinematics_integrator == 'friction':
                xacc = -K1 * x_dot + temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/n * xacc
                x = x + self.tau/n * x_dot
                theta_dot = theta_dot + self.tau/n * thetaacc
                theta = theta + self.tau/n * theta_dot
            theta = self.rescale_angle(theta)
            self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot], dtype=np.float32)
            done = False
            if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
                done = True
                x = np.clip(x, -self.x_threshold, self.x_threshold)
        # self.thetas.append(theta)
        cost = reward_fnCos(x, costheta)
        self.state=np.append(self.state,self.action_history_buffer)
        if x < -self.x_threshold or x > self.x_threshold:
            cost = cost - self.MAX_STEPS_PER_EPISODE / 4
        if DEBUG:
            print(cost)
            print(done)
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(7,))#
        # self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        if DEBUG:
            print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
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
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class CartPoleCosSinObsNDev(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 10
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'friction'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.37
        # FOR DATA
        self.v_max = 100
        self.w_max = 100


        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.n_obs=20
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.observation_space.contains(self.state), 'obs_err'
        #action=_action_static_friction(action)
        self.COUNTER+=1
        print(action)
        force = action * self.force_mag
        self.total_mass = (self.masspole + self.masscart)
        for i in range(self.n_obs):
            x, x_dot, costheta, sintheta, theta_dot = self.state
            theta=math.atan2(sintheta, costheta)
            temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
            thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass))-K2*theta_dot
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

            if self.kinematics_integrator == 'euler':
                x_dot = x_dot + self.tau/self.n_obs * xacc
                x = x + self.tau/self.n_obs * x_dot
                theta_dot = theta_dot + self.tau/self.n_obs * thetaacc
                theta = theta + self.tau/self.n_obs * theta_dot
            elif self.kinematics_integrator == 'friction':
                xacc = -K1 * x_dot + temp - self.polemass_length * thetaacc * costheta / self.total_mass
                x_dot = x_dot + self.tau/self.n_obs * xacc
                x = x + self.tau/self.n_obs * x_dot
                theta_dot = theta_dot + self.tau/self.n_obs * thetaacc
                theta = theta + self.tau/self.n_obs * theta_dot
            theta=self.rescale_angle(theta)
            self.state = np.array([x, x_dot, math.cos(theta), math.sin(theta), theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER==self.MAX_STEPS_PER_EPISODE:
            done = True
            x = np.clip(x,-self.x_threshold,self.x_threshold)

        cost = reward_fnCos(x, costheta)
        # print('cost: {}'.format(cost))
        # print('state: {}'.format(self.state))
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/5
        return self.state, cost, done, {}

    def reset(self):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(5,))
        self.state[1] = 0
        self.state[2] = -1
        self.state[3] = 0
        # self.state[2] = 0#visualise
        # self.state[3] = 1#visualise
        # print('reset state:{}'.format(self.state))
        return np.array(self.state)
    def rescale_angle(self,theta):
        return math.atan2(math.sin(theta),math.cos(theta))
    def render(self, mode='human'):
        screen_width = 800
        screen_height = 600

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = screen_height/2  # TOP OF CART
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
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

class CartPoleCusBottom(gym.Env):#CartPoleCosSinTension
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed : int = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.45  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = Applied_force
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 0.37
        # FOR DATA
        self.v_max = 100
        self.w_max = 100


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
        self.tauMec = 0.1
        self.wAngular = 4.488
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    def gTension(self, u, x_dot, uMin=0.805, slope=0.0545):  # u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u) < uMin:
            return 0
        #if x_dot == 0.0:
        return (u - np.sign(u) * uMin) * slope  # Fr opposes the tension
        #else:
           # return (u - np.sign(x_dot) * uMin) * slope  # Fr opposes the movement
   #def gTension(self, u, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        # if abs(u)<uMin:
        #     return 0
        # return (u-np.sign(u)*uMin)*slope
        #return u*slope
    def step(self, action):#180 = 8.47V
        assert self.observation_space.contains(self.state), 'obs_err'
        self.COUNTER+=1
        x, x_dot, costheta, sintheta, theta_dot = self.state
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                # xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47))
                # xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725) # static friction 0.0438725
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47,x_dot=x_dot))# - np.sign(x_dot)*0.0438725) # static friction 0.0438725
                # xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47)) # static friction 0.0438725
                thetaacc = self.wAngular ** 2 * sintheta - xacc * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta+=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)

        elif self.kinematics_integrator=='rk2':
            pass
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/5
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0):
        self.COUNTER=0
        self.steps_beyond_done = None
        self.state = np.zeros(shape=(5,))
        self.state[1] = iniSpeed
        self.state[2] = costheta
        self.state[3] = sintheta
        # self.state[0] = self.np_random.uniform(low=-0.2, high=0.2)
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
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr
