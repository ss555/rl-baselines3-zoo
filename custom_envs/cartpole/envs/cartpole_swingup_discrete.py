import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
def reward_fn(x,theta,action=0.0):
    cost=2+np.cos(theta)-x**2/25
    return cost
class CartPoleCusBottomDiscrete(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 Te = 0.05,
                 seed = 0):
        self.COUNTER = 0
        self.MAX_STEPS_PER_EPISODE = N_STEPS
        self.gravity = 9.81
        self.masscart = Mcart
        self.masspole = Mpole
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.45  # center of mass
        self.polemass_length = (self.masspole * self.length)
        self.tau = Te  # seconds between state updates
        self.kinematics_integrator = 'euler'#'rk'#
        self.theta_threshold_radians =  math.pi
        self.x_threshold = 0.37
        # FOR DATA
        self.v_max = 15
        self.w_max = 300
        # self.v_max = 15
        # self.w_max = 30
        high = np.array([
            self.x_threshold,
            self.v_max,
            1,
            1,
            self.w_max])

        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None
        self.total_mass = (self.masspole + self.masscart)
        self.tauMec = 0.1
        self.wAngular = 4.4#4.488 #T=1.4285, w=
        self.reward = None
    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random()
        return [seed]
    def gTension(self, u, x_dot, uMin=0.805, slope=0.0545): #u in volts, PWM 17.1/255 slope=1/19 m/s/V
        if abs(u)<uMin:
            return 0
        #if x_dot==0.0:
        return (u-np.sign(u)*uMin)*slope #Fr opposes the tension
        #else:
        #    return (u-np.sign(x_dot)*uMin)*slope #Fr opposes the movement
        ## while no tension and with speed, dynamic friction only?
        # return u*slope
    def step(self, action):#180 = 8.47V
        assert self.observation_space.contains(self.state), 'obs_err'
        self.COUNTER+=1
        if action==0:
            action=[-1.0]
        elif action==1:
             action=[0.0]
        elif action==2:
            action=[1.0]
        else:
            raise Exception
        x, x_dot, costheta, sintheta, theta_dot = self.state
        # x_dot=
        n=1
        if self.kinematics_integrator=='euler':
            for i in range(n):
                xacc = 1 / self.tauMec * (-x_dot + self.gTension(u=action[0] * 8.47,x_dot=x_dot))#- np.sign(x_dot)*0.0438725) # static friction 0.0438725
                thetaacc = self.wAngular ** 2 * sintheta - xacc * costheta - theta_dot * K2
                x_dot+=self.tau/n*xacc
                x+=x_dot*self.tau/n
                theta_dot+=thetaacc*self.tau/n
                theta=math.atan2(sintheta,costheta)
                theta+=theta_dot*self.tau/n
                costheta=np.cos(theta)
                sintheta=np.sin(theta)
            ## perturbations WORKING
            # x_dot=np.random.normal(x_dot,abs(x_dot/20),1)
            # theta_dot=np.random.normal(theta_dot,abs(theta_dot/20),1)
            # x_dot = np.random.normal(x_dot, abs(x_dot / 30), 1)
            # theta_dot = np.random.normal(theta_dot, abs(theta_dot / 30), 1)
        elif self.kinematics_integrator=='rk2':
            xacc = 1 / self.tauMec * (-x_dot + self.gTension(action[0] * 8.47) - np.sign(x_dot)*0.0438725)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER == self.MAX_STEPS_PER_EPISODE:
            # print('out of bound')
            done = True
            x = np.clip(x, -self.x_threshold, self.x_threshold)
        self.state=np.array([x,x_dot, costheta, sintheta, theta_dot],dtype=np.float32)

        cost = reward_fnCos(x, costheta,theta_dot)
        # print('cost: {}'.format(cost))
        if x == -self.x_threshold or x == self.x_threshold:
            cost=cost-self.MAX_STEPS_PER_EPISODE/2
        return self.state, cost, done, {}

    def reset(self, costheta=-1, sintheta=0, iniSpeed=0.0, deterministic=True):
        self.COUNTER=0
        self.steps_beyond_done = None
        if deterministic:
            self.state = np.zeros(shape=(5,))
            self.state[1] = iniSpeed
            self.state[2] = costheta
            self.state[3] = sintheta
        else:
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
        self.poletrans.set_rotation(-theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

            #model with action history/fr
class CartPoleCusBottomDiscreteOld(gym.Env):
    """
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The pendulum starts at the bottom and the goal is to swing up and balance
    Source:
        This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position             -4.8            4.8
        1	Cart Velocity             -Inf            Inf
        2	Pole Angle                -180°           180°
        3	Pole Velocity At Tip      -Inf            Inf

    Actions:
        Type: box(2)
        Num	Action
        -Force_threshold 	Push cart to the left
        1*Force_threshold	Push cart to the right

        Note: The amount the velocity is reduced or increased is not fixed as it depends on the angle the pole is pointing. This is because the center of gravity of the pole increases the amount of energy needed to move the cart underneath it

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        All observations are assigned a uniform random value between ±0.05

    Episode Termination:
        Cart Position is more than ±x_threshold (center of the cart reaches the edge of the display)
        Episode length is greater than N_counter
        Solved Requirements
        Considered solved when the average reward is greater than or equal to x over 100 consecutive trials.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self,
                 seed : int=0):
        self.COUNTER=0
        self.MAX_STEPS_PER_EPISODE=3000
        self.gravity = 10
        # 1 0.1 0.5 original
        self.masscart = 1
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 0.2
        self.tau = 0.05  # seconds between state updates
        self.kinematics_integrator = 'friction'#'euler'

        # Angle at which to fail the episode
        self.theta_threshold_radians = 180 * 2 * math.pi / 360
        self.x_threshold = 5.0
        # self.v_max=1.5
        # self.w_max=1
        # FOR DATA
        self.v_max = 500
        self.w_max = 500

        # Angle limit set to 2 * theta_threshold_radians so failing observation is still within bounds
        high = np.array([
            self.x_threshold * 2,
            self.v_max,
            self.theta_threshold_radians * 2,
            self.w_max])

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.seed(seed)
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=0):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.COUNTER+=1
        self.total_mass = (self.masspole + self.masscart)
        state = self.state
        x, x_dot, theta, theta_dot = state
        if action == 0:
            force = self.force_mag
        elif action == 1:
            force = -self.force_mag
        # force = action
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = np.random.normal((force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass,0)
        thetaacc = np.random.normal((self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)),0)
        xacc = np.random.normal(temp - self.polemass_length * thetaacc * costheta / self.total_mass,0)
        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max)
        elif self.kinematics_integrator == 'friction':
            xacc = -0.1 * x_dot / self.total_mass + temp - self.polemass_length * thetaacc * costheta / self.total_mass
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            # x_dot = np.clip(x_dot, -self.v_max, self.v_max)
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
            # theta_dot = np.clip(theta_dot, -self.w_max, self.w_max):
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot],dtype=np.float32)
        done=False
        if x < -self.x_threshold or x > self.x_threshold or self.COUNTER>self.MAX_STEPS_PER_EPISODE:
            done = True

        #cost = reward_fn(x,x_dot,theta,theta_dot,self.x_threshold,self.observation_space.high[1],self.observation_space.high[-1],action)
        cost=reward_fn(x,theta)
        if x < -self.x_threshold or x > self.x_threshold:
            cost=cost-100
        return self.state, cost, done, {}
#tensorboard --logdir ./sac_cartpole_tensorboard/
    #INFO
    #reward_fn 4096 takes 1h to train
    #sparse rewards doesnt learn
    def reset(self):
        # self.state = np.array([0, 0, math.pi, 0]).astype(np.float32)
        self.steps_beyond_done = None
        self.COUNTER = 0
        self.state = self.np_random.uniform(low=-0.2, high=0.2, size=(4,))
        self.state[2] = math.pi - self.state[2]
        self.state[0] = self.np_random.uniform(low=-2, high=2)
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 800
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * 1.0
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
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
