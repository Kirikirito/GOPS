#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab(iDLab), Tsinghua University
#
#  Creator: Yuhang Zhang
#  Description: Acrobat Environment
#
#  Update Date: 2021-05-55, Yuhang Zhang: create environment

import math
import warnings
import numpy as np

from gym import spaces
from gym.utils import seeding
import torch


class GymCartpolecontiModel:

    def __init__(self):
        """
        you need to define parameters here
        """
        # define your custom parameters here
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = (self.masspole * self.length)
        self.force_mag = 10.0
        self.theta_threshold_radians = 12 * 2 * math.pi / 360  # 12deg
        self.x_threshold = 2.4
        self.max_x = self.x_threshold * 2
        self.min_x = -self.max_x
        self.max_x_dot = np.finfo(np.float32).max
        self.min_x_dot = -np.finfo(np.float32).max
        self.max_theta = self.theta_threshold_radians * 2  # 24deg
        self.min_theta = -self.max_theta
        self.max_theta_dot = np.finfo(np.float32).max
        self.min_theta_dot = -np.finfo(np.float32).max
        self.min_action = -1.0
        self.max_action = 1.0

        # define common parameters here
        self.state_dim = 4
        self.action_dim = 1
        self.lb_state = np.array([self.min_x, self.min_theta, self.min_x_dot, self.min_theta_dot])
        self.hb_state = np.array([self.max_x, self.max_theta, self.max_x_dot, self.max_theta_dot])
        self.lb_action = np.array([self.min_action])
        self.hb_action = np.array([self.max_action])
        self.action_space = spaces.Box(self.lb_action, self.hb_action)
        self.observation_space = spaces.Box(self.lb_state, self.hb_state)
        self.tau = 0.02  # seconds between state updates
        self.np_random, seed = seeding.np_random()
        self.max_iter = 200
        self.iter = 0
        self.state = None

    def seed(self, seed=None):
        """
        change the random seed of the random generator in numpy
        :param seed: random seed
        :return: the seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: torch.Tensor):
        """
        rollout the model one step, notice this method will change the value of self.state
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :return: next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                 reward:  datatype:torch.Tensor, shape:[batch_size, 1]
                 done:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        self.state, reward, done = self.forward(state=self.state, action=action)
        return self.state, reward, done

    def reset(self, state=None):  # TODO: batch reset
        """
        reset the state use the passed in parameter
        if no parameter is passed in, random initialize will be used
        :param state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        :return: state: datatype:torch.Tensor, shape:[batch_size, state_dim]
        """
        if state is None:
            state = self.np_random.uniform(low=-0.05, high=0.05, size=(self.state_dim,))
            self.state = torch.from_numpy(state)
        else:
            self.state = torch.from_numpy(state)
        return self.state

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """
        rollout the model one step, notice this method will not change the value of self.state
        you need to define your own state transition  function here
        notice that all the variables contains the batch dim you need to remember this point
        when constructing your function
        :param action: datatype:torch.Tensor, shape:[batch_size, action_dim]
        :return: next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
                              the state will not change anymore when the corresponding flag done is set to True
                 reward:  datatype:torch.Tensor, shape:[batch_size, 1]
                 done:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        warning_msg = "action out of action space!"
        if not ((action <= torch.from_numpy(self.hb_action)).all() and
                (action >= torch.from_numpy(self.lb_action)).all()):
            warnings.warn(warning_msg)
        #  define your forward function here: the format is just like: state_next = f(state,action)
        x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        force = self.force_mag * action
        temp = (torch.squeeze(force) + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
                   (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        state_next = torch.stack([x, x_dot, theta, theta_dot]).transpose(1, 0)
        ############################################################################################
        reward = self.reward(state_next, action)
        done = self.isdone(state_next)
        state_next = (~done) * state_next + done * state
        return state_next, reward, done

    def reward(self, state: torch.Tensor, action: torch.Tensor):
        """
        you need to define your own reward function here
        notice that all the variables contains the batch dim you need to remember this point
        :param state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
        :param action:  datatype:torch.Tensor, shape:[batch_size, action_dim]
        :return: reward:  datatype:torch.Tensor, shape:[batch_size, 1]
        """
        # define the reward function here the format is just like: reward = l(state,state_next,reward)
        reward = 1-self.isdone(state).float()
        #############################################################################################
        return reward

    def isdone(self,next_state: torch.Tensor):
        """
        define the ending condition here
        notice that all the variables contains the batch dim you need to remember this point
        :param next_state:  datatype:torch.Tensor, shape:[batch_size, state_dim]
        :return: done:   datatype:torch.Tensor, shape:[batch_size, 1]
                         flag done will be set to true when the model reaches the max_iteration or the next state
                         satisfies ending condition
        """
        # define the ending condation here the format is just like done = l(next_state)
        x, x_dot, theta, theta_dot = next_state[:, 0], next_state[:, 1], next_state[:, 2], next_state[:, 3]
        done = (x < -self.x_threshold) + \
               (x > self.x_threshold) + \
               (theta < -self.theta_threshold_radians) + \
               (theta > self.theta_threshold_radians)
        done = torch.unsqueeze(done, 1)
        ##############################################################################################
        if self.iter >= self.max_iter:
            done = torch.full(done.size(), True)
            self.iter = 0
        return done


if __name__ == "__main__":
    from modules.env.gym_cartpoleconti_data import GymCartpoleConti
    import matplotlib.pyplot as plt

    f = GymCartpolecontiModel()
    env = GymCartpoleConti()
    s = env.reset()
    s = s.astype(np.float32)
    s_real = []
    s_fake = []
    a = env.action_space.sample()
    a = torch.from_numpy(a)
    a.requires_grad_()
    a.retain_grad()
    tsp, _, done, _ = f(torch.tensor(s).view([1, 4]), torch.unsqueeze(a, 0))
    tsp[0][1].backward(retain_graph=True)
    print(a.grad)

'''
    for i in range(200):
        #print(i)
        a = env.action_space.sample()
        sp, r, d, _ = env.step(a)
        # print(s, a, sp)
        sp = sp.astype(np.float32)
        s_real.append(sp)
        # print(tts.shape)
        tsp, _, done, _ = f(torch.tensor(s).view([1, 4]), torch.tensor(a).view([1, 1]))
       # print(tsp.shape)
        s_fake.append(tsp.detach().numpy().astype(np.float32))
        if done:
            print(i)
            print(s)
            break
        s = sp

    # print(tsp)
    s_real = np.array(s_real)
    s_fake = np.hstack(s_fake)
    s_fake = s_fake.reshape(-1,4)
    plt.plot(s_real)
    plt.show()
    plt.plot(s_fake)
    plt.show()
    print("All states match, The model is right")
    s = torch.zeros([10, 4])
    a = torch.zeros([10, 1])
    sp = f(s, a)
    print(sp)
    print("batch_support") 
'''
