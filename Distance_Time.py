# Import Helpers
from typing import Any

import numpy as np
import random
import os
import pandas as pd

# Import Gymnasium stuff
import gymnasium as gym
from gymnasium import Env
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

import random
import gym
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import matplotlib.pyplot as plt
import rl_utils

class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class SACContinuous:
    ''' 处理连续动作的SAC算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim,
                                         action_bound).to(device)  # 策略网络
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第一个Q网络
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)  # 第二个Q网络
        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第一个目标Q网络
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state)[0]
        return [action.item()]

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)
        # 和之前章节一样,对倒立摆环境的奖励进行重塑以便训练
        rewards = (rewards + 8.0) / 8.0

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)



class TrainSpeedControl(Env):
    def __init__(self):
        self.dt = 1  # in s
        self.sensor_range = 300.0  # in m

        self.Mass = 300.0  # in Ton
        self.position = 0.0  # in m; track is 1-dim, only one coord is needed
        self.velocity = 0.0  # in m/s
        self.acceleration = 0.0  # in m/s**2
        self.prev_acceleration = 0.0  # in m/s**2
        self.Max_traction_F = 0.0  # in kN
        self.traction_power = 0.0  # in kW
        self.action_clipped = 0.0  # in m/s**2
        self.jerk = 0.0  # in m/s**3
        self.prev_action = 0.0  # [-1,1]

        self.time = 0.0  # in s
        self.total_energy_kWh = 0.0  # in Wh
        self.reward = 0.0

        self.reward_weights = [1.0, 0.5, 0.0, 1.0]
        self.energy_factor = 1.0
        # self.friction_deceleration = 0.02

        self.track_length = 100000.0
        self.target = 2000.0
        self.speed_limit_positions = [0.0, self.target]
        self.speed_limits = [20.0, 0.0]
        self.speed_final = 0.0
        self.running_time = 170.0


        self.Time = TimeSpeed[0]
        self.Speed = TimeSpeed[1]

        self.terminated = False
        self.truncated = False
        self.done = False
        self.episode_count = 0
        self.reroll_frequency = 10

        (self.current_speed_limit, self.future_speed_limits,
         self.future_speed_limit_distances) = self.sensor(self.position)

        self.specs = {
            'mass': 1000,
            'frontal_area': 2.38,
            'cW': 0.29,
            'acceleration_limits': [-1, 1],
            'velocity_limits': [-1, 100],
            'power_limits': [-50, 75],
            'track_length': [0, 2500]
        }

        """
        # Meaning of state features
        # 1. Train's current positon
        # 2. Train's current velocity
        # 3. Current speed limit
        # 4. Next speed limit
        # 5. Distance to next speed limit
        """

        # self.state_max = np.hstack(
        #     (self.specs['velocity_limits'][1],
        #      self.specs['acceleration_limits'][1],
        #      self.specs['velocity_limits'][1],
        #      self.specs['velocity_limits'][1],
        #      self.specs['track_length'][1]))

        # self.state_min = np.hstack(
        #     (self.specs['velocity_limits'][0],
        #      self.specs['acceleration_limits'][0],
        #      self.specs['velocity_limits'][0],
        #      self.specs['velocity_limits'][0],
        #      self.specs['track_length'][0],))
        self.state_max = np.hstack(
              ( self.specs['track_length'][1],
                self.specs['velocity_limits'][1],
                self.specs['velocity_limits'][1],
                self.specs['velocity_limits'][1] * np.ones(1),
                self.sensor_range * np.ones(1)))

        self.state_min = np.hstack(
              ( self.specs['track_length'][0],
                self.specs['velocity_limits'][0],
                self.specs['velocity_limits'][0],
                self.specs['velocity_limits'][0] * np.ones(1),
                np.zeros(1)))

        self.action_space = Box(low=-1.0,
                                high=1.0,
                                shape=(1,),
                                dtype=np.float32)

        self.observation_space = Box(low=self.state_min,
                                     high=self.state_max,
                                     dtype=np.float64)

    def step(self, action):
        """
        Take one 10Hz step:
        Update time, position, velocity, jerk, limits.
        Check if episode is done.
        Get reward.
        :param action: float within (-1, 1)
        :return: state, reward, done, info
        """

        assert self.action_space.contains(action), \
            f'{action} ({type(action)}) invalid shape or bounds'

        self.action_clipped = np.clip(action, -1, 1)[0]
        # print("velocity:", self.velocity)
        # print("positon:", self.position)
        self.update_motion(self.action_clipped)


        # s = 0.5 * a * t² + v0 * t + s0
        # self.position += (0.5 * self.acceleration * self.dt ** 2 +
        #                   self.velocity * self.dt)
        # # v = a * t + v0
        # self.velocity += self.acceleration * self.dt

        # Update speed limit
        (self.current_speed_limit, self.future_speed_limits,
         self.future_speed_limit_distances) = self.sensor(self.position)

        # Update others
        self.time += self.dt
        # self.jerk = abs(action_clipped - self.prev_action)
        # self.prev_action = self.action_clipped

        # Judge terminated condition
        self.terminated = bool(self.position >= self.track_length or self.time > self.running_time + 1)
        if self.terminated:
          self.episode_count += 1

        self.truncated = False

        # Calculate reward
        reward_list = self.get_reward()
        # print("reward_list:", reward_list)
        self.reward = np.array(reward_list).dot(np.array(self.reward_weights))

        if self.time == self.running_time:
          self.reward -= (self.velocity - self.speed_final)**2 + (self.position - self.target)**2

        self.prev_acceleration = self.acceleration

        # Update info
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'jerk': self.jerk,
            'time': self.time,
            'power': self.traction_power,
            'reward': self.reward,
            'action': self.action_clipped
        }

        # Update state
        # state = self.feature_scaling(self.get_state())
        state = np.hstack([self.position, self.velocity, self.current_speed_limit,
                          self.future_speed_limits, self.future_speed_limit_distances])

        return state, self.reward, self.terminated, self.truncated, info

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ):
        self.position = 0.0  # in m; track is 1-dim, only one coord is needed
        self.velocity = 0.0  # in m/s
        self.acceleration = 0.0  # in m/s**2
        self.prev_acceleration = 0.0  # in m/s**2
        self.Max_traction_F = 0.0  # in kN
        self.jerk = 0.0  # in m/s**3
        self.time = 0.0  # in s
        self.total_energy_kWh = 0.0  # in Wh
        self.terminated = False
        self.truncated = False
        self.current_speed_limit = 0.0
        self.future_speed_limits = 0.0
        self.future_speed_limit_distances = 0.0
        self.action_clipped = 0.0  # in m/s**
        self.traction_power = 0.0
        # if self.episode_count % self.reroll_frequency == 0:
        #     second_limit_position = np.random.uniform(500, 1000)
        #     self.speed_limit_positions = [0.0, second_limit_position, 1800]
        #     self.speed_limits = np.append(np.random.randint(5, 21, size=2), 0.0)

        # Update to call sensor method to initialize speed limits correctly
        (self.current_speed_limit, self.future_speed_limits, self.future_speed_limit_distances) = self.sensor(self.position)
        # print("current_speed_limit:", self.current_speed_limit)
        info = {
            'position': self.position,
            'velocity': self.velocity,
            'acceleration': self.acceleration,
            'jerk': self.jerk,
            'time': self.time,
            'power': self.traction_power,
            'reward': self.reward,
            'action': self.action_clipped
        }


        # state = self.feature_scaling(self.get_state
        state = np.hstack([self.position, self.velocity, self.current_speed_limit,
                           self.future_speed_limits, self.future_speed_limit_distances])
        return state, info


    def update_motion(self, action_clipped):
        resistance = self.Calc_Resistance()
        # print("resistance:", resistance)
        if self.velocity > 0:
            if action_clipped >= 0:
                force = action_clipped * self.Calc_Max_traction_F()
                # print("force1:", force)
                self.traction_power = force * self.velocity
            else:
                force = action_clipped * self.Calc_Max_braking_F()
                # print("force2:", force)
                self.traction_power = 0.0

            self.acceleration = (force - resistance) / self.Mass
            # Prevent reversing if velocity might turn negative
            if self.velocity + self.acceleration * self.dt < 0:
                self.acceleration = -self.velocity / self.dt

        elif self.velocity == 0:
            if action_clipped > 0:
                force = action_clipped * self.Calc_Max_traction_F()
                # print("force3:", force)

            else:
                force = 0
                # print("force4:", force)

            self.acceleration = max(0, (force - resistance) / self.Mass)
            self.traction_power = 0  # No power since velocity is 0 at this step

        # Update position and velocity using kinematic equations
        self.position += (0.5 * self.acceleration * self.dt ** 2 + self.velocity * self.dt)
        self.velocity += self.acceleration * self.dt



    def sensor(self, position):
    # Treat negative positions as 0
        if position < 0:
            position = 0

        current_speed_limit = 0.0
        current_speed_limit_i = 0
        next_speed_limit = 0.0
        next_speed_limit_distance = 0.0
          #next2_speed_limit = 0.0
          #next2_speed_limit_distance = 0.0

          # Determine the current speed limit
        for i, (pos, sl) in enumerate(
                zip(self.speed_limit_positions, self.speed_limits)):
          if pos <= position:
                current_speed_limit = sl
                current_speed_limit_i = i

        # Determine the current speed limit
        if current_speed_limit_i + 1 > len(self.speed_limits) - 1:
            next_speed_limit = current_speed_limit
            next_speed_limit_distance = self.sensor_range
        elif (self.speed_limit_positions[current_speed_limit_i + 1] - position
              > self.sensor_range):
            next_speed_limit = current_speed_limit
            next_speed_limit_distance = self.sensor_range
        else:
            next_speed_limit = self.speed_limits[current_speed_limit_i + 1]
            next_speed_limit_distance = self.speed_limit_positions[
                                          current_speed_limit_i + 1] - position

        # if current_speed_limit_i + 2 > len(self.speed_limits) - 1:
        #     next2_speed_limit = next_speed_limit
        #     next2_speed_limit_distance = self.sensor_range
        # elif (self.speed_limit_positions[current_speed_limit_i + 2] - position
        #       > self.sensor_range):
        #     next2_speed_limit = next_speed_limit
        #     next2_speed_limit_distance = self.sensor_range
        # else:
        #     next2_speed_limit = self.speed_limits[current_speed_limit_i + 2]
        #     next2_speed_limit_distance = self.speed_limit_positions[
        #                                      current_speed_limit_i + 2] - position
        # future_speed_limits = [next_speed_limit, next2_speed_limit]
        # future_speed_limit_distances = [
        #     next_speed_limit_distance, next2_speed_limit_distance
        # ]
        future_speed_limits = next_speed_limit
        future_speed_limit_distances = next_speed_limit_distance
        return (current_speed_limit, future_speed_limits,
                future_speed_limit_distances)

    def get_reward(self):
        """
        Calculate the reward for this time step.
        Requires current limits, velocity, acceleration, jerk, time.
        Get predicted energy rate (power) from car data.
        Use negative energy as reward.
        Use negative jerk as reward (scaled).
        Use velocity as reward (scaled).
        Use a shock penalty as reward.
        :return: reward
        """
        target_speed = 0
        # calc forward or velocity reward

        # if self.time < 161:
        #   target_speed = self.Speed[self.time]
        #   reward_forward = abs(self.velocity - target_speed) / (1 +
        #                         abs(self.velocity - target_speed))
        # else:
        #   reward_forward = 0

        if self.future_speed_limits > 0:
          reward_forward = abs(self.velocity-self.future_speed_limits) / self.future_speed_limits
        else:
          reward_forward = abs(self.velocity - self.future_speed_limits) / (1 +
                                abs(self.velocity - self.future_speed_limits))
        # calc distance reward
        # reward_distance = abs(self.position - self.target) / self.target
        # calc energy reward
        # if self.velocity >= 0:
        #   reward_energy = self.traction_power
        #   energy_max = self.Calc_Max_traction_F() * self.velocity
        #   reward_energy /= energy_max
        # else:
        reward_energy = (self.velocity / self.speed_limits[0])**4 * max(0, self.action_clipped)
        # reward_energy = 0
        # print("reward_energy:", reward_energy

        # print("reward_energy:", reward_energy

        # calc jerk reward
        # reward_jerk = 1 if self.acceleration * self.prev_acceleration < 0 else 0
        reward_jerk = 0

        # calc shock reward
        reward_shock = 1 if self.velocity > self.current_speed_limit else 0

        # print(f"reward_forward: {reward_forward}")
        # print(f"reward_energy: {reward_energy}")
        # print(f"reward_jerk: {reward_jerk}")
        # print(f"reward_shock: {reward_shock}")

        # print("reward_stop:", reward_stop

        reward_list = [
            -reward_forward, -reward_energy, -reward_jerk, -reward_shock]
        # print("reward_list:", reward_list)
        return reward_list

    def Calc_Max_traction_F(self):
        """
        Calculate the traction force based on the speed in m/s.

        Parameters:
        - speed (float): Speed in km/h

        Returns:
        - float: Traction force in kN
        """
        speed = self.velocity * 3.6  # Convert speed from m/s to km/h
        f_t = 263.9  # Initial traction force value in kN (acceleration phase)
        p_max = f_t * 43 / 3.6  # Maximum power during acceleration in kW

        # If power exceeds the maximum power limit, then limit the traction force
        if speed > 0:
          if (f_t * speed / 3.6) > p_max:
              f_t = p_max / (speed / 3.6)

          # Additional condition to limit the traction force
          if f_t > (263.9 * 43 * 50 / (speed ** 2)):
              f_t = 263.9 * 43 * 50 / (speed ** 2)
        if speed == 0:
            f_t = 263.9  # Set traction force to initial value if speed is 0

        return f_t

    def Calc_Max_braking_F(self):
        """
        Calculate the braking force based on the speed.

        Parameters:
        - speed (float): Speed in km/h

        Returns:
        - float: Braking force in kN
        """

        speed = self.velocity * 3.6  # Convert speed from m/s to km/h
        if speed <= 0:
            f_b = 200
        else:
            if speed > 0 and speed <= 5:
                f_b = 200
            elif speed > 5 and speed <= 48.5:
                f_b = 389
            elif speed > 48.5 and speed <= 80:
                f_b = 913962.5 / (speed ** 2)
            else:
                f_b = 200  # Assumes no braking force calculation outside specified range

        # Apply a final modification factor to the braking force
        # f_b = 0.8 * f_b

        return f_b

    def Calc_Resistance(self):
        """
        Calculate the basic resistance of a train running at a given speed.

        :param speed: Speed of the train in km/h
        :return: Basic resistance in kN
        """
        n = 24  # Number of axles
        N = 6  # Number of cars
        A = 10.64  # Cross-sectional area of the train in m^2
        speed = self.velocity * 3.6  # Convert speed from m/s to km/h

        f_r = (6.4 * self.Mass + 130 * n + 0.14 * self.Mass * abs(speed) +
              (0.046 + 0.0065 * (N - 1)) * A * speed**2) / 1000
        # f_r = 0.1 * f_r
        return f_r


    def render(self):
        pass




env = TrainSpeedControl()
# check_env(env, warn=True)
log_path = os.path.join('Training', 'Logs')
checkpoint_path = os.path.join('Training', 'Saved Models')

# Create checkpoint callback
checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=checkpoint_path,
                                         name_prefix='SAC_model')

model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=log_path)
model.learn(total_timesteps=10, callback=checkpoint_callback)


