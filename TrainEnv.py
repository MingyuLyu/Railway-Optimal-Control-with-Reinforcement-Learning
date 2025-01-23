# Import Gymnasium stuff
import gymnasium as gym
from gymnasium import Env
from gymnasium.core import ObsType
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
import numpy as np
import random
class TrainSpeedControl(Env):
    def __init__(self):
        # Fixed parameters
        self.dt = 1.0  # Time step in seconds
        self.Mass = 300.0  # Mass in tons
        self.Max_traction_F = 0.0  # Max traction force in kN
        self.Episode_time = 200.0  # Total episode time in seconds
        self.Running_time = 150.0

        # Environmental parameters
        self.track_length = 2500.0  # Track length in meters
        self.station = 2000.0
        self.max_speed = 22.222

        # Specs (fixed properties of the environment)
        self.specs = {
            'velocity_limits': [-1, 100],
            'power_limits': [-50, 75],
            'distance_limits': [-2500, 2500],
            'Episode_time': [0, 300]
        }
        """
         # Meaning of state features
         # 1. Train's distance left
         # 2. Train's running time left
         # 3. Train's current velocity
         # 4. Current speed limit
         """
        # Define action and observation spaces (fixed bounds)
        self.state_max = np.hstack((
            self.specs['distance_limits'][1],
            self.specs['Episode_time'][1],
            self.specs['velocity_limits'][1],
            self.specs['velocity_limits'][1]))

        self.state_min = np.hstack((
            self.specs['distance_limits'][0],
            self.specs['Episode_time'][0],
            self.specs['velocity_limits'][0],
            self.specs['velocity_limits'][0]))

        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=self.state_min, high=self.state_max, dtype=np.float64)

        # Reward structure (fixed)
        self.reward_weights = [0.0, 1.0, 0.0, 0.5, 1.0]
        self.energy_factor = 1.0

        # Max episode steps derived from Episode time and dt
        self._max_episode_steps = int(self.Episode_time / self.dt)

        # Miscellaneous constants
        self.episode_count = 0
        self.reroll_frequency = 10  # How often to reroll (fixed)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, any] | None = None,
    ):
        # Seed the environment for reproducibility, if a seed is provided
        # if seed is not None:
        #     self.np_random, seed = seeding.np_random(seed)

        # Dynamic state variables (reset for every episode)
        self.position = 0.0  # Starting position in meters
        self.distance_left = self.station  # Reset distance left to full track length
        self.velocity = 0.0  # Initial velocity in m/s
        self.speed_limit = self.max_speed  # Initial speed limit (can be randomized later if needed)
        self.acceleration = 0.0  # Initial acceleration in m/s^2
        self.prev_acceleration = 0.0  # Previous acceleration in m/s^2
        self.traction_power = 0.0  # Traction power in kW
        self.action_clipped = 0.0  # Clipped action in m/s^2
        self.jerk = 0.0  # Rate of change of acceleration in m/s^3
        self.prev_action = 0.0  # Previous action in [-1, 1]

        # Time-related variables (reset each episode)
        self.time = 0.0  # Current time in seconds
        self.Running_time = float(np.random.randint(137, 157))  # Remaining time in episode
        self.time_left = self.Running_time  # Remaining time in episode
        # print('Running Time:', self.time_left)
        self.total_energy_kWh = 0.0  # Total energy consumption in kWh
        self.reward = 0.0  # Reset reward accumulator

        # Episode-specific status flags
        self.terminated = False
        self.truncated = False
        self.done = False

        # Reroll logic for changing speed limits every `reroll_frequency` episodes
        # if self.episode_count % self.reroll_frequency == 0:
        #     second_limit_position = np.random.uniform(500, 1000)  # Random second speed limit position
        #     self.speed_limit_positions = [0.0, second_limit_position, 1800]
        #     self.speed_limits = np.append(np.random.randint(5, 21, size=2), 0.0)
        #
        # # Increment episode counter
        # self.episode_count += 1

        # Information dictionary (tracking key environment variables)
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

        # Create the initial state (distance_left, time_left, velocity, speed_limit)
        state = np.hstack([self.distance_left, self.time_left, self.velocity, self.speed_limit])

        # Return the initial state and info dictionary
        return state, info

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
        # self.action_clipped = 0.0
        # if self.time < 80:
        #     self.action_clipped = 1.0
        # else:
        #     self.action_clipped = -1.0
        # # print("velocity:", self.velocity)
        # # print("positon:", self.position)
        self.update_motion(self.action_clipped)


        # s = 0.5 * a * t² + v0 * t + s0
        # self.position += (0.5 * self.acceleration * self.dt ** 2 +
        #                   self.velocity * self.dt)
        # # v = a * t + v0
        # self.velocity += self.acceleration * self.dt


        # Update others
        self.time += self.dt
        self.time_left = self.Running_time - self.time

        self.distance_left = self.station - self.position
        # self.jerk = abs(action_clipped - self.prev_action)
        # self.prev_action = self.action_clipped

        if self.station <= self.position:
            self.speed_limit = 0
        else:
            self.speed_limit = self.max_speed

        # Judge terminated condition
        self.terminated = bool(self.position - 20 <= self.station <= self.position + 20 and self.Running_time - 20 <= self.time <= self.Running_time + 20 and self.velocity < 0.1)

        self.truncated = bool(self.position >= self.track_length or self.time > self.Episode_time or self.velocity > self.speed_limit)

        # Calculate reward
        reward_list = self.get_reward()
        # print("reward_list:", reward_list)
        self.reward = np.array(reward_list).dot(np.array(self.reward_weights))

        if self.terminated:
            self.episode_count += 1
            self.reward += 1000
            self.reward -= (self.velocity ** 2 + abs(self.position - self.station) + abs(self.time - self.Running_time))

        if self.truncated:
            self.episode_count += 1
            self.reward -= (self.velocity ** 2 + abs(self.position - self.station) + abs(self.time - self.Running_time))

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
        state = np.hstack([self.distance_left, self.time_left, self.velocity, self.speed_limit])

        return state, self.reward, self.terminated, self.truncated, info



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
        # calc forward or velocity reward
        reward_forward = abs(self.position - self.station) / self.station

        # calc time reward

        reward_time = 1 if self.position < self.station else 0

        # calc energy reward
        reward_energy = self.action_clipped * self.velocity if self.action_clipped > 0 else 0

        # calc jerk reward
        reward_jerk = 1 if self.acceleration * self.prev_acceleration < 0 else 0

        # calc shock reward
        reward_shock = 1 if self.velocity > self.speed_limit else 0

        # print(f"reward_forward: {reward_forward}")
        # print(f"reward_energy: {reward_energy}")
        # print(f"reward_jerk: {reward_jerk}")
        # print(f"reward_shock: {reward_shock}")

        # print("reward_stop:", reward_stop

        reward_list = [
            -reward_forward, -reward_energy, -reward_time, -reward_jerk, -reward_shock]
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


