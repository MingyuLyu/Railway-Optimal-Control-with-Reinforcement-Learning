import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

class BehaviorCloning(Env):
    def __init__(self):
        super().__init__()

        # ----------------------
        # Environment Settings
        # ----------------------
        self.dt = 1.0
        self.track_length = 2500.0
        self.max_speed = 22.222
        self._max_episode_steps = 2000  # or however many steps you want

        # Action = 1D in [-1, 1]
        self.action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # State = 4D, but we’ll define the bounds large enough
        high = np.array([1e6, 1e6, 1e3, 1e3], dtype=np.float32)
        self.observation_space = Box(-high, high, dtype=np.float32)

        # ----------------------
        # Load Expert Data
        # ----------------------
        expert_data_file = (
            r"C:\Users\root\Documents\GitHub\Railway-Optimal-Control-with-Reinforcement-Learning\expert_data.csv"
        )
        data = pd.read_csv(expert_data_file, header=None)

        # Keep only the first 81,999 rows:
        data = data.iloc[40000:81999].copy()
        # Suppose columns: s0, s1, s2, s3, a0
        self.states_array = data.iloc[:, :4].values  # shape: [N, 4]
        self.actions_array = data.iloc[:, 4].values  # shape: [N,] or [N,1] if you prefer

        # Make sure we have at least as many steps as we can run
        self.num_data = len(self.states_array)

        # ----------------------
        # Internal Variables
        # ----------------------
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.done = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # Gymnasium’s recommended pattern

        # Start the environment from the beginning of the dataset (or random, up to you).
        self.step_count = 0
        self.terminated = False
        self.truncated = False
        self.done = False

        # Set the initial state from the first row in your dataset
        init_state = self.states_array[self.step_count]
        state = np.array(init_state, dtype=np.float32)
        info = {}
        return state, info

    def step(self, action):
        # 1) Clip action to [-1, 1]
        action_clipped = float(np.clip(action, -1, 1)[0])
        # 3) Calculate Reward
        #    By default, MSE is “low is better,” but RL *maximizes* reward.
        #    If you literally do F.mse_loss(...) you are giving a *larger* reward for bigger error.
        #    So you might prefer reward = - MSE, so that the agent is *incentivized* to produce smaller error.

        # Convert to PyTorch Tensors for MSE
        # shape: (1,) or just a scalar
        next_state = self.states_array[self.step_count]
        pred_action_t = torch.tensor([action_clipped], dtype=torch.float32)
        expert_action_t = torch.tensor([self.actions_array[self.step_count]], dtype=torch.float32)

        # MSE
        mse_value = F.mse_loss(pred_action_t, expert_action_t)
        # Use negative MSE as the reward so that smaller MSE => higher reward
        reward = -mse_value.item()
        # 2) Advance your “dataset index”
        self.step_count += 1

        # Safety check: if we exceed the dataset length, we can’t continue
        if self.step_count >= self.num_data:
            self.terminated = True
        # 4) Check Termination or Truncation
        #    For example, if we exceed 2000 steps or we run out of data, we end the episode
        if self.step_count >= self._max_episode_steps:
            self.truncated = True

        # 5) Build next_state as a float32 array
        next_state = np.array(next_state, dtype=np.float32)

        info = {
            "step": self.step_count,
            "expert_action": float(expert_action_t.item()),
            "pred_action": action_clipped,
            "mse": mse_value.item()
        }

        done = self.terminated or self.truncated
        return next_state, reward, done, self.truncated, info

    def render(self):
        pass
