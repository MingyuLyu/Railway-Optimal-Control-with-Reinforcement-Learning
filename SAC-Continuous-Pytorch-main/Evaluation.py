from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
from datetime import datetime
from SAC import SAC_countinuous
import gymnasium as gym
import os, shutil
import argparse
import torch
from TrainEnv import TrainSpeedControl
import matplotlib.pyplot as plt

'''Hyperparameter Setting'''
parser = argparse.ArgumentParser()
parser.add_argument('--dvc', type=str, default='cpu', help='running device: cuda or cpu')
parser.add_argument('--EnvIdex', type=int, default=0, help='PV1, Lch_Cv2, Humanv4, HCv4, BWv3, BWHv3')
parser.add_argument('--write', type=str2bool, default=False, help='Use SummaryWriter to record the training')
parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
parser.add_argument('--Loadmodel', type=str2bool, default=False, help='Load pretrained model or Not')
parser.add_argument('--ModelIdex', type=int, default=100, help='which model to load')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--Max_train_steps', type=int, default=int(5e6), help='Max training steps')
parser.add_argument('--save_interval', type=int, default=int(100e3), help='Model saving interval, in steps.')
parser.add_argument('--eval_interval', type=int, default=int(2.5e3), help='Model evaluating interval, in steps.')
parser.add_argument('--update_every', type=int, default=50, help='Training Fraquency, in stpes')

parser.add_argument('--gamma', type=float, default=0.99, help='Discounted Factor')
parser.add_argument('--net_width', type=int, default=256, help='Hidden net width, s_dim-400-300-a_dim')
parser.add_argument('--a_lr', type=float, default=3e-4, help='Learning rate of actor')
parser.add_argument('--c_lr', type=float, default=3e-4, help='Learning rate of critic')
parser.add_argument('--batch_size', type=int, default=256, help='batch_size of training')
parser.add_argument('--alpha', type=float, default=0.12, help='Entropy coefficient')
parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive_alpha or Not')
opt = parser.parse_args()
opt.dvc = torch.device(opt.dvc) # from str to torch.device
# opt.dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(opt)

def main():
    env = TrainSpeedControl()
    # eval_env = TrainSpeedControl()
    opt.state_dim = env.observation_space.shape[0]
    opt.action_dim = env.action_space.shape[0]
    opt.max_action = float(env.action_space.high[0])   #remark: action space【-max,max】
    opt.max_e_steps = env._max_episode_steps
    print(f'Env:TrainSpeedControl  state_dim:{opt.state_dim}  action_dim:{opt.action_dim}  '
          f'max_a:{opt.max_action}  min_a:{env.action_space.low[0]}  max_e_steps:{opt.max_e_steps}')

    # Seed Everything
    env_seed = opt.seed
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random Seed: {}".format(opt.seed))

    agent = SAC_countinuous(**vars(opt))  # var: transfer argparse to dictionary
    if opt.Loadmodel: agent.load('TrainSpeedControl', opt.ModelIdex)

    s, info = env.reset(seed=env_seed)  # Do not use opt.seed directly, or it can overfit to opt.seed
    env_seed += 1
    done = False

    # Lists to store data for plotting
    positions = []
    velocities = []
    accelerations = []
    jerks = []
    times = []
    powers = []
    rewards = []
    actions = []

    while not done:
        a = agent.select_action(s, deterministic=False)
        s_next, r, dw, tr, info = env.step(a)
        done = (dw or tr)

        positions.append(info['position'])
        velocities.append(info['velocity'])
        accelerations.append(info['acceleration'])
        jerks.append(info['jerk'])
        times.append(info['time'])
        powers.append(info['power'])
        rewards.append(info['reward'])
        actions.append(info['action'])

        s = s_next
    plot_info_data(times, positions, velocities, accelerations, jerks, powers, rewards, actions)

def plot_info_data(times, positions, velocities, accelerations, jerks, powers, rewards, actions):
    # Create subplots for each data type
    plt.figure(figsize=(12, 10))

    # Position
    plt.subplot(3, 3, 1)
    plt.plot(times, positions, label='Position')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.legend()

    # Velocity
    plt.subplot(3, 3, 2)
    plt.plot(times, velocities, label='Velocity')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.legend()

    # Acceleration
    plt.subplot(3, 3, 3)
    plt.plot(times, accelerations, label='Acceleration')
    plt.xlabel('Time')
    plt.ylabel('Acceleration')
    plt.legend()

    # Jerk
    plt.subplot(3, 3, 4)
    plt.plot(times, jerks, label='Jerk')
    plt.xlabel('Time')
    plt.ylabel('Jerk')
    plt.legend()

    # Power
    plt.subplot(3, 3, 5)
    plt.plot(times, powers, label='Power')
    plt.xlabel('Time')
    plt.ylabel('Power (Traction)')
    plt.legend()

    # Reward
    plt.subplot(3, 3, 6)
    plt.plot(times, rewards, label='Reward')
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend()

    # Actions
    plt.subplot(3, 3, 7)
    plt.plot(times, actions, label='Actions')
    plt.xlabel('Time')
    plt.ylabel('Actions (Clipped)')
    plt.legend()

    # Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()