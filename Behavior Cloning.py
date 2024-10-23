import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch
from SAC import SAC_countinuous
import argparse
from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter

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
    agent = SAC_countinuous(**vars(opt)) # var: transfer argparse to dictionary
    agent.load('TrainSpeedControl', opt.ModelIdex)



def train_behavior_cloning(agent, csv_file, num_epochs=100, batch_size=64):
    # Load expert data from CSV
    data = pd.read_csv(csv_file, header=None)
    states = torch.FloatTensor(data.iloc[:, :agent.state_dim].values).to(agent.dvc)
    actions = torch.FloatTensor(data.iloc[:, agent.state_dim:].values).to(agent.dvc)

    dataset = torch.utils.data.TensorDataset(states, actions)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Train behavior cloning using supervised learning
    for epoch in range(num_epochs):
        for batch_states, batch_actions in data_loader:
            predicted_actions, _ = agent.actor(batch_states, deterministic=True, with_logprob=False)
            loss = F.mse_loss(predicted_actions, batch_actions)

            agent.actor_optimizer.zero_grad()
            loss.backward()
            agent.actor_optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    print("Behavior cloning training completed")


def pretrain_bc(self, expert_loader, pretrain_epochs):
    self.actor.train()  # Put actor in training mode

    for epoch in range(pretrain_epochs):
        for expert_batch in expert_loader:
            expert_states, expert_actions = expert_batch

            # Get the predicted actions from the actor
            predicted_actions, _ = self.actor(expert_states, deterministic=True, with_logprob=False)

            # Compute behavior cloning loss (MSE between expert and predicted actions)
            bc_loss = F.mse_loss(predicted_actions, expert_actions)

            # Update actor using the behavior cloning loss
            self.actor_optimizer.zero_grad()
            bc_loss.backward()
            self.actor_optimizer.step()

        # Optional: Print loss or save model after each epoch
        print(f'Epoch {epoch + 1}/{pretrain_epochs}, BC Loss: {bc_loss.item()}')

    print("Pretraining complete.")
