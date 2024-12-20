from utils import str2bool, evaluate_policy, Action_adapter, Action_adapter_reverse, Reward_adapter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from SAC import SAC_countinuous
import argparse

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

# Define Behavior Cloning Function
def train_behavior_cloning(agent, dataloader, num_epochs=50):
    agent.actor.train()  # Set the actor to training mode
    epoch_losses = []  # Store epoch losses for visualization

    for epoch in range(num_epochs):
        batch_losses = []
        for states, actions in dataloader:
            # Forward pass: Predict actions
            predicted_actions, _ = agent.actor(states, deterministic=True, with_logprob=False)

            # Compute behavior cloning loss (MSE)
            loss = F.mse_loss(predicted_actions, actions)

            # Backward pass: Update model parameters
            agent.actor_optimizer.zero_grad()
            loss.backward()
            agent.actor_optimizer.step()

            batch_losses.append(loss.item())

        # Record and print loss for the epoch
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    return epoch_losses

# Evaluate Behavior Cloning Model
def evaluate_behavior_cloning(agent, dataloader):
    agent.actor.eval()  # Set the actor to evaluation mode
    total_loss = 0

    with torch.no_grad():
        for states, actions in dataloader:
            # Predict actions
            predicted_actions, _ = agent.actor(states, deterministic=True, with_logprob=False)

            # Compute loss
            loss = F.mse_loss(predicted_actions, actions)
            total_loss += loss.item()

    return total_loss / len(dataloader)

# Main Script
def main():
    # Configuration
    state_dim = 4  # Number of state features
    action_dim = 2  # Number of action features
    hidden_dim = 128  # Hidden layer size
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3
    expert_data_file = "expert_data.csv"  # Path to your expert data

    # Load Expert Data
    data = pd.read_csv(expert_data_file, header=None)
    states = torch.FloatTensor(data.iloc[:, :state_dim].values)
    actions = torch.FloatTensor(data.iloc[:, state_dim:].values)

    # Create DataLoader
    dataset = TensorDataset(states, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize SAC Agent
    agent = SAC_countinuous(state_dim=state_dim, action_dim=action_dim, net_width=hidden_dim, a_lr=learning_rate, c_lr=learning_rate, dvc=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Train Behavior Cloning Model
    print("Training Behavior Cloning Model...")
    epoch_losses = train_behavior_cloning(agent, dataloader, num_epochs)

    # Evaluate the Model
    print("Evaluating Behavior Cloning Model...")
    eval_loss = evaluate_behavior_cloning(agent, dataloader)
    print(f"Evaluation Loss: {eval_loss:.4f}")

    # Save the Model
    torch.save(agent.actor.state_dict(), "bc_actor.pth")
    print("Behavior Cloning Model saved to 'bc_actor.pth'.")

    # Plot Training Loss
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Behavior Cloning Training Loss")
    plt.show()

if __name__ == "__main__":
    main()
