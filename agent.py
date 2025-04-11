import os
import numpy as np
import random
import copy
from collections import namedtuple, deque
# Import the enhanced models from model.py (which now work with the new ArmModel functions)
from model import EnhancedActor as Actor, EnhancedCritic as Critic
# Import your updated integrated muscle control environment
from env import IntegratedMuscleControlEnv
import torch
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import csv

# Import SummaryWriter for TensorBoard logging
from torch.utils.tensorboard import SummaryWriter

# ----------------------------------------------------------------------------
# Agent Hyperparameters
# ----------------------------------------------------------------------------
BUFFER_SIZE  = int(1e6)     # Replay buffer size
BATCH_SIZE = 128            # Mini-batch size
GAMMA = 0.99                # Discount factor
TAU = 1e-3                  # Soft update parameter
LR_ACTOR = 1e-3             # Learning rate of the actor
LR_CRITIC = 1e-3            # Learning rate of the critic
WEIGHT_DECAY = 0            # L2 weight decay

# Exploration parameters:
eps_start = 1.0  # initial scale for noise
eps_end   = 0.01 # final scale for noise
eps_decay = 1e-6 # decay per step

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------
# Agent Class for the New Environment (No Hill Model)
# ----------------------------------------------------------------------------
class Agent():
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random_seed

        # Actor Networks (Local + Target)
        self.actor_local = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Networks (Local + Target)
        self.critic_local = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, self.seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process (Ornstein-Uhlenbeck)
        self.noise = OUNoise(action_size, random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # TensorBoard logging (default disabled)
        self.writer = None
        # A counter to track training steps for logging or epsilon decay
        self.train_step = 0

        # Epsilon for noise scale
        self.eps = eps_start

    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory, then decide whether to learn.
        Also decay epsilon for exploration noise.
        """
        # Add experience to buffer
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples => learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        # Decay epsilon after every environment step
        self.eps = max(eps_end, self.eps - eps_decay)

    def act(self, state, add_noise=True):
        """
        Returns actions for given state according to current policy.
        Optionally add noise for exploration, scaled by self.eps.
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state).cpu().data.numpy().flatten()
        self.actor_local.train()

        if add_noise:
            noise_sample = self.noise.sample() * self.eps
            action_values += noise_sample

        # Clip action to [0, 1] (muscle activation range)
        return np.clip(action_values, 0.0, 1.0)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
        Update policy and value params using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        """
        states, actions, rewards, next_states, dones = experiences

        # Compute the next action using actor_target
        action_next = self.actor_target(next_states)
        # Evaluate Q(s', a') from the critic_target
        Q_targets_next = self.critic_target(next_states, action_next)
        # Q_target = r + γ * Q_targets_next * (1 - done)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Evaluate Q_expected from critic_local
        Q_expected = self.critic_local(states, actions)

        # Critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update all target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        # Logging
        self.train_step += 1
        if self.writer is not None:
            self.writer.add_scalar("Loss/critic", critic_loss.item(), self.train_step)
            self.writer.add_scalar("Loss/actor", actor_loss.item(), self.train_step)
            self.writer.add_scalar("Noise/epsilon", self.eps, self.train_step)

    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model params: θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, filename):
        """
        Save local networks and optimizers.
        """
        actor_path = filename + '_actor'
        actor_optimizer_path = filename + '_actor_optimizer'
        critic_path = filename + '_critic'
        critic_optimizer_path = filename + '_critic_optimizer'
        torch.save(self.actor_local.state_dict(), actor_path)
        torch.save(self.actor_optimizer.state_dict(), actor_optimizer_path)
        torch.save(self.critic_local.state_dict(), critic_path)
        torch.save(self.critic_optimizer.state_dict(), critic_optimizer_path)
        print('Model saved to:')
        print(actor_path, actor_optimizer_path, critic_path, critic_optimizer_path)

    def load(self, filename):
        """
        Load local networks and also copy them to target networks.
        """
        actor_path = filename + '_actor'
        actor_optimizer_path = filename + '_actor_optimizer'
        critic_path = filename + '_critic'
        critic_optimizer_path = filename + '_critic_optimizer'
        self.actor_local.load_state_dict(torch.load(actor_path))
        self.actor_optimizer.load_state_dict(torch.load(actor_optimizer_path))
        self.actor_target.load_state_dict(torch.load(actor_path))
        self.critic_local.load_state_dict(torch.load(critic_path))
        self.critic_optimizer.load_state_dict(torch.load(critic_optimizer_path))
        self.critic_target.load_state_dict(torch.load(critic_path))
        print('Model loaded from:', filename)


# ----------------------------------------------------------------------------
# Ornstein-Uhlenbeck Noise Process
# ----------------------------------------------------------------------------
class OUNoise:
    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state


# ----------------------------------------------------------------------------
# Replay Buffer
# ----------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'next_state', 'done'])
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        Sample a random batch of experiences from memory.
        """
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])
        ).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])
        ).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])
        ).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])
        ).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        ).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)


# ----------------------------------------------------------------------------
# Controller class that integrates the environment and agent
# ----------------------------------------------------------------------------
class Controller():
    def __init__(self, rand_seed=0, rew_type=None):
        """
        Creates an instance of the environment + agent, sets up synergy for training/testing.
        """
        #self.env = IntegratedMuscleControlEnv()  # uses the new ArmModel functions (with shaping, etc.)
        self.env = IntegratedMuscleControlEnv(max_steps=500)

        self.env.seed(rand_seed)
        self.agents = Agent(
            state_size=self.env.observation_space.shape[0],
            action_size=self.env.action_space.shape[0],
            random_seed=rand_seed
        )
        self.device = device
        self.rew_ver = rew_type

    def train(
        self,
        num_episodes=5000,
        max_timesteps=500,
        model_name='default',
        continue_from_model=None,
        start_episode=1
    ):
        print('Training started')
        tb_log_dir = os.path.join("logs", "tensorboard_run", model_name)
        writer = SummaryWriter(log_dir=tb_log_dir)
        self.agents.writer = writer

        if continue_from_model is not None:
            print(f"Continuing training from saved model: {continue_from_model}")
            self.agents.load(filename=continue_from_model)
        
        avg_reward = []
        base_dir = os.path.dirname(os.path.abspath(__file__))
        trained_models_dir = os.path.join(base_dir, "trained_models")
        if not os.path.exists(trained_models_dir):
            os.makedirs(trained_models_dir)
        
        for i_episode in range(start_episode, start_episode + num_episodes):
            state = self.env.reset()
            score = 0.0
            for t in range(max_timesteps):
                action = self.agents.act(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agents.step(state, action, reward, next_state, done)
                score += reward
                state = next_state
                if done:
                    break

            avg_reward.append(score)
            print(f"Episode {i_episode} | Steps {t+1} | Score {score:.2f}")
            writer.add_scalar('Reward/episode_score', score, i_episode)

            if i_episode % 100 == 0:
                rew_str = f"_{self.rew_ver}" if self.rew_ver is not None else ""
                save_filename = os.path.join(trained_models_dir, f"{model_name}_{i_episode}{rew_str}")
                self.agents.save(filename=save_filename)
                #self.plot_reward(avg_reward)
        
        writer.close()

    def plot_reward(self, avg_reward):
        logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        rew_str = f"_{self.rew_ver}" if self.rew_ver is not None else ""
        csv_filename = os.path.join(logs_dir, f'rewards_{len(avg_reward)}{rew_str}.csv')
        with open(csv_filename, 'w', newline='') as myFile:
            writer = csv.writer(myFile)
            writer.writerows([[r] for r in avg_reward])
        
        mean_over = 100
        R_mean = []
        for start_idx in range(0, len(avg_reward) - mean_over):
            window = avg_reward[start_idx : start_idx+mean_over]
            R_mean.append(np.mean(window))
        R_mean = np.asarray(R_mean)

        fig = plt.figure()
        plt.plot(np.arange(len(R_mean)), R_mean)
        plt.ylabel('Average reward (window=100)')
        plt.xlabel('Episode')
        pdf_filename = os.path.join(logs_dir, f'avg_reward_{len(avg_reward)}{rew_str}.pdf')
        plt.savefig(pdf_filename)
        timer = fig.canvas.new_timer(interval=20000)
        timer.add_callback(plt.close, fig)
        timer.start()
        plt.show()

    def test(self, num_test=10, max_timesteps=500, model_name='default'):
        print('Testing started')
        NumSuccess = 0
        base_dir = os.path.dirname(os.path.abspath(__file__))
        trained_models_dir = os.path.join(base_dir, "trained_models")
        rew_str = f"_{self.rew_ver}" if self.rew_ver is not None else ""
        model_path = os.path.join(trained_models_dir, f"{model_name}{rew_str}")
        actor_file = model_path + '_actor'
        if not os.path.exists(actor_file):
            print("Error: The model file does not exist:", actor_file)
            return

        self.agents.load(filename=model_path)
        for i in range(num_test):
            state = self.env.reset()
            self.agents.reset()
            for t in range(max_timesteps + 1):
                action = self.agents.act(state, add_noise=False)
                self.env.render()
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                if done:
                    break
            if t < max_timesteps:
                print('TEST: {}, SUCCESS'.format(i))
                NumSuccess += 1
            else:
                print('TEST: {}, FAIL'.format(i))
            self.env.close()
        accuracy = (NumSuccess / num_test) * 100
        print("Total {} success out of {} tests. Accuracy: {:.2f}%".format(NumSuccess, num_test, accuracy))


# ----------------------------------------------------------------------------
# Main testing code.
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    controller = Controller(rand_seed=0)
    controller.train(num_episodes=1000, max_timesteps=500, model_name="IntegratedMuscleControl")
