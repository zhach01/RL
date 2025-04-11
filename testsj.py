import numpy as np
import torch
from env import IntegratedMuscleControlEnv
from TD3 import TD3_agent  # or use TD3_agent from TD3 (1).py if preferred

def main():
    # Create the environment
    env = IntegratedMuscleControlEnv()
    state_dim = env.observation_space.shape[0]   # e.g., 34 dimensions
    action_dim = env.action_space.shape[0]         # e.g., 6 muscles
    max_action = 1.0  # Since muscle activations are in the range [0,1]

    # Initialize the TD3 agent with appropriate hyperparameters.
    agent = TD3_agent(
         state_dim=state_dim,
         action_dim=action_dim,
         max_action=max_action,
         net_width=256,         # Example width; adjust as needed
         a_lr=1e-3,             # Actor learning rate
         c_lr=1e-3,             # Critic learning rate
         gamma=0.99,            # Discount factor
         delay_freq=2,          # Delay frequency for policy updates
         batch_size=128,        # Batch size for training
         dvc=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
         explore_noise=0.1      # Scale for exploration noise
    )

    episodes = 1000
    max_timesteps = 500

    for episode in range(episodes):
         state = env.reset()
         episode_reward = 0.0
         for t in range(max_timesteps):
              # Select action using TD3 agent (stochastic during training)
              action = agent.select_action(state, deterministic=False)
              next_state, reward, done, info = env.step(action)
              
              # Add experience to the agent's replay buffer
              agent.replay_buffer.add(state, action, reward, next_state, done)
              
              # Train the agent (this function samples from the replay buffer and updates networks)
              agent.train()
              
              state = next_state
              episode_reward += reward
              
              if done:
                   break
         print(f"Episode {episode+1} | Reward: {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    main()
