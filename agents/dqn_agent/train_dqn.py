import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import random
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
from agents.dqn_agent.dqn_model import DQN

ENV_NAME = "CartPole-v1"
TOTAL_TIMESTEPS = int(5e4)
GAMMA = 0.99
LR = 2.3e-3
BATCH_SIZE = 64
MEMORY_SIZE = 100_000
TARGET_UPDATE_FREQ = 10
LEARNING_STARTS = 1000
TRAIN_FREQ = 256
GRADIENT_STEPS = 128
EXPLORATION_FRACTION = 0.16
EXPLORATION_FINAL_EPS = 0.04
MODEL_PATH = "agents/dqn_agent/dqn_model.pth"
N_STEPS = 10

def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def linear_schedule(start_eps, final_eps, fraction, current_step, total_steps):
    progress = min(current_step / (fraction * total_steps), 1.0)
    return start_eps - (start_eps - final_eps) * progress

def get_n_step_info(n_step_buffer, gamma):
    """Compute n-step state, action, cum reward, next_state, done."""
    cum_reward = 0.0
    for idx, (_, _, r, _, d) in enumerate(n_step_buffer):
        cum_reward += (gamma ** idx) * r
        if d:
            break
    state_n, action_n, _, _, _ = n_step_buffer[0]
    next_state_n, _, _, next_done = n_step_buffer[-1][3], None, None, n_step_buffer[-1][4]
    done_n = next_done
    return state_n, action_n, cum_reward, next_state_n, done_n

def train(seed=123):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    env = gym.make(ENV_NAME)
    set_seed(seed, env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim).to(device)
    target_net = DQN(state_dim, action_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)
    n_step_buffer = deque(maxlen=N_STEPS)

    state, _ = env.reset()
    total_timesteps = 0
    episode = 0

    episode_rewards = []
    while total_timesteps < TOTAL_TIMESTEPS:
        done = False
        total_reward = 0
        episode_timesteps = 0

        while not done:
            epsilon = linear_schedule(1.0, EXPLORATION_FINAL_EPS, EXPLORATION_FRACTION, total_timesteps, TOTAL_TIMESTEPS)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                    q_values = policy_net(state_tensor)
                    action = int(torch.argmax(q_values).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD-N
            n_step_buffer.append((state, action, reward, next_state, done)) 
            if len(n_step_buffer) == N_STEPS:
                transition = get_n_step_info(n_step_buffer, GAMMA)
                memory.push(transition)
                n_step_buffer.popleft()  # slide window by one
            if done:
                # Flush remaining n-step transitions at episode end
                while len(n_step_buffer) > 0:
                    transition = get_n_step_info(n_step_buffer, GAMMA)
                    memory.push(transition)
                    n_step_buffer.popleft()

            state = next_state
            total_reward += reward
            total_timesteps += 1
            episode_timesteps += 1

            if total_timesteps > LEARNING_STARTS and total_timesteps % TRAIN_FREQ == 0:
                for _ in range(GRADIENT_STEPS):
                    if len(memory) >= BATCH_SIZE:
                        transitions = memory.sample(BATCH_SIZE)
                        batch = list(zip(*transitions))

                        states = torch.tensor(np.array(batch[0]), dtype=torch.float32).to(device)
                        actions = torch.tensor(np.array(batch[1]), dtype=torch.int64).unsqueeze(1).to(device)
                        rewards = torch.tensor(np.array(batch[2]), dtype=torch.float32).unsqueeze(1).to(device)
                        next_states = torch.tensor(np.array(batch[3]), dtype=torch.float32).to(device)
                        dones = torch.tensor(np.array(batch[4]), dtype=torch.float32).unsqueeze(1).to(device)

                        q_values = policy_net(states).gather(1, actions)
                        next_actions = policy_net(next_states).argmax(1, keepdim=True) # Double DQN
                        next_q_values = target_net(next_states).gather(1, next_actions).detach() # Double DQN
                        targets = rewards + (1 - dones) * (GAMMA ** N_STEPS) * next_q_values # TD-N

                        loss = nn.MSELoss()(q_values, targets)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

            if total_timesteps % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(policy_net.state_dict())

        episode_rewards.append(total_reward)
        print(f"Episode {episode}: Reward {total_reward}")
        state, _ = env.reset()
        episode += 1

    torch.save(policy_net.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards Over Time")
    plt.legend()
    plt.grid(True)

    plot_path = MODEL_PATH.replace(".pth", "_rewards.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Reward plot saved to {plot_path}")

if __name__ == "__main__":
    train()
