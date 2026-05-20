from collections import deque

import matplotlib.pyplot as plt
import torch

from FrameStack import FrameStack
from PPO import PPO
from RolloutBuffer import RolloutBuffer
import gymnasium
from vizdoom import gymnasium_wrapper

num_episodes = 100000
reward_smoothing_points = 10

frame_channel_size = 3
frame_stack_size = 4
frame_skip = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    env = initialize_env()

    agent_ppo = initialize_agent_ppo(env, device)
    buffer = RolloutBuffer()

    total_reward = 0
    reward_list = list()
    episode_window_reward = 0

    frame_stack = FrameStack(stack_size=frame_stack_size)

    for episode in range(num_episodes):
        episode_reward = 0
        done = False

        obs_current, info = env.reset()


        frame_stack.reset(obs_current["screen"])

        while not done:


            stack_np = frame_stack.step(obs_current["screen"])

            state = torch.tensor(stack_np, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, entropy, value = agent_ppo.act(state)

            env_action = action.item()

            next_observation, reward, terminated, truncated, info = env.step(env_action)

            done = terminated or truncated

            next_value = 0.0

            if truncated and not terminated:
                next_state = torch.tensor(next_observation["screen"], dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, next_value_tensor = agent_ppo.forward(next_state)
                    next_value = next_value_tensor.item()

            buffer.store(state.squeeze(0), env_action, reward, terminated, truncated, log_prob, value, next_value)

            obs_current = next_observation
            episode_reward += reward

        total_reward += episode_reward
        reward_list.append(episode_reward)
        print(f"Episode: {episode}, Episode Reward: {episode_reward}, Average Reward: {total_reward / (episode + 1)}")

        episode_window_reward += episode_reward

        if (episode + 1) % 5 == 0:
            optimize_agent(agent_ppo, buffer)
            print(f"Episode Window Average Reward: {episode_window_reward / 5}")
            print("Optimizing Agent...")

            episode_window_reward = 0

    env.close()
    smoothed_rewards = plot_rewards(reward_list, reward_smoothing_points)
    print("Training Complete!")

def initialize_env():
    env = gymnasium.make(
        "VizdoomBasic-v1",
        render_mode="human",
        frame_skip=frame_skip
    )

    return env

def initialize_agent_ppo(env, device):
    action_dim = env.action_space.n
    agent = PPO(action_dim=action_dim, frame_channel= frame_channel_size, stack_frames = frame_stack_size, learning_rate=2.5e-4).to(device)

    return agent

def optimize_agent(agent, buffer):

    states, actions, rewards, terminateds, truncateds, old_log_probs, values, next_values = buffer.get_tensors()
    device = next(agent.parameters()).device

    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    terminateds = terminateds.to(device)
    truncateds = truncateds.to(device)
    old_log_probs = old_log_probs.to(device)
    values = values.to(device).view(-1)
    next_values = next_values.to(device).view(-1)

    advantages = agent.compute_advantages(
        rewards,
        values,
        terminateds,
        truncateds,
        next_values,
        discount_factor=0.99,
        gae_lambda=0.95
    )
    returns = advantages + values

    metrics = agent.optimize_model(states, actions, old_log_probs, returns, advantages)

    buffer.clear()

def plot_rewards(reward_list, smoothing_points):
    if not reward_list:
        print("No rewards to plot.")
        return []

    if smoothing_points < 1:
        raise ValueError("smoothing_points must be at least 1")

    episodes = range(1, len(reward_list) + 1)
    reward_window = deque(maxlen=smoothing_points)
    smoothed_rewards = []

    for reward in reward_list:
        reward_window.append(reward)
        smoothed_rewards.append(sum(reward_window) / len(reward_window))

    plt.figure(figsize=(10, 5))
    plt.plot(episodes, reward_list, label="Episode Reward")
    plt.plot(episodes, smoothed_rewards, label=f"{smoothing_points}-Episode Average")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Rewards")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return smoothed_rewards

if __name__ == '__main__':
    main()
