from collections import deque

import torch

from FrameStack import FrameStack
from PPO import PPO
from RolloutBuffer import RolloutBuffer
import gymnasium
from vizdoom import gymnasium_wrapper

num_episodes = 100000

frame_channel_size = 3
frame_stack_size = 4
frame_skip = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    env = initialize_env()

    agent_ppo = initialize_agent_ppo(env, device)
    buffer = RolloutBuffer()

    total_reward = 0
    episode_window_reward = 0

    for episode in range(num_episodes):
        episode_reward = 0
        done = False

        obs_current, info = env.reset()

        frame_stack = FrameStack(stack_size = frame_stack_size)
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
        print(f"Episode: {episode}, Episode Reward: {episode_reward}, Average Reward: {total_reward / (episode + 1)}")

        episode_window_reward += episode_reward

        if (episode + 1) % 5 == 0:
            optimize_agent(agent_ppo, buffer)
            print(f"Episode Window Average Reward: {episode_window_reward / 5}")
            print("Optimizing Agent...")

            episode_window_reward = 0

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
    old_log_probs = old_log_probs.to(device)
    values = values.to(device)

    returns = agent.compute_returns(
        rewards,
        terminateds,
        truncateds,
        discount_factor=0.99,
        next_values=next_values
    )

    #print(f"Returns: {returns}")

    returns = returns.to(device)
    advantages = agent.compute_advantages(returns, states)

    metrics = agent.optimize_model(states, actions, old_log_probs, returns, advantages)

    buffer.clear()

if __name__ == '__main__':
    main()
