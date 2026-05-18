import torch

from PPO import PPO
from RolloutBuffer import RolloutBuffer
import gymnasium
from vizdoom import gymnasium_wrapper

num_episodes = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():

    env = initialize_env()

    agent_ppo = initialize_agent_ppo(env, device)
    buffer = RolloutBuffer()


    for episode in range(num_episodes):
        episode_reward = 0
        done = False

        obs_current, info = env.reset()

        while not done:
            state = torch.tensor(obs_current["screen"], dtype=torch.float32, device=device).unsqueeze(0)

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

        print(f"Episode: {episode}, Total Reward: {episode_reward}")

        if (episode + 1) % 5 == 0:
            optimize_agent(agent_ppo, buffer)
            print("Optimizing Agent.")

def initialize_env():
    env = gymnasium.make(
        "VizdoomBasic-v1",
        render_mode="human"
    )

    return env

def initialize_agent_ppo(env, device):
    action_dim = env.action_space.n
    agent = PPO(action_dim=action_dim, learning_rate=0.001).to(device)

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
