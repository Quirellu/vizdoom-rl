import torch

from agents.ppo.ppo import PPO
from buffers.rollout_buffer import RolloutBuffer
from configs.environment_config import EnvironmentConfig
from configs.training_config import TrainingConfig
from vizdoom import gymnasium_wrapper

class Trainer:

    def __init__(self, env, model, device):

        self.env = env
        self.env_config = EnvironmentConfig()

        self.agent = model
        self.agent_device = device
        self.config = TrainingConfig()

        self.buffer = RolloutBuffer()


    def train(self):

        for episode in range(self.config.num_episodes):
            episode_reward = 0
            done = False

            obs_current, info = self.env.reset()

            while not done:

                state = torch.tensor(obs_current["screen"],dtype=torch.float32,device=self.agent_device)

                state = state.permute(0, 3, 1, 2) #[stack_size, height, width, channels] -> [stack_size, channels, height, width]

                state = state.flatten(0, 1).unsqueeze(0) #[batch_size, channels * stack_size, height, width]]

                with torch.no_grad():
                    action, log_prob, entropy, value = self.agent.act(state)

                env_action = action.item()

                next_observation, reward, terminated, truncated, info = self.env.step(env_action)

                done = terminated or truncated

                next_value = 0.0

                if truncated and not terminated:
                    next_state = torch.tensor(next_observation["screen"], dtype=torch.float32, device=self.agent_device).unsqueeze(
                        0)
                    with torch.no_grad():
                        _, next_value_tensor = self.agent.forward(next_state)
                        next_value = next_value_tensor.item()

                self.buffer.store(state.squeeze(0), env_action, reward, terminated, truncated, log_prob, value, next_value)

                obs_current = next_observation
                episode_reward += reward

            if (episode + 1) % 5 == 0:
                self.optimize_agent()


    def optimize_agent(self):

        states, actions, rewards, terminateds, truncateds, old_log_probs, values, next_values = self.buffer.get_tensors()
        device = next(self.agent.parameters()).device

        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        terminateds = terminateds.to(device)
        truncateds = truncateds.to(device)
        old_log_probs = old_log_probs.to(device)
        values = values.to(device).view(-1)
        next_values = next_values.to(device).view(-1)

        advantages = self.agent.compute_advantages(
            rewards,
            values,
            terminateds,
            truncateds,
            next_values,
            discount_factor=0.99,
            gae_lambda=0.95
        )
        returns = advantages + values

        metrics = self.agent.optimize_model(states, actions, old_log_probs, returns, advantages)

        self.buffer.clear()
