from collections import deque

import torch
import torch.nn as nn

from PPOPolicyValueNetwork import PolicyValueNetwork
from PPOValueNetwork import ValueNetwork

class PPO(nn.Module):
    def __init__(self, action_dim, optimizer=None, learning_rate = 1e-3, clip_epsilon = 0.2) -> None:
        super().__init__()

        self.network = PolicyValueNetwork(action_dim)

        self.clip_epsilon_ = clip_epsilon
        self.value_coef_ = 0.5
        self.entropy_coef_ = 0.01

        if optimizer is None:
            self.optimizer_ = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate
            )
        else:
            self.optimizer_ = optimizer

    def forward(self, x):
        logits, value = self.network(x)

        dist = torch.distributions.Categorical(logits=logits)

        return dist, value

    def act(self, x):

        dist, state_value = self.forward(x)

        action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return action, log_prob, entropy, state_value

    def compute_returns(self, rewards, terminateds, truncateds, discount_factor, next_values=None ):
        returns = deque()

        if next_values is None:
            next_values = torch.zeros_like(rewards)

        for reward, terminated, truncated, next_value in zip(reversed(rewards), reversed(terminateds), reversed(truncateds), reversed(next_values)):

            if terminated:
                discounted_return = 0
            elif truncated:
                discounted_return = next_value
            else:
                discounted_return = 1 #This condition should never happen if function is called after episode end

            discounted_return = reward + discount_factor * discounted_return

            returns.appendleft(discounted_return)

        return torch.tensor(list(returns), dtype=torch.float32)

    #A_t = G_t - V(s_t)
    def compute_advantages(self, returns, states):

        _ , values = self.forward(states)

        advantages = returns - values
        advantages = advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages

    def optimize_model(self, states, actions, old_log_probs, returns, advantages):

        dist, values = self.forward(states)
        new_log_probs = dist.log_prob(actions)

        entropy = dist.entropy().mean()

        values = values.squeeze(-1)

        ratio = torch.exp(new_log_probs - old_log_probs)

        unclipped = ratio * advantages

        clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon_,1.0 + self.clip_epsilon_) * advantages

        policy_loss = -torch.min(unclipped,clipped).mean()
        value_loss = torch.nn.functional.mse_loss(values,returns )

        loss = (policy_loss + self.value_coef_ * value_loss - self.entropy_coef_ * entropy)

        self.optimizer_.zero_grad()

        loss.backward()

        self.optimizer_.step()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }









