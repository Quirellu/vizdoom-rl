import torch


class RolloutBuffer:

    def __init__(self):
        self.states = []
        self.actions = []

        self.rewards = []
        self.terminateds = []
        self.truncateds = []

        self.log_probs = []
        self.values = []
        self.next_values = []

    def store(self, state, action, reward, terminated, truncated, log_prob, value, next_value=0.0):

        self.states.append(state)

        self.actions.append(action)

        self.rewards.append(reward)

        self.terminateds.append(terminated)

        self.truncateds.append(truncated)

        self.log_probs.append(log_prob.detach())

        self.values.append(value.detach())

        self.next_values.append(next_value)

    def get_tensors(self):

        states = torch.stack(self.states)

        actions = torch.tensor(self.actions)

        rewards = torch.tensor(
            self.rewards,
            dtype=torch.float32
        )

        terminateds = torch.tensor(
            self.terminateds,
            dtype=torch.float32
        )

        truncateds = torch.tensor(
            self.truncateds,
            dtype=torch.float32
        )

        log_probs = torch.stack(
            self.log_probs
        )

        values = torch.stack(
            self.values
        ).squeeze(-1)

        next_values = torch.tensor(
            self.next_values,
            dtype=torch.float32
        )

        return (
            states,
            actions,
            rewards,
            terminateds,
            truncateds,
            log_probs,
            values,
            next_values
        )

    def clear(self):

        self.states.clear()

        self.actions.clear()

        self.rewards.clear()

        self.terminateds.clear()

        self.truncateds.clear()

        self.log_probs.clear()

        self.values.clear()

        self.next_values.clear()
