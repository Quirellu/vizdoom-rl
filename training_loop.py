import torch

from configs.environment_config import EnvironmentConfig
from agents.ppo.ppo import PPO
from trainer import Trainer
import gymnasium
from vizdoom import gymnasium_wrapper

def main():
    env_config = EnvironmentConfig()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = initialize_env(env_config)

    agent_ppo = initialize_agent_ppo(env, device, env_config)
    trainer = Trainer(env, agent_ppo, device)
    trainer.train()

    env.close()

    print("Training Complete!")

def initialize_env(env_config):
    env = gymnasium.make(
        env_config.name,
        render_mode=env_config.render_mode,
        frame_skip=env_config.frame_skip
    )

    return env

def initialize_agent_ppo(env, device, env_config):
    action_dim = env.action_space.n
    agent = PPO(action_dim=action_dim, frame_channel= env_config.frame_channel_size, stack_frames = env_config.frame_stack_size, learning_rate=2.5e-4).to(device)

    return agent

if __name__ == '__main__':
    main()
