
from dataclasses import dataclass

@dataclass
class EnvironmentConfig:
    name = "VizdoomBasic-v1"
    render_mode = "human"
    frame_skip: int = 3
    frame_stack_size: int = 4
    frame_channel_size: int = 3