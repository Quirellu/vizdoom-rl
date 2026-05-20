from dataclasses import dataclass

@dataclass
class TrainingConfig:
    num_episodes: int = 10000

    step_num_optimization: int = 1000
    

