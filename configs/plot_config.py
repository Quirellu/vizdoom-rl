from dataclasses import dataclass

@dataclass
class PlotConfig:
    reward_smoothing_points: int = 10