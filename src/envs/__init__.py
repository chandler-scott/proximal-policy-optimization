from .test_world import TestWorld
from .veins_simulation import VeinsSimulation

import gymnasium as gym


gym.register(
    id='TestWorld-v0',
    entry_point='envs:TestWorld',
)