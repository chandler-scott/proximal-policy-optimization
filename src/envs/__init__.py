from .test_world import TestWorld
import gymnasium as gym


gym.register(
    id='TestWorld-v0',
    entry_point='envs:TestWorld',
)