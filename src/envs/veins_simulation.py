import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
import numpy as np


class VeinsSimulation(gym.Env):
    def __init__(self, render_mode="None"):
        self.render_mode = render_mode
        
        # Define the action space
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=float)

        observation_low = np.array([0.0, 0.0, 0], dtype=np.float32)
        observation_high = np.array([1.0, 1.0, 4], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=observation_low, high=observation_high, dtype=np.float32)

        # Define other properties of your environment, such as observation space

    def step(self, action):
        # Implement the logic for taking a step in the environment based on the given action
        # Return the next observation, reward, whether the episode is done, and additional info
        pass

    def reset(self):
        # Reset the environment to its initial state and return the initial observation
        initial_observation = np.array([0.0, 0.0, 0], dtype=np.float32)
        return initial_observation
    
    def render(self):
        print('render hello!')
        # Implement visualization of the environment, if desired
        pass

    