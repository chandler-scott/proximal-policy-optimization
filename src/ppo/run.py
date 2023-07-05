from util.parser import *
from util.logger import CustomLogger
from .agent import Agent
import gymnasium as gym
from typing import Tuple
import torch
import numpy as np

LOCAL_STEPS = 100000


def setup(argparser: Parser, actor_critic: Agent = Agent) -> Tuple[CustomLogger, gym.Env, Agent, int]:
    """
    Sets up the environment, agent, logger, and other necessary components for the program.

    Parameters:
        argparser (Parser): An instance of the argument parser used to retrieve command-line arguments.

    Returns:
        Tuple[CustomLogger, gym.Env, Agent, int]: A tuple containing the logger, environment, agent, and the number of episodes.

    """
    parser = argparser()

    # get parsed args
    environment = parser.get_environment()
    n_episodes = int(parser.get_n_episodes())
    logs_file, stats_file = parser.get_logs_and_stats()

    # create env, agent, logger
    env = gym.make(environment, render_mode="human")
    agent = actor_critic(env.observation_space,
                         env.action_space, local_steps_per_epoch=LOCAL_STEPS)
    logger = CustomLogger(logs_file, stats_file)

    return logger, env, agent, n_episodes


def train() -> None:
    """
    Trains the agent on a gymnasium environment using PPO.
    """
    logger, env, agent, n_episodes = setup(TrainingParser)

    ## Training Loop ##
    logger.info('Train')
    observation, cumulative_reward = env.reset()[0], 0
    logger.info(f'cumulative_reward')
    for i in range(n_episodes):
        for t in range(LOCAL_STEPS):
            # run policy in env for T timesteps
            action, value, logp = agent.step(torch.as_tensor(
                np.array(observation), dtype=torch.float32))

            next_observation, reward, terminated, truncated, _ = env.step(
                action)
            done = terminated or truncated
            cumulative_reward += reward

            # save and log
            agent.buffer.store(observation, action, reward, value, logp)

            # update observation
            observation = next_observation

            epoch_ended = t == LOCAL_STEPS-1

            if done or epoch_ended:
                logger.info(f'{cumulative_reward}')
                if truncated or epoch_ended:
                    _, v, _ = agent.step(torch.as_tensor(
                        np.array(observation), dtype=torch.float32))
                    # logger.info('truncated' if truncated else 'epoch ended')
                else:
                    v = 0
                # compute advantage estimates
                agent.buffer.finish_path(v)
                observation, cumulative_reward = env.reset()[0], 0

        # optimize surrogate L wrt parameters
        # replace old network with new
        agent.learn()


def play() -> None:
    """
    Plays the environment using a pretrained agent.
    """
    logger, env, agent, n_episodes = setup(PlayingParser)

    logger.info('Play')
    for i in range(n_episodes):
        observation, cumulative_reward = env.reset()[0], 0
        while True:
            # run policy in env for T timesteps
            action, _, _ = agent.step(torch.as_tensor(
                np.array(observation), dtype=torch.float32))
            observation, reward, terminated, truncated, _ = env.step(
                action)
            done=terminated or truncated
            cumulative_reward += reward

            if done:
                logger.info(f'Cumulative Reward: {cumulative_reward}')
                break
