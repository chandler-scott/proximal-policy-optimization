import sys
from util.parser import *
from util.logger import CustomLogger
from util.model_utils import aggregate_models, print_model
from .agent import Agent, AgentClient
import gymnasium as gym
from typing import Tuple
import torch
from torch import nn
import numpy as np
import envs
from util.aggregator import Aggregator, AggregatorServer
from util.socket_util import Server, Client


def setup_aggregator() -> Tuple[CustomLogger, Aggregator]:
    parser = AggregatorParser()

    p_load = parser.args.policy_load_file
    p_save = parser.args.policy_save_file
    v_load = parser.args.value_load_file
    v_save = parser.args.value_save_file
    n_agents = parser.args.n_episodes
    n_episodes = parser.args.n_agents

    aggregator = Aggregator(p_load=p_load, p_save=p_save, v_load=v_load, v_save=v_save,
                            n_agents=n_agents, n_episodes=n_episodes)
    return CustomLogger(), aggregator


def setup_agent(argparser: Parser, actor_critic: Agent = Agent) -> Tuple[CustomLogger, gym.Env, Agent, int]:
    """
    Sets up the environment, agent, logger, and other necessary components for the program.

    Parameters:
        argparser (Parser): An instance of the argument parser used to retrieve command-line arguments.

    Returns:
        Tuple[CustomLogger, gym.Env, Agent, int]: A tuple containing the logger, environment, agent, and the number of episodes.

    """
    parser = argparser()

    # get parsed args
    environment, render_mode = parser.get_environment()
    n_episodes = int(parser.args.n_episodes)
    steps_per_epoch = int(parser.args.steps_per_epoch)
    logs_file, stats_file = parser.args.logs, parser.args.stats
    lam, gamma = parser.args.lam, parser.args.gamma
    policy_lr, policy_size = parser.args.policy_lr, parser.args.policy_hidden_size
    policy_save_file, policy_load_file = parser.args.policy_save_file, parser.args.policy_load_file
    value_lr, value_size = parser.args.value_lr, parser.args.value_hidden_size
    value_save_file, value_load_file = parser.args.value_save_file, parser.args.value_load_file

    # create env, agent, logger

    env = gym.make(environment, render_mode='rgb_array')
    agent = actor_critic(observation_space=env.observation_space, action_space=env.action_space,
                         local_steps_per_epoch=steps_per_epoch,
                         policy_lr=policy_lr, policy_hidden_sizes=policy_size,
                         policy_save_file=policy_save_file, policy_load_file=policy_load_file,
                         value_lr=value_lr, value_hidden_sizes=value_size,
                         value_save_file=value_save_file, value_load_file=value_load_file,
                         lam=lam, gamma=gamma)
    logger = CustomLogger(logs_file, stats_file)

    return logger, env, agent, n_episodes, steps_per_epoch


def setup_agentclient(argparser: Parser, actor_critic: AgentClient = AgentClient) -> Tuple[CustomLogger, gym.Env, Agent, int]:
    """
    Sets up the environment, agent, logger, and other necessary components for the program.

    Parameters:
        argparser (Parser): An instance of the argument parser used to retrieve command-line arguments.

    Returns:
        Tuple[CustomLogger, gym.Env, Agent, int]: A tuple containing the logger, environment, agent, and the number of episodes.

    """
    parser = argparser()

    # get parsed args
    environment, render_mode = parser.get_environment()
    n_episodes = int(parser.args.n_episodes)
    steps_per_epoch = int(parser.args.steps_per_epoch)
    logs_file, stats_file = parser.args.logs, parser.args.stats
    lam, gamma = parser.args.lam, parser.args.gamma
    policy_lr, policy_size = parser.args.policy_lr, parser.args.policy_hidden_size
    policy_save_file, policy_load_file = parser.args.policy_save_file, parser.args.policy_load_file
    value_lr, value_size = parser.args.value_lr, parser.args.value_hidden_size
    value_save_file, value_load_file = parser.args.value_save_file, parser.args.value_load_file

    # create env, agent, logger

    env = gym.make(environment, render_mode='rgb_array')
    agent = actor_critic(observation_space=env.observation_space, action_space=env.action_space,
                         local_steps_per_epoch=steps_per_epoch,
                         policy_lr=policy_lr, policy_hidden_sizes=policy_size,
                         policy_save_file=policy_save_file, policy_load_file=policy_load_file,
                         value_lr=value_lr, value_hidden_sizes=value_size,
                         value_save_file=value_save_file, value_load_file=value_load_file,
                         lam=lam, gamma=gamma)

    logger = CustomLogger(logs_file, stats_file)

    return logger, env, agent, n_episodes, steps_per_epoch


def train() -> None:
    """
    Trains the agent on a gymnasium environment using PPO.
    """
    logger, env, agent, n_episodes, steps_per_epoch = setup_agent(
        TrainingParser)

    ## Training Loop ##
    logger.info('Train')
    observation, cumulative_reward = env.reset()[0], 0
    logger.info(f'cumulative_reward')
    for i in range(n_episodes):
        for t in range(steps_per_epoch):
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

            epoch_ended = t == steps_per_epoch-1

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

        # learn
        agent.learn()

    # done training
    agent.close()


def play() -> None:
    """
    Plays the environment using a pretrained agent.
    """
    logger, env, agent, n_episodes, _ = setup_agent(PlayingParser)
    total_reward = 0

    logger.info('Cumulative Reward')
    for i in range(n_episodes):
        observation, cumulative_reward = env.reset()[0], 0
        while True:
            # run policy in env for T timesteps
            action, _, _ = agent.step(torch.as_tensor(
                np.array(observation), dtype=torch.float32))
            observation, reward, terminated, truncated, _ = env.step(
                action)
            done = terminated or truncated
            cumulative_reward += reward

            if done:
                logger.info(f'{i}, {cumulative_reward}')
                total_reward += cumulative_reward

                break
    avg_reward = total_reward / n_episodes
    logger.info(avg_reward)


def train_model():
    logger, env, agent, n_episodes, steps_per_epoch = setup_agent(
        TrainingParser)

    agent.train(env, n_episodes, steps_per_epoch)

    ## Training Loop ##
    logger.info('Train')
    logger.info(f'cumulative_reward')


def federated_server():
    # Setup
    parser = AggregatorParser()
    environment, render_mode = parser.get_environment()
    env = gym.make(environment, render_mode=render_mode)
    p_load = parser.args.policy_load_file
    p_save = parser.args.policy_save_file
    v_load = parser.args.value_load_file
    v_save = parser.args.value_save_file
    n_episodes = int(parser.args.n_episodes)
    port = int(parser.args.port)
    n_clients = int(parser.args.n_agents)

    logger = CustomLogger()
    agg_server = AggregatorServer(obs_dim=env.observation_space, act_dim=env.action_space,
                                  p_load=p_load, p_save=p_save, v_load=v_load, v_save=v_save,
                                  n_clients=n_clients, n_episodes=n_episodes, port=port)

    payload = (agg_server.policy.p_net.state_dict(),
               agg_server.value.v_net.state_dict())

    logger.info('sending payload...')
    agg_server.send(payload)
    logger.info('payload sent!')

    try:
        logger.info('waiting to receive payload...')
        p_models, v_models = agg_server.receive_models()
        logger.info(
            f'received payload!')

        logger.info('aggregating payload...')
        policy, value = agg_server.aggregate(p_models, v_models)

    except ConnectionError:
        CustomLogger().error("Connection error!")
    except Exception as e:
        CustomLogger().error(f"Different error!\n {str(e)}")
    finally:
        # print(value.state_dict().items())
        agg_server.close()


def federated_client():
    logger, env, agent, n_episodes, steps_per_epoch = setup_agentclient(
        TrainingParser)

    try:

        logger.info('waiting to receive payload.')
        p_net, v_net = agent.receive()
        agent.load_models(p_net, v_net)
        logger.info('payload received! commencing training..')

        agent.train(env, n_episodes, steps_per_epoch)

        logger.info('sending payload...')
        agent.send_models()
        logger.info('payload sent!')

        logger.info('waiting to receive ACK...')
        agent.receive()
        logger.info('received ACK. closing...')

    except Exception as e:
        logger.error(f'Error: {str(e)}')
    finally:
        agent.close()
