from time import sleep
import time
from util.model_utils import save_models as save_m, state_dict_to_bytes, bytes_to_state_dict, zeros_box_space
from util.parser import *
from util.logger import CustomLogger
import gymnasium as gym
from typing import Tuple
import torch
from torch import nn
import numpy as np
from models.aggregator import Aggregator, AggregatorServer
from models.agent import Agent, AgentClient


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
    policy_load_file = parser.args.policy_load_file
    value_lr, value_size = parser.args.value_lr, parser.args.value_hidden_size
    value_load_file = parser.args.value_load_file

    # create env, agent, logger

    env = gym.make(environment, render_mode='rgb_array')
    agent = actor_critic(observation_space=env.observation_space, action_space=env.action_space,
                         local_steps_per_epoch=steps_per_epoch,
                         policy_lr=policy_lr, policy_hidden_sizes=policy_size,
                         policy_load_file=policy_load_file,
                         value_lr=value_lr, value_hidden_sizes=value_size,
                         value_load_file=value_load_file,
                         lam=lam, gamma=gamma)
    logger = CustomLogger(logs_file, stats_file=stats_file)

    logger.stats_file = f'./out/{stats_file}'

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
    policy_load_file = parser.args.policy_load_file
    value_lr, value_size = parser.args.value_lr, parser.args.value_hidden_size
    value_load_file = parser.args.value_load_file
    save_models = parser.args.save_models

    port = int(parser.args.port)
    # create env, agent, logger

    env = gym.make(environment, render_mode='rgb_array')
    agent = actor_critic(observation_space=env.observation_space, action_space=env.action_space,
                         local_steps_per_epoch=steps_per_epoch,
                         policy_lr=policy_lr, policy_hidden_sizes=policy_size,
                         value_load_file=value_load_file, policy_load_file=policy_load_file,
                         value_lr=value_lr, value_hidden_sizes=value_size,
                         lam=lam, gamma=gamma, port=port, save_models=save_models)

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
    logger.statistics(str(avg_reward))


def federated_server():
    # Setup
    parser = AggregatorServerParser()
    environment, render_mode = parser.get_environment()
    env = gym.make(environment, render_mode=render_mode)
    save_models = parser.args.save_models
    save_each_epoch = parser.args.save_each_epoch
    save_client_models = parser.args.save_client_models
    n_episodes = int(parser.args.n_episodes)
    port = int(parser.args.port)
    n_clients = int(parser.args.n_agents)
    steps_per_epoch = int(parser.args.steps_per_epoch)
    n_epochs = int(parser.args.n_epochs)

    logger = CustomLogger()
    print(type(env.observation_space))
    print(type(env.action_space))
    agg_server = AggregatorServer(obs_dim=env.observation_space, act_dim=env.action_space,
                                  n_clients=n_clients, n_episodes=n_episodes, port=port,
                                  save_each_epoch=save_each_epoch, save_client_models=save_client_models,
                                  save_models=save_models)

    training_configuration = {
        'env': env.unwrapped.spec.id,
        'n_episodes': n_episodes,
        'n_epochs': n_epochs,
        'steps_per_epoch': steps_per_epoch,
    }

    final_episode = 0

    # send training config
    ack = agg_server.send__rcv_payload(training_configuration)

    try:
        for i in range(n_epochs):
            # send nets
            logger.info(f'Training epoch {i+1} of {n_epochs}.')
            logger.info('sending models..')
            payload = (agg_server.policy.p_net.state_dict(),
                       agg_server.value.v_net.state_dict())
            models = agg_server.send__rcv_payload(payload)
            logger.info('models sent!')
            client_models = []
            p_models = []
            v_models = []
            episode = (i+1) * n_episodes

            for index, (x, y) in enumerate(models):
                p_models.append(x)
                v_models.append(y)
                if agg_server.save_client_models is True:
                    client_models.append((x, f'client{(index+1)}_p_{episode}'))
                    client_models.append((y, f'client{index+1}_v_{episode}'))

            logger.info('aggregating...')
            agg_server.aggregate(p_models, v_models)

            if agg_server.save_each_epoch is True:
                logger.info("saving this epoch's aggregated model..")
                agg_server.save_aggregate(
                    p_save=f'fed_p_{episode}', v_save=f'fed_v_{episode}')
                logger.info('saved .')

            if agg_server.save_client_models is True:
                logger.info("saving clients' models")
                save_m(client_models)
                logger.info('save client success')

            agg_server.send_ack()
            final_episode = (i+1) * n_episodes

    except ConnectionError:
        CustomLogger().error("Connection error!")
    except Exception as e:
        CustomLogger().error(f"Different error!\n {str(e)}")
    finally:
        # print(value.state_dict().items())
        logger.info('closing ..')
        agg_server.close(final_episode)


def federated_client():
    logger, env, agent, n_episodes, steps_per_epoch = setup_agentclient(
        AgentClientParser)

    # get training config
    agent.receive_config()
    logger.info(agent.training_config.items())
    agent.configure_training()
    agent.send_ack()

    try:
        for i in range(agent.n_epochs):
            logger.info(f'training epoch {i+1} of {agent.n_epochs}')
            try:
                logger.info('waiting to receive models..')
                agent.receive_models()
                logger.info('models received!')
            except ConnectionError:
                model = agent.ask_for_models()
                logger.info(model)
                logger.info('received resend model')

            logger.info('recieved updated model.')
            agent.train()
            ack = agent.send_models()

    except Exception as e:
        logger.error(f'Error:\n {type(e).__name__}: {str(e)}')
    finally:
        agent.close()


class RandomModel(nn.Module):
    def __init__(self):
        super(RandomModel, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def io_test():
    obs = zeros_box_space(3)
    act = zeros_box_space(3)

    agent = Agent(obs, act)
    agent2 = Agent(obs, act)


    p_net = agent.policy.p_net.state_dict()
    v_net = agent.value.v_net.state_dict()
    # print(p_net)


    json_p, json_v = agent.state_dicts_to_json()
    print(json_p)
    p_net, v_net = agent.json_to_state_dicts(json_p, json_v)
    agent2.load_state_dicts(p_net, v_net)



    p_net = agent2.policy.p_net.state_dict()
    v_net = agent2.value.v_net.state_dict()
    print(p_net)