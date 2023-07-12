import asyncio
import sys
from .model_utils import *
from .socket_util import Server
from ppo import GaussianActor, CategoricalActor, Critic
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete


class Aggregator:
    def __init__(self, obs_dim, act_dim, p_load=None, p_save=None,
                 v_load=None, v_save=None,
                 n_agents=1, n_episodes=100,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh,) -> None:
        self.p_load = p_load
        self.p_save = p_save
        self.v_load = v_load
        self.v_save = v_save
        self.n_agents = n_agents
        self.n_episodes = n_episodes
        self.create_networks(
            act_dim, obs_dim.shape[0], hidden_sizes, activation)
        if self.v_load is not None and self.p_load is not None:
            self.load()

        super(Aggregator, self).__init__()

    def create_networks(self, action_space, obs_dim, hidden_sizes, activation):
        # create policy
        if isinstance(action_space, Box):
            self.policy = GaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation)
            self.policy_copy = GaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation)
        else:
            self.policy = CategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation)
            self.policy_copy = GaussianActor(
                obs_dim, action_space.n, hidden_sizes, activation)
        # create value
        self.value = Critic(obs_dim, hidden_sizes, activation)
        self.value_copy = Critic(obs_dim, hidden_sizes, activation)

    def load(self):
        print('loading models...')
        load_models({
            self.policy.p_net: self.p_load,
            self.value.v_net: self.v_load,
        })

    def save(self):
        self.policy = save_models({
            self.policy.p_net: self.p_save,
            self.value.v_net: self.v_save,
        })

    def aggregate(self, policy_models, value_models):
        self.policy.p_net = aggregate_models(
            policy_models, self.policy_copy.p_net)
        self.value.v_net = aggregate_models(
            value_models, self.value_copy.v_net)
        return self.policy.p_net, self.value.v_net

    def close(self):
        if self.v_save is not None and self.p_save is not None:
            self.save()


class AggregatorServer(Aggregator, Server):
    def __init__(self, act_dim, obs_dim,
                 p_load=None, p_save=None,
                 v_load=None, v_save=None,
                 n_clients=2, n_episodes=100, port=1234):
        super(AggregatorServer, self).__init__(
            n_episodes=n_episodes, n_agents=n_clients,
            p_load=p_load, p_save=p_save,
            v_load=v_load, v_save=v_save,
            obs_dim=obs_dim,
            act_dim=act_dim
        )
        self.n_clients = n_clients
        self.port = port
        self.setup()

    def receive_models(self):
        models = self.receive()
        if (models is None):
            raise ConnectionError

        self.send('ACK')
        p_models = []
        v_models = []
        for x, y in models:
            p_models.append(x)
            v_models.append(y)
        return p_models, v_models

    def close(self):
        super().close()
