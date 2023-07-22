import pickle
from util.model_utils import *
from util.socket_util import Server, send_receive_with_timeout
from ppo import GaussianActor, CategoricalActor, Critic
import torch
import torch.nn as nn
from gymnasium.spaces import Box, Discrete


class Aggregator:
    def __init__(self, obs_dim, act_dim, p_load=None,
                 v_load=None, save_models=False,
                 n_agents=1, n_episodes=100,
                 hidden_sizes=(64, 64),
                 activation=nn.Tanh,) -> None:
        self.p_load = p_load
        self.v_load = v_load
        self.save_models = save_models
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

    def load(self, p_load=None, v_load=None):
        p_file = 'fed_p' if p_load is None else p_load
        v_file = 'fed_v' if v_load is None else v_load
        load_models({
            self.policy.p_net: p_file,
            self.value.v_net: v_file,
        })

    def save_aggregate(self, p_save=None, v_save=None, episode=None):
        p_file = 'fed_p' if episode is None else f'fed_p_{episode}'
        v_file = 'fed_v' if episode is None else f'fed_v_{episode}'

        save_models([
            (self.policy.p_net.state_dict(), p_save if p_save is not None else p_file),
            (self.value.v_net.state_dict(), v_save if v_save is not None else v_file)
        ])

    def aggregate(self, policy_models:list, value_models:list):
        self.policy.p_net = aggregate_models(
            policy_models, self.policy_copy.p_net)
        self.value.v_net = aggregate_models(
            value_models, self.value_copy.v_net)
        return self.policy.p_net, self.value.v_net

    def close(self, episode=None):
        if self.save_models is True:
            self.save_aggregate(episode=episode)
        print('closed.')


class AggregatorServer(Aggregator, Server):
    def __init__(self, act_dim, obs_dim,
                 p_load=None, p_save=None,
                 v_load=None, v_save=None,
                 n_clients=2, n_episodes=100,
                 port=1234, save_each_epoch=False,
                 save_client_models=False, save_models=False):
        super(AggregatorServer, self).__init__(
            n_episodes=n_episodes, n_agents=n_clients,
            p_load=p_load, save_models=save_models,
            v_load=v_load, obs_dim=obs_dim,
            act_dim=act_dim
        )
        self.save_each_epoch = save_each_epoch
        self.save_client_models = save_client_models
        self.n_clients = n_clients
        self.port = port
        self.setup()

    def send__rcv_payload(self, payload):
        return send_receive_with_timeout(lambda: self.send(payload),
                                         self.receive, timeout=1, max_retries=3)

    def send_ack(self):
        payload = 'ACK'
        self.send(payload)

    def receive_models(self, timeout=5):
        models = self.receive(timeout=timeout)
        if (models is None):
            raise ConnectionError

        self.send('ACK')
        p_models = []
        v_models = []
        for x, y in models:
            p_models.append(x)
            v_models.append(y)
        return p_models, v_models

    def close(self, episode=None):
        super().close(episode=episode)
