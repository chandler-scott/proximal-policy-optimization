import gymnasium as gym
from typing import Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from util.logger import CustomLogger
from util.model_utils import save_models, load_models, build_network
from util.socket_util import Client
from .buffer import Buffer


class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, observation, action=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        action_distribution = self._distribution(observation)
        logp_a = None
        if action is not None:
            logp_a = self._log_prob_from_distribution(
                action_distribution, action)
        return action_distribution, logp_a


class CategoricalActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()

        self.p_net = build_network(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.p_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # create network to handle mean values of gaussian distribution of continuous actions
        self.p_net = build_network(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.p_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)


class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = build_network(
            [obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        # Critical to ensure v has right shape.
        return torch.squeeze(self.v_net(obs), -1)