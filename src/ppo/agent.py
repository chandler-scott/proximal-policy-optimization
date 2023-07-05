from typing import Tuple, Any
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Box, Discrete
from util.logger import CustomLogger
from .buffer import Buffer


def build_network(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


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

        self.logits_network = build_network(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        logits = self.logits_network(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)


class GaussianActor(Actor):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        # create network to handle mean values of gaussian distribution of continuous actions
        self.mu_net = build_network(
            [obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
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


class Agent:
    def __init__(self, observations_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh,
                 policy_lr=3e-4, value_lr=1e-3,
                 local_steps_per_epoch=50,  gamma=0.99, lam=0.95) -> None:
        obs_dim = observations_space.shape[0]

        # policy builder depends on action space..
        if isinstance(action_space, Box):
            self.policy = GaussianActor(
                obs_dim, action_space.shape[0], hidden_sizes, activation)
        else:
            self.policy = CategoricalActor(
                obs_dim, action_space.n, hidden_sizes, activation)

        self.value = Critic(obs_dim, hidden_sizes, activation)
        self.buffer = Buffer(obs_dim, action_space.shape,
                             local_steps_per_epoch, gamma, lam)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=value_lr)

    
    def save_models():
        pass

    def load_models():
        pass

    def step(self, observation) -> Tuple[Any, Any, Any]:
        with torch.no_grad():
            policy_distribution = self.policy._distribution(observation)
            actions = policy_distribution.sample()
            logp_actions = self.policy._log_prob_from_distribution(
                policy_distribution, actions)
            value = self.value(observation)
        return actions.numpy(), value.numpy(), logp_actions.numpy()

    def act(self, observation):
        return self.step(observation)[0]

    def compute_loss_policy(self, data, clip_ratio=0.2):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.policy(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1+clip_ratio) | ratio.lt(1-clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_value(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.value(obs) - ret)**2).mean()

    def learn(self,  train_policy_iters=80, train_value_iters=80, target_kl=0.01) -> None:
        data = self.buffer.get()

        policy_loss_old, policy_info_old = self.compute_loss_policy(data)
        policy_loss_old = policy_loss_old.item()
        value_loss_old = self.compute_loss_value(data).item()

        # Train policy with multiple steps of gradient descent
        for i in range(train_policy_iters):
            self.policy_optimizer.zero_grad()
            loss_pi, policy_info = self.compute_loss_policy(data)
            kl = policy_info['kl']

            if kl > 1.5 * target_kl:
                #CustomLogger().info('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.policy_optimizer.step()

        # Value function learning
        for i in range(train_value_iters):
            self.value_optimizer.zero_grad()
            loss_v = self.compute_loss_value(data)
            loss_v.backward()
            self.value_optimizer.step()

    def print_network(self, message, network):
        '''
        Print the neural network for debugging purposes
        '''
        CustomLogger().debug(f"{message}")
        for i, layer in enumerate(network):
            if isinstance(layer, torch.nn.Linear):
                # Print the weight and bias of each Linear layer
                CustomLogger().debug(f"Layer {i}:")
                CustomLogger().debug(f"Weight: {layer.weight}")
                CustomLogger().debug(f"Bias: {layer.bias}")
            elif isinstance(layer, torch.nn.Tanh):
                CustomLogger().debug(f"Layer {i}: Tanh activation")
            elif isinstance(layer, torch.nn.Identity):
                CustomLogger().debug(f"Layer {i}: Identity activation")
            else:
                CustomLogger().debug(f"Layer {i}: Unknown layer type")
