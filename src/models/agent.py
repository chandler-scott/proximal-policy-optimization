from util.socket_util import *
from util import CustomLogger
from torch.optim import Adam
from torch.distributions.categorical import Categorical
from gymnasium.spaces import Box, Discrete
from ppo.actor_critic import *
from .base import BaseModel

import torch


class Agent(BaseModel):
    def __init__(self, observation_space, action_space,
                 policy_hidden_sizes=(64, 64), value_hidden_sizes=(64, 64),
                 activation=nn.Tanh,
                 policy_lr=3e-4, value_lr=1e-3,
                 policy_load_file=None,
                 save_models=False, value_load_file=None,
                 local_steps_per_epoch=50,  gamma=0.99, lam=0.95) -> None:
        super(Agent, self).__init__()
        obs_dim = observation_space.shape[0]
        self.buffer = Buffer(obs_dim, action_space.shape,
                             local_steps_per_epoch, gamma, lam)

        if save_models is True:
            self.policy_save_file, self.value_save_file = 'p_save', 'v_save'

        self.save_models = save_models

        # if load file not specified, create neural networks from scratch
        self.create_networks(action_space, obs_dim, policy_hidden_sizes,
                             value_hidden_sizes, activation)
        # else, load our file
        if (policy_load_file is not None):
            self.load_models_from_file(policy_load_file, value_load_file)

        self.policy_optimizer = Adam(self.policy.parameters(), lr=policy_lr)
        self.value_optimizer = Adam(self.value.parameters(), lr=value_lr)

    def save_networks(self):
        CustomLogger().info('Saving value and policy networks..')
        save_models([
            (self.policy.p_net.state_dict(), self.policy_save_file),
            (self.value.v_net.state_dict(), self.value_save_file)
        ])

    def load_state_dicts(self, p_state_dict, v_state_dict):
        self.policy.p_net.load_state_dict(p_state_dict)
        self.value.v_net.load_state_dict(v_state_dict)

    def load_models_from_file(self, policy_load_file, value_load_file):
        CustomLogger().info('Loading value and policy networks..')
        load_models({
            self.policy.p_net: policy_load_file,
            self.value.v_net: value_load_file
        })

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
                # CustomLogger().info('Early stopping at step %d due to reaching max kl.' % i)
                break
            loss_pi.backward()
            self.policy_optimizer.step()

        # Value function learning
        for i in range(train_value_iters):
            self.value_optimizer.zero_grad()
            loss_v = self.compute_loss_value(data)
            loss_v.backward()
            self.value_optimizer.step()

    def close(self):
        if (self.save_models):
            self.save_networks()


class AgentClient(Agent, Client):
    def __init__(self, observation_space, action_space,
                 policy_hidden_sizes=(64, 64), value_hidden_sizes=(64, 64),
                 activation=nn.Tanh,
                 policy_lr=3e-4, value_lr=1e-3,
                 policy_load_file=None, value_load_file=None,
                 save_models=save_models,
                 local_steps_per_epoch=50,  gamma=0.99, lam=0.95, port=1234):
        super(AgentClient, self).__init__(
            observation_space=observation_space, action_space=action_space,
            policy_hidden_sizes=policy_hidden_sizes, value_hidden_sizes=value_hidden_sizes,
            activation=activation,
            policy_lr=policy_lr, value_lr=value_lr,
            save_models=save_models, policy_load_file=policy_load_file,
            value_load_file=value_load_file,
            local_steps_per_epoch=local_steps_per_epoch,  gamma=gamma, lam=lam
        )
        self.port = port
        self.setup()

    def receive_config(self):
        self.training_config = self.receive()
        return self.training_config

    def configure_training(self):
        self.env = gym.make(self.training_config['env'])
        self.n_episodes = self.training_config['n_episodes']
        self.n_epochs = self.training_config['n_epochs']
        self.steps_per_epoch = self.training_config['steps_per_epoch']

    def receive_models(self):
        p_net, v_net = self.receive()
        self.load_state_dicts(p_net, v_net)

    def send_models(self):
        payload = (self.policy.p_net.state_dict(),
                   self.value.v_net.state_dict())
        return send_receive_with_timeout(lambda: self.send(payload),
                                         self.receive, timeout=1, max_retries=3)

    def send_payload(self, payload):
        return send_receive_with_timeout(lambda: self.send(payload),
                                         self.receive, timeout=1, max_retries=3)

    def send_ack(self):
        payload = 'ACK'
        self.send(payload)

    def train(self):
        env = self.env
        n_episodes = self.n_episodes
        steps_per_epoch = self.steps_per_epoch

        observation, cumulative_reward = env.reset()[0], 0
        for i in range(n_episodes):
            print(f'episodes {i+1} of {n_episodes}')

            for t in range(steps_per_epoch):
                # run policy in env for T timesteps
                action, value, logp = self.step(torch.as_tensor(
                    np.array(observation), dtype=torch.float32))

                next_observation, reward, terminated, truncated, _ = env.step(
                    action)
                done = terminated or truncated
                cumulative_reward += reward

                # save and log
                self.buffer.store(observation, action, reward, value, logp)

                # update observation
                observation = next_observation

                epoch_ended = t == steps_per_epoch-1

                if done or epoch_ended:
                    if truncated or epoch_ended:
                        _, v, _ = self.step(torch.as_tensor(
                            np.array(observation), dtype=torch.float32))
                        # logger.info('truncated' if truncated else 'epoch ended')
                    else:
                        v = 0
                    # compute advantage estimates
                    self.buffer.finish_path(v)
                    observation, cumulative_reward = env.reset()[0], 0

            # learn
            self.learn()

    def close(self):
        super().close()
