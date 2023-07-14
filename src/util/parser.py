import argparse
from typing import Tuple

N_EPISODES = 100
N_AGENTS = 1
N_EPOCHS = 5
STEPS_PER_EPOCH = 100
LAMBDA = 0.95
GAMMA = 0.99
POLICY_LR = 3e-4
POLICY_HIDDEN_SIZE = (64, 64)
VALUE_LR = 1e-3
VALUE_HIDDEN_SIZE = (64, 64)


class Parser:
    def __init__(self) -> None:
        super(Parser, self).__init__()
        self.envs = {
            'lunar': 'LunarLander-v2',
            'walker': 'BipedalWalker-v3',
            'cartpole': 'CartPole-v1',
            'acrobot': 'Acrobot-v1',
            'test': 'TestWorld-v0'
        }
        self.parser = argparse.ArgumentParser(
            description='Parser for the program :)')

        self.parser.add_argument('-e', '--env', help='Environment to train ons',
                                 choices=self.envs.keys(), default='cartpole')
        self.parser.add_argument('-r', '--render', help='Render mode',
                                 choices=["human", "None"], default='None')

        # training args
        self.parser.add_argument(
            '-n', '--n_episodes', help='Number of episodes to train', default=N_EPISODES)
        self.parser.add_argument(
            '-N', '--n_epochs', help='Number of epochs to train', default=N_EPOCHS)
        self.parser.add_argument(
            '-s', '--steps_per_epoch', help='''Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.''', default=STEPS_PER_EPOCH)

        # file load/save
        self.parser.add_argument(
            '-p_save', '--policy_save_file', help='Policy neural network save file', default='policy')
        self.parser.add_argument(
            '-p_load', '--policy_load_file', help='Policy neural network load file',
            default=None)
        self.parser.add_argument(
            '-v_save', '--value_save_file', help='Value neural network save file', default='value')
        self.parser.add_argument(
            '-v_load', '--value_load_file', help='Value neural network load file',
            default=None)

        # logs and stats
        self.parser.add_argument(
            '-L', '--logs', help='File name for log output', default=None)
        self.parser.add_argument(
            '-S', '--stats', help='File name for statistics output', default=None)

    def get_environment(self) -> str:
        env = self.args.env

        if env in self.envs:
            return self.envs[env], self.args.render
        else:
            raise Exception(
                'Environment not found in the dictionary. Use --help for reference.')


class AggregatorServerParser(Parser):
    def __init__(self) -> None:
        super(AggregatorServerParser, self).__init__()
        self.parser.add_argument(
            '-p', '--port', help='Port Number', default=1234)
        self.parser.add_argument(
            '-a', '--n_agents', help='Number of client agents', default=1)

        self.args = self.parser.parse_args()


class AgentClientParser(Parser):
    """
    Custom argument parser to handle command line arguments for an agent training.
    """

    def __init__(self) -> None:
        super(AgentClientParser, self).__init__()
        self.parser.add_argument(
            '-p', '--port', help='Port Number', default=1234)
        self.parser.add_argument(
            '-H', '--host', help='Hostname', default='localhost')

        # buffer args
        self.parser.add_argument(
            '-g', '--gamma', help='Discount factor. (Always between 0 and 1.)', default=GAMMA)
        self.parser.add_argument(
            '-l', '--lam', help='''Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)''', default=LAMBDA)

        # neural network args
        self.parser.add_argument(
            '-p_lr', '--policy_lr', help='Policy neural network learning rate', default=POLICY_LR)
        self.parser.add_argument(
            '-p_size', '--policy_hidden_size', help='Policy neural network hidden size ( format: (64, 64) )',
            default=POLICY_HIDDEN_SIZE, type=tuple)
        self.parser.add_argument(
            '-v_lr', '--value_lr', help='Value neural network learning rate', default=VALUE_LR)
        self.parser.add_argument(
            '-v_size', '--value_hidden_size', help='Value neural network hidden size ( format: (64, 64) )',
            default=VALUE_HIDDEN_SIZE, type=tuple)

        self.args = self.parser.parse_args()


class AgentParser(Parser):
    """
    Custom argument parser to handle command line arguments.
    """

    def __init__(self) -> None:
        super(AgentParser, self).__init__()

        # buffer args
        self.parser.add_argument(
            '-g', '--gamma', help='Discount factor. (Always between 0 and 1.)', default=GAMMA)
        self.parser.add_argument(
            '-l', '--lam', help='''Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)''', default=LAMBDA)

        # neural network args
        self.parser.add_argument(
            '-p_lr', '--policy_lr', help='Policy neural network learning rate', default=POLICY_LR)
        self.parser.add_argument(
            '-p_size', '--policy_hidden_size', help='Policy neural network hidden size ( format: (64, 64) )',
            default=POLICY_HIDDEN_SIZE, type=tuple)
        self.parser.add_argument(
            '-v_lr', '--value_lr', help='Value neural network learning rate', default=VALUE_LR)
        self.parser.add_argument(
            '-v_size', '--value_hidden_size', help='Value neural network hidden size ( format: (64, 64) )',
            default=VALUE_HIDDEN_SIZE, type=tuple)

        self.args = self.parser.parse_args()


class PlayingParser(AgentParser):
    """
    Custom argument parser to handle command line arguments for an agent playing.
    """

    def __init__(self) -> None:
        super().__init__()
