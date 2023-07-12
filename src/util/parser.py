import argparse
from typing import Tuple

N_EPISODES = 100
N_AGENTS = 1
STEPS_PER_EPOCH = 4000
LAMBDA = 0.95
GAMMA = 0.99
POLICY_LR = 3e-4
POLICY_HIDDEN_SIZE = (64, 64)
VALUE_LR = 1e-3
VALUE_HIDDEN_SIZE = (64, 64)


class Parser:
    def __init__(self) -> None:
        pass


class AggregatorParser(Parser):
    def __init__(self) -> None:
        self.envs = {
            'lunar': 'LunarLander-v2',
            'walker': 'BipedalWalker-v3',
            'cartpole': 'CartPole-v1',
            'acrobot': 'Acrobot-v1',
            'test': 'TestWorld-v0'
        }
        parser = argparse.ArgumentParser(
            description='Aggregator for combining models')
        parser.add_argument('-e', '--env', help='Environment to train ons',
                            choices=self.envs.keys(), default='cartpole')
        parser.add_argument('-r', '--render', help='Render mode',
                            choices=["human", "None"], default='None')
        # neural network save/load
        parser.add_argument(
            '-p_save', '--policy_save_file', help='Policy neural network save file', default='policy')
        parser.add_argument(
            '-p_load', '--policy_load_file', help='Policy neural network load file',
            default=None)
        parser.add_argument(
            '-v_save', '--value_save_file', help='Value neural network save file', default='value')
        parser.add_argument(
            '-v_load', '--value_load_file', help='Value neural network load file',
            default=None)
        parser.add_argument(
            '-n', '--n_episodes', help='Number of episodes to train', default=N_EPISODES)
        parser.add_argument(
            '-a', '--n_agents', help='Number of agents training', default=N_AGENTS)
        parser.add_argument(
            '-p', '--port', help='Port Number', default=1234)
        self.args = parser.parse_args()
    

    def get_environment(self) -> str:
        env = self.args.env

        if env in self.envs:
            return self.envs[env], self.args.render
        else:
            raise Exception(
                'Environment not found in the dictionary. Use --help for reference.')


class AgentParser(Parser):
    """
    Custom argument parser to handle command line arguments.
    """

    def __init__(self) -> None:
        self.envs = {
            'lunar': 'LunarLander-v2',
            'walker': 'BipedalWalker-v3',
            'cartpole': 'CartPole-v1',
            'acrobot': 'Acrobot-v1',
            'test': 'TestWorld-v0'
        }

        parser = argparse.ArgumentParser(
            description='Training a proximal policy optimization agent')
        parser.add_argument('-e', '--env', help='Environment to train ons',
                            choices=self.envs.keys(), default='cartpole')
        parser.add_argument('-r', '--render', help='Render mode',
                            choices=["human", "None"], default='None')
        parser.add_argument(
            '-n', '--n_episodes', help='Number of episodes to train', default=N_EPISODES)
        parser.add_argument(
            '-s', '--steps_per_epoch', help='''Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.''', default=STEPS_PER_EPOCH)

        # buffer args
        parser.add_argument(
            '-g', '--gamma', help='Discount factor. (Always between 0 and 1.)', default=GAMMA)
        parser.add_argument(
            '-l', '--lam', help='''Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)''', default=LAMBDA)

        # neural network args
        parser.add_argument(
            '-p_lr', '--policy_lr', help='Policy neural network learning rate', default=POLICY_LR)
        parser.add_argument(
            '-p_size', '--policy_hidden_size', help='Policy neural network hidden size ( format: (64, 64) )',
            default=POLICY_HIDDEN_SIZE, type=tuple)
        parser.add_argument(
            '-v_lr', '--value_lr', help='Value neural network learning rate', default=VALUE_LR)
        parser.add_argument(
            '-v_size', '--value_hidden_size', help='Value neural network hidden size ( format: (64, 64) )',
            default=VALUE_HIDDEN_SIZE, type=tuple)

        # neural network save/load
        parser.add_argument(
            '-p_save', '--policy_save_file', help='Policy neural network save file', default=None)
        parser.add_argument(
            '-p_load', '--policy_load_file', help='Policy neural network load file',
            default=None)
        parser.add_argument(
            '-v_save', '--value_save_file', help='Value neural network save file', default=None)
        parser.add_argument(
            '-v_load', '--value_load_file', help='Value neural network load file',
            default=None)

        # logs and stats
        parser.add_argument(
            '-L', '--logs', help='File name for log output', default=None)
        parser.add_argument(
            '-S', '--stats', help='File name for statistics output', default=None)

        self.args = parser.parse_args()

        # check save/load args
        if (
            (self.args.policy_load_file is not None and self.args.value_load_file is None) or
            (self.args.policy_load_file is None and self.args.value_load_file is not None)
        ):
            raise Exception(
                'Load file must be specified for both value and policy networks or neither.')
        if (
            (self.args.policy_save_file is not None and self.args.value_save_file is None) or
            (self.args.policy_save_file is None and self.args.value_save_file is not None)
        ):
            raise Exception(
                'Save file must be specified for both value and policy networks or neither.')

    def get_environment(self) -> str:
        env = self.args.env

        if env in self.envs:
            return self.envs[env], self.args.render
        else:
            raise Exception(
                'Environment not found in the dictionary. Use --help for reference.')


class TrainingParser(AgentParser):
    """
    Custom argument parser to handle command line arguments for an agent training.
    """

    def __init__(self) -> None:
        super().__init__()


class PlayingParser(AgentParser):
    """
    Custom argument parser to handle command line arguments for an agent playing.
    """

    def __init__(self) -> None:
        super().__init__()
