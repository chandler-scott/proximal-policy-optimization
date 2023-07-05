import argparse
from typing import Tuple

N_EPISODES = 100


class Parser:
    """
    Custom argument parser to handle command line arguments.
    """

    def __init__(self) -> None:
        self.envs = {
            'lunar': 'LunarLander-v2',
            'walker': 'BipedalWalker-v3',
            'cartpole': 'CartPole-v1',
        }

        self.parser = argparse.ArgumentParser(
            description='Training a proximal policy optimization agent')
        self.parser.add_argument('-e', '--env', help='Environment to train ons',
                                 choices=self.envs.keys(), default='cartpole')
        self.parser.add_argument(
            '-n', '--n_episodes', help='Number of episodes to train', default=N_EPISODES)
        self.parser.add_argument(
            '-l', '--logs', help='File name for log output', default=None)
        self.parser.add_argument(
            '-s', '--stats', help='File name for statistics output', default=None)

        self.args = self.parser.parse_args()

    def get_environment(self) -> str:
        env = self.args.env

        if env in self.envs:
            return self.envs[env]
        else:
            raise Exception(
                'Environment not found in the dictionary. Use --help for reference.')

    def get_n_episodes(self) -> int:
        return self.args.n_episodes

    def get_logs_and_stats(self) -> Tuple[str, str]:
        return (self.args.logs, self.args.stats)


class TrainingParser(Parser):
    """
    Custom argument parser to handle command line arguments for an agent training.
    """

    def __init__(self) -> None:
        super().__init__()


class PlayingParser(Parser):
    """
    Custom argument parser to handle command line arguments for an agent playing.
    """

    def __init__(self) -> None:
        super().__init__()
