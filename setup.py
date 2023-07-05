from setuptools import setup, find_packages

setup(
    name='ppo',
    version='1.0.0',
    description='Proximal Policy Optimization implemented on Gymnasium environments.',
    author='Chandler Scott',
    author_email='scottcd1@etsu.edu',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=1.8.0',
        'torchvision>=0.9.0',
        'gymnasium[all]',
        'pygame',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'ppo.train = ppo.run:train',
            'ppo.play = ppo.run:play'
        ]
    }
)