![format check](https://github.com/takuseno/d4rl-pybullet/workflows/format%20check/badge.svg)
![test](https://github.com/takuseno/d4rl-pybullet/workflows/test/badge.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)

# d4rl-pybullet
Datasets for Data-Driven Deep Reinforcement Learning with Pybullet environments.
This work is intending to provide datasets for data-driven deep reinforcement
learning with open-sourced bullet simulator, which encourages more people to
join this community.

This repository is built on top of [d4rl](https://github.com/rail-berkeley/d4rl).
However, currently, it is impossible to import d4rl without checking MuJoCo
activation keys, which fails the program.
Thus, `d4rl_pybullet.offline_env` is directly copied from [d4rl repository](https://github.com/rail-berkeley/d4rl/blob/1899859e3ebdac8f587abbe9cb1663761be69141/d4rl/offline_env.py).

## usage
The API is mostly identical to the original d4rl.
```py
import gym
import d4rl_pybullet

# dataset will be automatically downloaded into ~/.d4rl/datasets
env = gym.make('hopper-bullet-mixed-v0')

# interaction with its environment
env.reset()
env.step(env.action_space.sample())

# access to the dataset
dataset = env.get_dataset()
dataset['observations'] # observation data in N x dim_observation
dataset['actions'] # action data in N x dim_action
dataset['rewards'] # reward data in N x 1
dataset['terminals'] # terminal flags in N x 1 
```

## available datasets
- `random` denotes datasets sampled with a randomly initialized policy.
- `medium` denotes datasets sampled with a medium-level policy.
- `mixed` denotes datasets collected during policy training.

| id | task | mean reward | std reward | max reward | min reward | samples |
|:-|:-:|:-|:-|:-|:-|:-|
| hopper-bullet-random-v0 | HopperBulletEnv-v0 | 18.64 | 3.04 | 53.21 | -8.58 | 1000000 |
| hopper-bullet-medium-v0 | HopperBulletEnv-v0 | 1078.36 | 325.52 | 1238.9569 | 220.23 | 1000000 |
| hopper-bullet-mixed-v0 | HopperBulletEnv-v0 | 139.08 | 147.62 | 1019.94 | 9.15 | 59345 |
| halfcheetah-bullet-random-v0 | HalfCheetahBulletEnv-v0 | | | | | |
| halfcheetah-bullet-medium-v0 | HalfCheetahBulletEnv-v0 | | | | | |
| halfcheetah-bullet-mixed-v0 | HalfCheetahBulletEnv-v0 | | | | | |

## train policy
You can train Soft Actor-Critic policy on your own machine.
```
# giving -g option to enable GPU
$ ./scripts/train -e HopperBulletEnv-v0 -g -n 1
```

## data collection
You can collect datasets with the trained policy.
```
$ ./scripts/collect -e HopperBulletEnv-v0 -g -n 1
```

## data collection with randomly initialized policy
You can collect datasets with the random policy.
```
$ ./scripts/random_collect -e HopperBulletEnv-v0 -g -n 1
```

## contribution
Any contributions will be welcomed!!

### coding style
This repository is formatted with [yapf](https://github.com/google/yapf).
You can format the entire repository (excluding `offline_env.py`) as follows:
```
$ ./scripts/format
```

## acknowledgement
This work is supported by Information-technology Promotion Agency, Japan
(IPA), Exploratory IT Human Resources Project (MITOU Program) in the fiscal
year 2020.
