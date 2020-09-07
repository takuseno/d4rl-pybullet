![format check](https://github.com/takuseno/d4rl-pybullet/workflows/format%20check/badge.svg)
![test](https://github.com/takuseno/d4rl-pybullet/workflows/test/badge.svg)
![MIT](https://img.shields.io/badge/license-MIT-blue)
[![Gitter](https://img.shields.io/gitter/room/d3rlpy/d4rl-pybullet)](https://gitter.im/d3rlpy/d4rl-pybullet)

# d4rl-pybullet
Datasets for Data-Driven Deep Reinforcement Learning with Pybullet environments.
This work is intending to provide datasets for data-driven deep reinforcement
learning with open-sourced bullet simulator, which encourages more people to
join this community.

This repository is built on top of [d4rl](https://github.com/rail-berkeley/d4rl).
However, currently, it is impossible to import d4rl without checking MuJoCo
activation keys, which fails the program.
Thus, `d4rl_pybullet.offline_env` is directly copied from [d4rl repository](https://github.com/rail-berkeley/d4rl/blob/1899859e3ebdac8f587abbe9cb1663761be69141/d4rl/offline_env.py).

## install
```
$ pip install git+https://github.com/takuseno/d4rl-pybullet
```

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
| halfcheetah-bullet-random-v0 | HalfCheetahBulletEnv-v0 | -1304.49 | 99.30 | -945.29 | -1518.58 | 1000000 |
| halfcheetah-bullet-medium-v0 | HalfCheetahBulletEnv-v0 | 787.35 | 104.31 | 844.91 | -522.57 | 1000000 |
| halfcheetah-bullet-mixed-v0 | HalfCheetahBulletEnv-v0 | 453.12 | 498.19 | 801.02 | -1428.22 | 178178 |
| ant-bullet-random-v0 | AntBulletEnv-v0 | 10.35 | 0.31 | 13.04 | 9.82 | 1000000 |
| ant-bullet-medium-v0 | AntBulletEnv-v0 | 570.80 | 104.82 | 816.79 | 70.87 | 1000000 |
| ant-bullet-mixed-v0 | AntBulletEnv-v0 | 255.40 | 196.22 | 609.66 | -32.74 | 53572 |
| walker2d-bullet-random-v0 | Walker2DBulletEnv-v0 | 14.98 | 2.94 | 66.90 | 5.73 | 1000000 |
| walker2d-bullet-medium-v0 | Walker2DBulletEnv-v0 | 1106.68 | 417.79 | 1394.38 | 16.00 | 1000000 |
| walker2d-bullet-mixed-v0 | Walker2DBulletEnv-v0 | 181.51 | 277.71 | 1363.94 | 9.45 | 89772 |

## train policy
You can train Soft Actor-Critic policy on your own machine.
```
# giving -g option to choose GPU device
$ ./scripts/train -e HopperBulletEnv-v0 -g 0 -n 1
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
