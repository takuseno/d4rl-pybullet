![MIT](https://img.shields.io/badge/license-MIT-blue)
# d4rl-pybullet
Datasets for Data-Driven Deep Reinforcement Learning with Pybullet environments.
This work is intending to provide datasets for data-driven deep reinforcement learning with open-sourced bullet simulator, which encourages more people to join this community.

This repository is built on top of [d4rl](https://github.com/rail-berkeley/d4rl).
However, currently, it is impossible to import d4rl without checking MuJoCo activation keys, which fails the program.
Thus, `d4rl_pybullet.offline_env` is directly copied from [d4rl repository](https://github.com/rail-berkeley/d4rl/blob/1899859e3ebdac8f587abbe9cb1663761be69141/d4rl/offline_env.py).

## usage
TODO.

## supported environments
- `random` denotes datasets sampled with a randomly initialized policy.
- `medium` denotes datasets sampled with a medium-level policy.
- `mixed` denotes datasets collected during policy training.

| id | task |
|:-|:-:|
| hopper-bullet-random-v0 | HopperBulletEnv-v0 |
| hopper-bullet-medium-v0 | HopperBulletEnv-v0 |
| hopper-bullet-mixed-v0 | HopperBulletEnv-v0 |

## train policy
You can train Soft Actor-Critic policy on your own machine.
```
# giving -g option to enable GPU
$ ./scripts/train -e HopperBulletEnv-v0 -g
```

## data collection
You can collect datasets with the trained policy and random policy.
```
$ ./scripts/collect -e HopperbulletEnv-v0 -g
```

## contribution
Welcome to contributions!!

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
