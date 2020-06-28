# d4rl-pybullet
Datasets for Data-Driven Deep Reinforcement Learning with Pybullet environments.
This work is intending to provide datasets for data-driven deep reinforcement learning with open-sourced bullet simulator, which encourages more people to join this community.

This repository is built on top of [d4rl](https://github.com/rail-berkeley/d4rl).
However, currently, it is impossible to import d4rl without checking MuJoCo activation keys, which fails the program.
Thus, `d4rl_pybullet.offline_env` is directly copied from `https://github.com/rail-berkeley/d4rl/blob/1899859e3ebdac8f587abbe9cb1663761be69141/d4rl/offline_env.py`.

## usage
TODO.

## supported environments
TODO.

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
