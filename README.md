# d4rl-pybullet
Datasets for Data-Driven Deep Reinforcement Learning with Pybullet environments.

This work is intending to provide datasets for data-driven deep reinforcement learning with open-sourced simulator, which encourage more people to join this community.
This repository is built on top of [d4rl](https://github.com/rail-berkeley/d4rl/tree/1899859e3ebdac8f587abbe9cb1663761be69141).

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
