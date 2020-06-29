import pytest

import gym
import d4rl_pybullet


@pytest.mark.parametrize('name', [
    'hopper-bullet-mixed-v0', 'halfcheetah-bullet-mixed-v0',
    'ant-bullet-mixed-v0'
])
def test_env(name):
    env = gym.make(name)

    dataset = env.get_dataset()
    assert 'observations' in dataset
    assert 'actions' in dataset
    assert 'rewards' in dataset
    assert 'terminals' in dataset
