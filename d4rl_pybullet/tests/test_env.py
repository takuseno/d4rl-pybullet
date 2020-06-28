import gym
import d4rl_pybullet


def test_env():
    env = gym.make('hopper-bullet-mixed-v0')

    dataset = env.get_dataset()
    assert 'observations' in dataset
    assert 'actions' in dataset
    assert 'rewards' in dataset
    assert 'terminals' in dataset
