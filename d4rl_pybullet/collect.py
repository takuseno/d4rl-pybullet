import numpy as np
import gym
import pybullet_envs
import argparse
import os

from tqdm import tqdm
from .sac import SAC, seed_everything
from .utility import save_buffer


def collect(env, sac, logdir, final_step, deterministic):
    buffer = []

    step = 0
    pbar = tqdm(total=final_step)
    while step < final_step:
        obs_t = env.reset()
        ter_t = False
        rew_t = 0.0
        timeout = False
        while step < final_step and not ter_t:
            act_t = sac.act([obs_t], deterministic=deterministic)[0]

            buffer.append([obs_t, act_t, [rew_t], [ter_t], [timeout]])

            obs_t, rew_t, ter_t, info = env.step(act_t)

            timeout = "TimeLimit.truncated" in info
            step += 1
            pbar.update(1)

        if ter_t:
            dummy_action = np.zeros_like(act_t)
            buffer.append([obs_t, dummy_action, [rew_t], [ter_t], [timeout]])

    save_buffer(buffer[:final_step], logdir)
    print('Collected data has been saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--final-step', type=int, default=1000000)
    parser.add_argument('--load', type=str)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(args.seed)
    seed_everything(args.seed)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = 'cuda:0' if args.gpu else 'cpu:0'

    sac = SAC(observation_size, action_size, device)

    if args.load:
        sac.load(args.load)
        name = 'medium'
        deterministic = True
    else:
        name = 'random'
        deterministic = False

    logdir = os.path.join('logs', '{}_{}_{}'.format(args.env, name, args.seed))
    os.makedirs(logdir)

    collect(env, sac, logdir, args.final_step, deterministic)
