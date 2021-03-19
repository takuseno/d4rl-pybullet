import numpy as np
import gym
import pybullet_envs
import os
import pickle
import argparse

from datetime import datetime
from .sac import SAC, seed_everything
from .logger import SimpleLogger
from .utility import save_buffer


def update(buffer, sac, batch_size):
    obs_ts = []
    act_ts = []
    rew_tp1s = []
    obs_tp1s = []
    ter_tp1s = []
    while len(obs_ts) != batch_size:
        index = np.random.randint(len(buffer) - 1)

        # skip if index indicates the terminal state
        if buffer[index][3][0]:
            continue

        obs_ts.append(buffer[index][0])
        act_ts.append(buffer[index][1])
        rew_tp1s.append(buffer[index + 1][2])
        obs_tp1s.append(buffer[index + 1][0])

        # take timeout into account
        terminal = buffer[index + 1][3][0]
        timeout = buffer[index + 1][4][0]
        ter_tp1s.append([terminal and not timeout])

    critic_loss = sac.update_critic(obs_ts, act_ts, rew_tp1s, obs_tp1s,
                                    ter_tp1s)

    actor_loss = sac.update_actor(obs_ts)

    temp_loss = sac.update_temp(obs_ts)

    sac.update_target()

    return critic_loss, actor_loss, temp_loss


def evaluate(env, sac, n_episodes=10):
    episode_rews = []
    for episode in range(n_episodes):
        obs = env.reset()
        ter = False
        episode_rew = 0.0
        while not ter:
            act = sac.act([obs], deterministic=True)[0]
            obs, rew, ter, _ = env.step(act)
            episode_rew += rew
        episode_rews.append(episode_rew)
    return np.mean(episode_rews)


def train(env,
          eval_env,
          sac,
          logdir,
          desired_level,
          total_step,
          batch_size=100,
          save_interval=10000,
          eval_interval=10000):
    logger = SimpleLogger(logdir)

    buffer = []

    step = 0
    while step < total_step:
        obs_t = env.reset()
        ter_t = False
        rew_t = 0.0
        timeout = False
        episode_rew = 0.0
        while not ter_t and step < total_step:
            act_t = sac.act([obs_t])[0]

            buffer.append([obs_t, act_t, [rew_t], [ter_t], [timeout]])

            obs_t, rew_t, ter_t, info = env.step(act_t)

            timeout = "TimeLimit.truncated" in info
            episode_rew += rew_t
            step += 1

            if len(buffer) > batch_size:
                update(buffer, sac, batch_size)

            if step % save_interval == 0:
                sac.save(os.path.join(logdir, 'model_%d.pt' % step))

            if step % eval_interval == 0:
                logger.add('eval_reward', step, evaluate(eval_env, sac))

        if ter_t:
            dummy_action = np.zeros_like(act_t)
            buffer.append([obs_t, dummy_action, [rew_t], [ter_t], [timeout]])

        logger.add('reward', step, episode_rew)

        if desired_level is not None and episode_rew >= desired_level:
            break

    # save final buffer
    save_buffer(buffer, logdir)
    print('Final buffer has been saved.')

    # save final parameters
    sac.save(os.path.join(logdir, 'final_model.pt'))
    print('Final model parameters have been saved.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--desired-level', type=float)
    parser.add_argument('--total-step', type=int, default=2000000)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    env = gym.make(args.env)
    eval_env = gym.make(args.env)
    env.seed(args.seed)
    seed_everything(args.seed)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = 'cuda:%d' % args.gpu if args.gpu is not None else 'cpu:0'

    sac = SAC(observation_size, action_size, device)

    logdir = os.path.join('logs', '{}_{}'.format(args.env, args.seed))
    os.makedirs(logdir)

    train(env, eval_env, sac, logdir, args.desired_level, args.total_step)
