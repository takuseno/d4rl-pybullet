import numpy as np
import gym
import pybullet_envs
import os
import pickle
import argparse

from datetime import datetime
from .sac import SAC, seed_everything
from .logger import SimpleLogger


def update(buffer, sac, batch_size):
    obs_ts = []
    act_ts = []
    rew_tp1s = []
    obs_tp1s = []
    ter_tp1s = []
    while len(obs_ts) != batch_size:
        index = np.random.randint(len(buffer) - 1)
        # skip if index indicates the terminal state
        if buffer[index][3]:
            continue
        obs_ts.append(buffer[index][0])
        act_ts.append(buffer[index][1])
        rew_tp1s.append(buffer[index + 1][2])
        obs_tp1s.append(buffer[index + 1][0])
        ter_tp1s.append(buffer[index + 1][3])

    critic_loss = sac.update_critic(obs_ts, act_ts, rew_tp1s, obs_tp1s,
                                    ter_tp1s)

    actor_loss = sac.update_actor(obs_ts)

    temp_loss = sac.update_temp(obs_ts)

    sac.update_target()

    return critic_loss, actor_loss, temp_loss


def train(env,
          sac,
          logdir,
          final_step=200000,
          batch_size=100,
          save_interval=10000):
    logger = SimpleLogger(logdir)

    buffer = []

    step = 0
    while step < final_step:
        obs_t = env.reset()
        ter_t = False
        rew_t = 0.0
        episode_rew = 0.0
        while step < final_step and not ter_t:
            act_t = sac.act([obs_t])[0]

            buffer.append([obs_t, act_t, [rew_t], [ter_t]])

            obs_t, rew_t, ter_t, _ = env.step(act_t)

            episode_rew += rew_t
            step += 1

            if len(buffer) > batch_size:
                update(buffer, sac, batch_size)

            if step % save_interval == 0:
                sac.save(os.path.join(logdir, 'model_%d.pt' % step))

        logger.add('reward', step, episode_rew)

    # save final buffer
    with open(os.path.join(logdir, 'buffer.pkl'), 'wb') as f:
        observations = []
        actions = []
        rewards = []
        terminals = []
        for i in range(len(buffer)):
            observations.append(buffer[i][0])
            actions.append(buffer[i][1])
            rewards.append(buffer[i][2])
            terminals.append(buffer[i][3])

        observations = np.array(observations, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        terminals = np.array(terminals, dtype=np.float32)

        pickle.dump((observations, actions, rewards, terminals), f)

    # save final parameters
    sac.save(os.path.join(logdir, 'final_model.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--final-step', type=float, default=200000)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    env = gym.make(args.env)
    env.seed(args.seed)
    seed_everything(args.seed)

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    device = 'gpu:0' if args.gpu else 'cpu:0'

    sac = SAC(observation_size, action_size, device)

    logdir = os.path.join('logs', '{}_{}'.format(args.env, args.seed))
    os.makedirs(logdir)

    train(env, sac, logdir, args.final_step)
