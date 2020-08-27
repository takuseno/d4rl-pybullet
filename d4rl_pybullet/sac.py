import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import copy

from itertools import chain
from torch.distributions import Normal
from torch.optim import Adam


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def _squash_action(dist, raw_action):
    squashed_action = torch.tanh(raw_action)
    jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=1, keepdims=True)
    return squashed_action, log_prob


def _soft_sync(targ_model, model, tau):
    with torch.no_grad():
        params = model.parameters()
        targ_params = targ_model.parameters()
        for p, p_targ in zip(params, targ_params):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)


class Actor(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_size)
        self.logstd = nn.Linear(256, action_size)

    def dist(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2.0)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x, with_log_prob=False, deterministic=False):
        dist = self.dist(x)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        squashed_action, log_prob = _squash_action(dist, action)

        if with_log_prob:
            return squashed_action, log_prob

        return squashed_action


class Critic(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(observation_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        h = torch.relu(self.fc1(torch.cat([x, action], dim=1)))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


class SAC:
    def __init__(self,
                 observation_size,
                 action_size,
                 device,
                 learning_rate=3e-4,
                 gamma=0.99,
                 tau=0.005):
        self.observation_size = observation_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(observation_size, action_size)
        self.critic1 = Critic(observation_size, action_size)
        self.critic2 = Critic(observation_size, action_size)
        self.targ_critic1 = copy.deepcopy(self.critic1)
        self.targ_critic2 = copy.deepcopy(self.critic2)
        self.log_temp = nn.Parameter(torch.zeros(1, 1, device=device))

        self.actor.to(device)
        self.critic1.to(device)
        self.critic2.to(device)
        self.targ_critic1.to(device)
        self.targ_critic2.to(device)
        self.device = device

        self.actor_optim = Adam(self.actor.parameters(), lr=learning_rate)
        critic_parameters = chain(self.critic1.parameters(),
                                  self.critic2.parameters())
        self.critic_optim = Adam(critic_parameters, lr=learning_rate)
        self.temp_optim = Adam([self.log_temp], lr=learning_rate)

    def act(self, x, deterministic=False):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
            action = self.actor(x, deterministic=deterministic)
            return action.cpu().detach().numpy()

    def update_critic(self, obs_t, act_t, rew_tp1, obs_tp1, ter_tp1):
        obs_t = torch.tensor(obs_t, device=self.device).float()
        act_t = torch.tensor(act_t, device=self.device).float()
        rew_tp1 = torch.tensor(rew_tp1, device=self.device).float()
        obs_tp1 = torch.tensor(obs_tp1, device=self.device).float()
        ter_tp1 = torch.tensor(ter_tp1, device=self.device).float()

        with torch.no_grad():
            act_tp1, log_prob = self.actor(obs_tp1, with_log_prob=True)
            q1_tp1 = self.targ_critic1(obs_tp1, act_tp1)
            q2_tp1 = self.targ_critic2(obs_tp1, act_tp1)
            target = torch.min(q1_tp1, q2_tp1) - self.log_temp.exp() * log_prob

        y = rew_tp1 + self.gamma * target * (1.0 - ter_tp1)

        q1_loss = F.mse_loss(self.critic1(obs_t, act_t), y)
        q2_loss = F.mse_loss(self.critic2(obs_t, act_t), y)
        loss = q1_loss + q2_loss

        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

        return loss.cpu().detach().numpy()

    def update_actor(self, obs_t):
        obs_t = torch.tensor(obs_t, device=self.device).float()

        act_t, log_prob = self.actor(obs_t, with_log_prob=True)

        q1_t = self.critic1(obs_t, act_t)
        q2_t = self.critic2(obs_t, act_t)
        q_t = torch.min(q1_t, q2_t)

        loss = (self.log_temp.exp() * log_prob - q_t).mean()

        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

        return loss.cpu().detach().numpy()

    def update_temp(self, obs_t):
        obs_t = torch.tensor(obs_t, device=self.device).float()

        with torch.no_grad():
            _, log_prob = self.actor(obs_t, with_log_prob=True)
            targ_temp = log_prob - self.action_size

        loss = -(self.log_temp.exp() * targ_temp).mean()

        self.temp_optim.zero_grad()
        loss.backward()
        self.temp_optim.step()

        return loss.cpu().detach().numpy()

    def update_target(self):
        _soft_sync(self.targ_critic1, self.critic1, self.tau)
        _soft_sync(self.targ_critic2, self.critic2, self.tau)

    def save(self, fname):
        torch.save(
            {
                'actor': self.actor.state_dict(),
                'critic1': self.critic1.state_dict(),
                'critic2': self.critic2.state_dict(),
            }, fname)

    def load(self, fname):
        chkpt = torch.load(fname)
        self.actor.load_state_dict(chkpt['actor'])
        self.critic1.load_state_dict(chkpt['critic1'])
        self.critic2.load_state_dict(chkpt['critic2'])
