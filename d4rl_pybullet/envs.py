import gym
import pybullet_envs

from pybullet_envs.gym_locomotion_envs import HopperBulletEnv
from pybullet_envs.gym_locomotion_envs import HalfCheetahBulletEnv
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
from pybullet_envs.gym_locomotion_envs import AntBulletEnv
from .offline_env import OfflineEnv


class OfflineHopperBulletEnv(HopperBulletEnv, OfflineEnv):
    def __init__(self, **kwargs):
        HopperBulletEnv.__init__(self,)
        OfflineEnv.__init__(self, **kwargs)


class OfflineHalfCheetahBulletEnv(HalfCheetahBulletEnv, OfflineEnv):
    def __init__(self, **kwargs):
        HalfCheetahBulletEnv.__init__(self,)
        OfflineEnv.__init__(self, **kwargs)


class OfflineWalker2DBulletEnv(Walker2DBulletEnv, OfflineEnv):
    def __init__(self, **kwargs):
        Walker2DBulletEnv.__init__(self,)
        OfflineEnv.__init__(self, **kwargs)


class OfflineAntBulletEnv(AntBulletEnv, OfflineEnv):
    def __init__(self, **kwargs):
        AntBulletEnv.__init__(self,)
        OfflineEnv.__init__(self, **kwargs)
