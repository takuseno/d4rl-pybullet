from gym.envs.registration import register

register(
    id='hopper-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/w22kgzldn6eng7j/hopper-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='hopper-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/xv3p0h7dzgxt8xb/hopper-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='hopper-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/bino8ojd7iq4p4d/hopper-bullet-random-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/v4xgssp1w968a9l/halfcheetah-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/scj1rqun963aw90/halfcheetah-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/jnvpb1hp60zt2ak/halfcheetah-bullet-random-v0.hdf5?dl=1'
    })

register(
    id='ant-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineAntBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/6n79kwd94xthr1t/ant-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='ant-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineAntBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/pmy3dzab35g4whk/ant-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='ant-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineAntBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/2xpmh4wk2m7i8xh/ant-bullet-random-v0.hdf5?dl=1'
    })

register(
    id='walker2d-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineWalker2DBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/v0f2kz48b1hw6or/walker2d-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='walker2d-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineWalker2DBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/i4u2ii0d85iblou/walker2d-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='walker2d-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineWalker2DBulletEnv',
    max_episode_steps=1000,
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/1gwcfl2nmx6878m/walker2d-bullet-random-v0.hdf5?dl=1'
    })
