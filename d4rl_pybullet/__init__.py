from gym.envs.registration import register

register(
    id='hopper-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/w22kgzldn6eng7j/hopper-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='hopper-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/xv3p0h7dzgxt8xb/hopper-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='hopper-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/bino8ojd7iq4p4d/hopper-bullet-random-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/v4xgssp1w968a9l/halfcheetah-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/scj1rqun963aw90/halfcheetah-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/jnvpb1hp60zt2ak/halfcheetah-bullet-random-v0.hdf5?dl=1'
    })
