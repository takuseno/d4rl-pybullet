from gym.envs.registration import register

register(
    id='hopper-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/4bndqk4f4pi8gya/hopper-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='hopper-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/yvl8wrb129seukb/hopper-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='hopper-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineHopperBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/fz1u9nak9z1sa9y/hopper-bullet-random-v0.hdf5?dl=1'
    })
