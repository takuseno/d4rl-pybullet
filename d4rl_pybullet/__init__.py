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
        'https://www.dropbox.com/s/bino8ojd7iq4p4d/hopper-bullet-random-v0.hdf5?dl=0'
    })

register(
    id='halfcheetah-bullet-medium-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/wdo6lktqgyqztfk/halfcheetah-bullet-medium-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-mixed-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/ne23j1jdoks240g/halfcheetah-bullet-mixed-v0.hdf5?dl=1'
    })

register(
    id='halfcheetah-bullet-random-v0',
    entry_point='d4rl_pybullet.envs:OfflineHalfCheetahBulletEnv',
    kwargs={
        'dataset_url':
        'https://www.dropbox.com/s/m30chvy08vg4dou/halfcheetah-bullet-random-v0.hdf5?dl=1'
    })
