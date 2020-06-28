import numpy as np
import os
import h5py


def save_hdf5(observations, actions, rewards, terminals, fname):
    with h5py.File(fname, 'w') as f:
        f.create_dataset('observations', data=observations)
        f.create_dataset('actions', data=actions)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('terminals', data=terminals)


def save_buffer(buffer, logdir):
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

    buffer_path = os.path.join(logdir, 'buffer.hdf5')
    save_hdf5(observations, actions, rewards, terminals, buffer_path)
