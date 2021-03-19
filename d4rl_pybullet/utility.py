import numpy as np
import os
import h5py


def save_hdf5(observations, actions, rewards, terminals, timeouts, fname):
    with h5py.File(fname, 'w') as f:
        f.create_dataset('observations', data=observations)
        f.create_dataset('actions', data=actions)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('terminals', data=terminals)
        f.create_dataset('timeouts', data=timeouts)


def save_buffer(buffer, logdir):
    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []
    for i in range(len(buffer)):
        observations.append(buffer[i][0])
        actions.append(buffer[i][1])
        rewards.append(buffer[i][2])
        terminals.append(buffer[i][3])
        timeouts.append(buffer[i][4])

    observations = np.array(observations, dtype=np.float32)
    actions = np.array(actions, dtype=np.float32).flatten()
    rewards = np.array(rewards, dtype=np.float32).flatten()
    terminals = np.array(terminals, dtype=np.float32).flatten()
    timeouts = np.array(timeouts, dtype=np.float32).flatten()

    buffer_path = os.path.join(logdir, 'buffer.hdf5')
    save_hdf5(observations, actions, rewards, terminals, timeouts, buffer_path)
