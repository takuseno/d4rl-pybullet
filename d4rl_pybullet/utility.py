import h5py


def save_hdf5(observations, actions, rewards, terminals, fname):
    with h5py.File(fname, 'w') as f:
        f.create_dataset('observations', data=observations)
        f.create_dataset('actions', data=actions)
        f.create_dataset('rewards', data=rewards)
        f.create_dataset('terminals', data=terminals)
