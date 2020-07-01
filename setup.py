from setuptools import setup, find_packages


setup(name="d4rl_pybullet",
      version="0.1",
      license="MIT",
      description="Datasets for data-driven deep reinforcement learnig with Pybullet environments",
      url="https://github.com/takuseno/d4rl-pybullet",
      install_requires=["gym", "pybullet", "h5py"],
      packages=find_packages())
