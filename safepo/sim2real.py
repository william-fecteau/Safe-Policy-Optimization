import numpy as np


def add_noise_to_friction(mujoco_env):
    mujoco_env.model.geom_friction[:] *= np.random.uniform(
        0.8, 1.2, mujoco_env.model.geom_friction[:].shape
    )
    return mujoco_env


def add_noise_to_gravity(mujoco_env):
    mujoco_env.model.opt.gravity[:] *= np.random.uniform(
        0.8, 1.2, mujoco_env.model.opt.gravity[:].shape
    )
    return mujoco_env


def add_noise_to_obs(obs):
    return obs + np.random.normal(0.0, 0.2, obs.shape)
