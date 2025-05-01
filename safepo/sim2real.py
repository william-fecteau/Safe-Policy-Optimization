import numpy as np


def scale_friction(mujoco_env, scale_factor: float):
    mujoco_env.model.geom_friction[:] *= scale_factor
    return mujoco_env


def scale_model_weight(mujoco_env, scale_factor: float):
    mujoco_env.model.body_mass[:] *= scale_factor
    return mujoco_env


def add_noise_to_obs(obs):
    return obs + np.random.normal(0.0, 0.2, obs.shape)
