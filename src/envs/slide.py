import numpy as np
import os
import copy
import sys
from gym import error, spaces
from gym.utils import seeding
from gym.envs.robotics import rotations
import pandas as pd

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (HINT: you need to install mujoco_py, and also perform the setup "
                                       "instructions here: https://github.com/openai/mujoco-py/.)".format(e))


class FetchSlideEnv():
    # y= 0.45 - 1
    def __init__(self, model_path="fetch/slide_table.xml", n_substeps=40):
        self.initial_qpos = {
            'object0:joint': [0.1, 0.45, 0.42, 1., 0., 0., 0.],
        }
        # fetch.FetchEnv.__init__(
        #     self, 'fetch/slide_table.xml', has_object=True, block_gripper=True, n_substeps=20,
        #     gripper_extra_height=-0.02, target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
        #     obj_range=0.1, target_range=0.3, distance_threshold=0.05,
        #     initial_qpos=initial_qpos, reward_type=reward_type)
        # utils.EzPickle.__init__(self)

        if model_path.startswith('/'):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(__file__), 'assets', model_path)

        if not os.path.exists(fullpath):
            raise IOError('File {} does not exist'.format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)

        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.viewer = None
        self.total_steps = 0

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        self.seed()
        self._env_setup(initial_qpos=self.initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        obs = self._get_obs()

        self.observation_space = spaces.Dict(dict(
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def get_friction(self):
        return self.sim.model.geom_friction[1][0]

    def reset(self, random=True, friction=None):
        if random:
            x = np.random.uniform(0.1, 0.2, 1)[0]
            y = np.random.uniform(0.45, 1, 1)[0]
            z = 0.42
            self.initial_qpos = {
                'object0:joint': [x, y, z, 1., 0., 0., 0.],
            }
        else:
            self.initial_qpos = {
                'object0:joint': [1, 0.45, 0.42, 1., 0., 0., 0.],
            }

        for name, value in self.initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        if friction is not None:
            self.sim.model.geom_friction[1][0] = friction
            self.sim.model.geom_friction[2][0] = friction
        # Randomize start position of object.

        self.sim.forward()

        return True

    def reset_mocap_welds(self,sim):
        """Resets the mocap welds that we use for actuation.
        """
        if sim.model.nmocap > 0 and sim.model.eq_data is not None:
            for i in range(sim.model.eq_data.shape[0]):
                if sim.model.eq_type[i] == mujoco_py.const.EQ_WELD:
                    sim.model.eq_data[i, :] = np.array(
                        [0., 0., 0., 1., 0., 0., 0.])
        sim.forward()

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)

        self.reset_mocap_welds(self.sim)
        self.sim.forward()

        for _ in range(10):
            self.sim.step()
            #self.render()

        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def _viewer_setup(self):
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        self.sim.forward()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_viewer(self):
        if self.viewer is None:
            self.viewer = mujoco_py.MjViewer(self.sim)
            self._viewer_setup()
        return self.viewer

    def render(self, mode='human'):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer().render()

    def _get_obs(self):
        # positions
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velr = self.sim.data.get_site_xvelr('object0') * dt

        achieved_goal = np.squeeze(object_pos.copy())

        obs = np.concatenate([
            object_pos.ravel(), object_rot.ravel(),
            object_velr.ravel(),
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
        }

    def step(self):
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': True,
        }
        # reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        self.total_steps += 1
        return obs, 0, done, info

    def apply_force(self, apply_force=1):
        self.sim.data.xfrc_applied[2][0] = apply_force
        self.step()

    def _step_callback(self):
        self.sim.forward()
