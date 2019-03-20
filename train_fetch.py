import sys
import os
import time
import gym
from gym import spaces
from src.utils.zooutils import ALGOS, create_test_env
import pybullet_envs
import numpy as np
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.common.vec_env import dummy_vec_env
from stable_baselines.bench import Monitor

from delfi.simulator.BaseSimulator import BaseSimulator
from sklearn.linear_model import LinearRegression
import delfi.distribution as dd

from delfi.summarystats import Identity
from delfi.generator import Default
from delfi.inference import Basic, CDELFI, SNPE
import yaml
sys.path.append('/home/rafaelpossas/Dev/projects/rl-baselines-zoo')


class RLSim(BaseSimulator):
    def __init__(self, dim=None, policy=None, env=None, env_params=[], gamma=1., seed=None):
        """RL Simulator. Given an environment and a policy, simulates data

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        env : env
            Gym environment
        env_params : list
            env parameters to change
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=dim, seed=seed)
        self.env = env
        self.policy = policy
        self.env_params = env_params
        self.gamma = gamma  # Parameter for discounted reward
        # self.a = self.env.action_space.sample()

    def gen_single(self, param):
        # Update parameters of the simulator

        for i in range(len(self.env_params)):
            setattr(self.env.get_attr('env')[0].env,
                    self.env_params[i],
                    param[i])

        # Do a rollout
        nsteps = 200
        s = self.env.reset()
        running_reward = 0.
        running_discounted = 0.
        deterministic = False
        ep_history = []
        for j in range(nsteps):
            # Pick an action given the policy
            a, _ = self.policy.predict(s, deterministic=deterministic)

            # Random Agent
            # action = [env.action_space.sample()]
            # Clip Action to avoid out of bound errors
            if isinstance(self.env.action_space, gym.spaces.Box):
                a = np.clip(a, self.env.action_space.low, self.env.action_space.high)

            # a = self.env.action_space.sample()
            # a = np.array([0.01, 0.01, 0.01, 0.01])
            # a = self.a

            s1, r, d, _ = self.env.step(a)


            s_shape = s.shape[1]

            tmp = list(s.reshape(s_shape))

            if type(a) is np.ndarray:
                # Necessary for some simulators
                if type(a[0]) is np.int64:
                    a = a.tolist()
                    tmp.extend(a)
                else:
                    at = a[0]
                    at = at.tolist()
                    tmp.extend(at)
            else:
                tmp.append(a)

            tmp.append(r)

            tmp.extend(list(s1.reshape((s_shape))))
            ep_history.append(tmp)
            s = s1

            running_reward += r
            running_discounted = running_discounted + self.gamma * r
            if d == True:
                s = env.reset()
                """
                #Sets the parameters again after each reset
                if self.env is type(VecNormalize):
                    for i in range(len(self.env_params)):
                        setattr(self.env.get_attr('env')[0].env,
                                self.env_params[i], 
                                param[i])

                else:
                    for i in range(len(self.env_params)):
                        setattr(self.env.venv.get_attr('env')[0].env,
                                self.env_params[i], 
                                param[i])
                """
        ep_history = np.array(ep_history)
        N = ep_history.shape[0]
        # Cross correlation between difference of states and actions
        sdim = env.observation_space.shape[0]
        adim = np.size(env.action_space.sample())
        sample = np.zeros((sdim, adim))
        tmp = ep_history[:, sdim + adim + 1:] - ep_history[:, 0:sdim]  # difference between states (s1-s)
        tmp2 = ep_history[:, sdim:sdim + adim]  # actions

        for i in range(sdim):
            for j in range(adim):
                sample[i, j] = np.dot(tmp[:, i], tmp2[:, j]) / (N - 1)
                # Add mean of absolut states changes and std to the summary statistics
        sample = sample.reshape(-1)
        sample = np.append(sample, np.mean(np.abs(tmp), axis=0))
        sample = np.append(sample, np.std(tmp.tolist(), axis=0))

        # ind = np.isnan(sample)
        # sample[ind] = 0.
        return {'data': sample.reshape(-1)}

    # Sets up the problem
class ARGS():
    def __init__(self, env):
        self.env = env
        self.folder = '/home/rafaelpossas/Dev/projects/rl-baselines-zoo/trained_agents'
        self.algo = 'ppo2'
        self.n_envs = 1
        self.no_render = True
        self.deterministic = False
        self.norm_reward = False
        self.seed = 0
        self.reward_log = '/tmp/'

# Update env parameters in multi environments
def update_nenv(env, env_param, param, env_i):
    # Update parameters of the simulator
    if hasattr(env,'venv'):
        setattr(env.venv.get_attr('env')[env_i].env,
                env_param,
                param)
    # For pybullet
    elif isinstance(env.env, (pybullet_envs.gym_locomotion_envs.HopperBulletEnv)):
        pybullet_server = env.env.robot._p
        bodyId = 0
        linkIndex = -1
        tmp = {env_param:param}
        pybullet_server.changeDynamics(bodyId, linkIndex, **tmp)
    else:
        setattr(env.get_attr('env')[env_i].env,
                env_param,
                param)
        print(getattr(env.get_attr('env')[env_i].env, env_param))


# Update env parameters
def update_env(env, env_param, param):
    # Update parameters of the simulator
    if hasattr(env, 'env'):

        setattr(env.env,
                env_param,
                param)
    else:
        setattr(env.get_attr('env')[0].env,
                env_param,
                param)

def make_env(env_id, rank=0, seed=0, env_params=None, sample=None, log_dir=None):
    if log_dir is None and log_dir != '':
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        set_global_seeds(args.seed + rank)
        env = gym.make(env_id)
        env.seed(seed + rank)
        if not env_params is None:
            for i in range(len(env_params)):
                update_env(env, env_params[i], sample[i])

        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init
if __name__ == "__main__":
    # CarPole
    # env = gym.make('CartPole-v1')
    env_id = 'CartPole-v1'
    dim = 2
    env_params = ['length', 'masspole']
    # env_params = ['masspole']
    p = dd.Uniform(lower=[0.1, 0.1], upper=[2., 2.])

    # True values for the observation
    true_obs = [0.7, 1.3]

    args = ARGS(env_id)
    algo = args.algo
    folder = args.folder
    model_path = "{}/{}/{}.pkl".format(folder, algo, env_id)

    # Sanity checks
    # assert os.path.isdir(folder + '/' + algo), "The {}/{}/ folder was not found".format(folder, algo)
    # assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, env_id, model_path)

    if algo in ['dqn', 'ddpg']:
        args.n_envs = 1

    set_global_seeds(args.seed)

    # is_atari = 'NoFrameskip' in env_id

    stats_path = "{}/{}/{}/".format(folder, algo, env_id)
    if not os.path.isdir(stats_path):
        stats_path = None

    log_dir = args.reward_log if args.reward_log != '' else None

    if not 'Bullet' in env_id:
        env = create_test_env(env_id, n_envs=args.n_envs, is_atari=False,
                              stats_path=stats_path, norm_reward=args.norm_reward,
                              seed=args.seed, log_dir=log_dir, should_render=not args.no_render)
    else:
        env = gym.make("HopperBulletEnv-v0")
        env.render(mode="human")

    model = ALGOS[algo].load(model_path)
    obs = env.reset()

    s = Identity()
    m = RLSim(dim, model, env, env_params)
    g = Default(model=m, prior=p, summary=s)

    #params, stats = g.gen(1)  # necessary for initiliasation

    #Number of testing samples
    # ntest = 10
    # x_test = np.zeros((stats.shape[1], ntest))
    # for i in range(ntest):
    #     x_test[:, i] = m.gen_single(np.array(true_obs))['data']
    # x_test = np.mean(x_test, axis=1)

    # Number of components for both methods
    # n_components = 5
    # inf_basic = SNPE(generator=g, obs=x_test.reshape(1, -1), n_components=n_components, n_hiddens=[24, 24], svi=False)
    #
    # log, train_data, _ = inf_basic.run(n_train=1000, epochs=1000, n_rounds=1)
    # posterior = inf_basic.predict(x_test.reshape(1, -1))
    #
    # for dim in range(params.shape[1]):
    #     print('Parameter ' + str(dim + 1) + ':')
    #     for k in range(posterior.ncomp):
    #         print(r'component {}: mixture weight = {:.4f}; mean = {:.4f}; variance = {:.4f}'.format(
    #             k + 1, posterior.a[k], posterior.xs[k].m[dim], posterior.xs[k].S[dim][dim]))

    env_id = args.env
    model = ALGOS[algo].load(model_path)

    samples = p.gen(args.n_envs)
    #samples = posterior.gen(args.n_envs)

    env_train = SubprocVecEnv([make_env(env_id, i, args.seed, env_params, samples[i]) for i in range(args.n_envs)])
    obs = env_train.reset()

    # Load hyperparameters from yaml file
    with open('/home/rafaelpossas/Dev/projects/rl-baselines-zoo/hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams = yaml.load(f)[env_id]

        hyperparams['n_envs'] = args.n_envs
        n_envs = hyperparams.get('n_envs', 1)
        print("Using {} environments".format(n_envs))

    del hyperparams['n_envs']
    #n_timesteps = int(hyperparams['n_timesteps'])
    n_timesteps = 10
    del hyperparams['n_timesteps']
    tensorboard_log = '/tmp'
    model_trained = ALGOS[args.algo](env=env_train, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)
    print('Training for', n_timesteps, 'steps')
    kwargs = {}
    kwargs = {'log_interval': 10}
    model_trained.learn(n_timesteps, **kwargs)


    save_path = os.path.join("/home/rafaelpossas/Dev/projects/ddpg_her", "{}_{}".format(env_id, 0))
    params_path = "{}/{}".format(save_path, env_id)
    os.makedirs(params_path, exist_ok=True)
    print("Saving to {}".format(save_path))

    model.save("{}/{}".format(save_path, env_id))