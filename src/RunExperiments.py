import sys
import argparse
import datetime
import pickle
import gym
from gym import spaces
import numpy as np
from src.fabio.MDN import MDNN, MDRFF
from src.fabio.BayesSim import BayesSim
sys.path.append('/home/rafaelpossas/Dev/projects/rl-baselines-zoo')
import os
from src.utils.zooutils import ALGOS, create_test_env
import pybullet_envs
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env import VecNormalize, VecFrameStack, SubprocVecEnv
from stable_baselines.common.vec_env import dummy_vec_env
from stable_baselines.bench import Monitor
from delfi.inference import Basic, CDELFI, SNPE
from delfi.summarystats import Identity
from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.generator import Default
from sklearn.linear_model import LinearRegression
import delfi.distribution as dd
from delfi.kernel import Gauss, Tricube, Epanechnikov, Uniform
from src.utils.reactive_policy import SmallReactivePolicy
import scipy
import elfi

class RunExperiments(object):
    """ Run a series of experiments and comparisons for BayesSim"""
    def __init__(self,
                 algos=None,
                 problems=None,
                 logdir='/tmp/'):

        if algos is None:
            # algos = ['MDRFF', 'MDNN', 'Basic', 'CDELFI', 'SNPE']
            algos = ['REJ-ABC']

        if problems is None:
            problems = ['CartPole-v1', 'Pendulum-v0', 'MountainCarContinuous-v0',\
                        'Acrobot-v1', 'BipedalWalker-v2', 'HopperBulletEnv-v0', 'Walker2DBulletEnv-v0']
        self.algos = algos
        self.problems = problems
        self.logdir = logdir

    def runTests(self, algos=None, problems=None):

        nalgos = len(algos)
        nproblems = len(problems)
        # Number of components for both methods
        n_components = 10
        # Summary statistic
        s = Identity()
        results = {}
        print(*algos)
        print(*problems)
        # Log directory
        logdir = os.path.realpath(self.logdir)
        os.makedirs(logdir, exist_ok=True)
        #logger.configure(logdir)
        suffix = 'pkl'

        for i in range(nproblems):
            results[problems[i]] = {}
            # Set up the problem for each environment
            prior, elfi_prior, env_params, true_obs, env_id, dim = self.setUpProblem(problems[i])
            args = ARGS(env_id)
            algo = args.algo
            folder = args.folder
            model_path = "{}/{}/{}.pkl".format(folder, algo, env_id)

            if algo in ['dqn', 'ddpg']:
                args.n_envs = 1

            set_global_seeds(args.seed)

            stats_path = "{}/{}/{}/".format(folder, algo, env_id)
            if not os.path.isdir(stats_path):
                stats_path = None

            log_dir = args.reward_log if args.reward_log != '' else None

            if not 'Bullet' in env_id:
                env = create_test_env(env_id, n_envs=args.n_envs, is_atari=False,
                                      stats_path=stats_path, norm_reward=args.norm_reward,
                                      seed=args.seed, log_dir=log_dir, should_render=not args.no_render)
            else:
                env = gym.make(env_id)
                #env.render(mode="human")

            if 'HopperBullet' in env_id:
                model = SmallReactivePolicy(env.observation_space, env.action_space)
                obs = env.reset()
            else:
                model = ALGOS[algo].load(model_path)
                obs = env.reset()

            # Force deterministic for DQN and DDPG
            deterministic = args.deterministic or algo in ['dqn', 'ddpg']

            # specific code for pybullet
            if 'Bullet' in env_id:
                bodyId = 0
                linkIndex = -1
                pybullet_server = env.env.robot._p
                pybullet_server.setGravity(0, 0, -9.8)

                # Adds mass to the problem
                pybullet_server.changeDynamics(bodyId, linkIndex, mass=1.)
                mass, lateral_friction, local_inertia_diag, \
                local_inertia_pos, local_inertia_orn, \
                restitution, rolling_friction, \
                spinning_friction, contact_damping, \
                contact_stiffness = pybullet_server.getDynamicsInfo(bodyId, linkIndex)

                print("==================== \nDynamics Info:\n"
                      "Mass: {}\n"
                      "Lateral Friction: {}\n"
                      "Local Ineratia Diagonal: {}\n"
                      "Local Inertia Position: {}\n"
                      "Local Inertia Orn: {}\n"
                      "Restitution: {}\n"
                      "Rolling Friction: {}\n"
                      "Spinning Friction: {}\n"
                      "Contact Damping: {}\n"
                      "Contact Stiffness: {}".
                      format(mass, lateral_friction, local_inertia_diag, local_inertia_pos, local_inertia_orn,
                             restitution,
                             rolling_friction, spinning_friction, contact_damping, contact_stiffness))


            m = RLSim(dim, model, env, env_params)
            g = Default(model=m, prior=prior, summary=s)

            # Generates observations with the true parameters
            params, stats = g.gen(1)  # necessary for initiliasation

            # Number of testing samples
            ntest = 10
            x_test = np.zeros((stats.shape[1], ntest))
            for ll in range(ntest):
                x_test[:, ll] = m.gen_single(np.array(true_obs))['data']
            x_test = np.mean(x_test, axis=1)

            for j in range(nalgos):
                print('Testing '+ algos[j] + ' on '+problems[i])
                results[problems[i]][algos[j]] = {}

                if algos[j] in 'MDRFF':
                    print('running MDRFF')
                    weight_fcn = Gauss(obs=x_test.reshape(1, -1), bandwidth=0.5)
                    mdn = MDRFF(ncomp=n_components, outputd=params.shape[1], inputd=stats.shape[1],
                                nfeat=1000, sigma=5., kernel='Matern52', weights=weight_fcn)
                    inf_basic = BayesSim(generator=g, obs=x_test, model=mdn)

                elif algos[j] in 'MDNN':
                    print('running MDNN')
                    mdn = MDNN(ncomp=n_components, outputd=params.shape[1], inputd=stats.shape[1], nhidden=2,
                               nunits=[24, 24])
                    inf_basic = BayesSim(generator=g, obs=x_test, model=mdn)

                elif algos[j] in 'Basic':
                    print('running Basic')
                    inf_basic = Basic(generator=g, obs=x_test.reshape(1, -1),
                                      n_components=n_components,
                                      prior_norm=False,
                                      n_hiddens=[24, 24], svi=False)

                elif algos[j] in 'CDELFI':
                    print('running CDELFI')
                    inf_basic = CDELFI(generator=g, obs=x_test.reshape(1, -1),
                                       n_components=1,
                                       prior_norm=False,
                                       n_hiddens=[24, 24], svi=False)

                elif algos[j] is 'SNPE':
                    print('running SNPE')
                    kernel = None
                    inf_basic = SNPE(generator=g, obs=x_test.reshape(1, -1),
                                     n_components=n_components,
                                     kernel=kernel,
                                     prior_norm=False,
                                     n_hiddens=[24, 24], svi=False)
                elif algos[j] == "REJ-ABC":
                    print('running REJ-ABC')

                    def predict_mog_from_stats(ntest=1, alpha_pred=None, mu_pred=None, sigma_pred=None, ncomp=None,
                                               output_dim=None):
                        alpha_pred = np.array(alpha_pred).reshape(-1, ncomp)
                        mu_pred = np.array(mu_pred).reshape(-1, ncomp, output_dim)
                        sigma_pred = np.array(sigma_pred).reshape(-11, ncomp, output_dim)
                        mog = []
                        for pt in range(ntest):
                            a = alpha_pred[pt, :]
                            ms = [mu_pred[pt, i, :] for i in range(ncomp)]
                            Ss = []
                            di = np.diag_indices(output_dim)  # diagonal indices
                            for i in range(ncomp):
                                tmp = np.zeros((output_dim, output_dim))
                                tmp[di] = sigma_pred[pt, i, :] ** 2
                                Ss.append(tmp)
                            mog.append(dd.MoG(a=a, ms=ms, Ss=Ss))
                        return mog[0]

                    def identity(y):
                        return y

                    def gen_data_elfi(m, t1, batch_size=1, random_state=None):
                        results = []
                        for param in t1:
                            data = m.gen_single(param)['data']
                            results.append(data)

                        return np.array(results)

                    y_obs = gen_data_elfi(m, np.expand_dims(true_obs, axis=0))
                    sim = elfi.Simulator(gen_data_elfi, m, elfi_prior, observed=y_obs)
                    S1 = elfi.Summary(identity, sim)
                    d = elfi.Distance('euclidean', S1)
                    rej = elfi.Rejection(d, batch_size=1000, seed=30052017)
                    # kernel = None
                    # inf_basic = SNPE(generator=g, obs=x_test.reshape(1, -1),
                    #                  n_components=n_components,
                    #                  kernel=kernel,
                    #                  prior_norm=False,
                    #                  n_hiddens=[24, 24], svi=False)

                # Trains the model
                if algos[j] != "REJ-ABC":
                    log, train_data, _ = inf_basic.run(n_train=2000, epochs=500, n_rounds=1)
                    # Computes the posterior
                    posterior = inf_basic.predict(x_test.reshape(1, -1))
                else:
                    res = rej.sample(1000, threshold=.3)
                    mean = np.mean(res.samples_array)
                    var = np.var(res.samples_array)
                    print("REJ-ABC Mean: {} Var: {}".format(mean, var))
                    posterior = predict_mog_from_stats(alpha_pred=[1], mu_pred=[mean], sigma_pred=[var], ncomp=1, output_dim=1)




                for par in range(len(env_params)):
                    llk = posterior.eval(np.array([true_obs[par]]), log=True)
                    results[problems[i]][algos[j]][env_params[par]] =llk[0]

                    # Saving the results
                    now = datetime.datetime.now()
                    filename = os.path.join(self.logdir,
                                            'BayesSimResults ' + now.strftime(' %m %d %Y %H:%M') +
                                            '.' + suffix)
                    fileobj = open(filename, 'wb')
                    pickle.dump([results], fileobj)
                    print('Saving the results...')
                    fileobj.close()
                    env.close()




        return results


    def setUpProblem(self, problem=None):
        env_params = []
        if 'CartPole' in problem:
            # CartPole
            env_id = 'CartPole-v1'
            env_params = ['length', 'masspole']
            elfi_prior = elfi.Prior(scipy.stats.uniform, [0.1, 0.1], [2., 2.])
            prior = dd.Uniform(lower=[0.1, 0.1], upper=[2., 2.])

            # True values for the observation
            true_obs = [0.7, 1.3]


        elif 'Pendulum' in problem:
            # Pendulum
            env_id = 'Pendulum-v0'
            env_params = ['dt']
            elfi_prior = elfi.Prior(scipy.stats.uniform, [0.01], [0.3])
            prior = dd.Uniform(lower=[0.01], upper=[0.3])

            # True values for the observation
            true_obs = [0.2]

        elif 'MountainCar' in problem:
            # MountainCarcontinuous
            env_id = 'MountainCarContinuous-v0'
            env_params = ['power']
            elfi_prior = elfi.Prior(scipy.stats.uniform, [0.0005], [0.01])
            prior = dd.Uniform(lower=[0.0005], upper=[0.1])

            # True values for the observation
            true_obs = [0.03]

        elif 'Acrobot' in problem:
            # Acrobot-v1
            env_id = 'Acrobot-v1'
            env_params = ['LINK_MASS_1', 'LINK_MASS_2', 'LINK_LENGTH_1', 'LINK_LENGTH_2']
            elfi_prior = elfi.Prior(scipy.stats.uniform, [0.5, 0.5, 0.1, 0.5], [2., 2., 1.5, 1.5])
            prior = dd.Uniform(lower=[0.5, 0.5, 0.1, 0.5], upper=[2., 2., 1.5, 1.5])

            # True values for the observation
            true_obs = [0.9, 1.2, 0.9, 1.4]

        elif 'HopperBullet' in problem:
            # Hopper from pybullet
            env_id = 'HopperBulletEnv-v0'
            env_params = ['lateralFriction']
            elfi_prior = elfi.Prior(scipy.stats.uniform, [0.3], [0.5])
            prior = dd.Uniform(lower=[0.3], upper=[0.5])
            # True values for the observation
            true_obs = [0.35]

        elif 'Walker2dBullet' in problem:
            # Walker from pybullet
            env_id = 'Walker2DBulletEnv-v0'
            env_params = ['lateralFriction']
            elfi_prior = elfi.Prior(scipy.stats.uniform, [0.3], [0.5])
            prior = dd.Uniform(lower=[0.3], upper=[0.5])

            # True values for the observation
            true_obs = [0.45]

        else:
            ValueError('Environment not setup')

        dim = len(env_params)


        return prior, elfi_prior, env_params, true_obs, env_id, dim


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
        if not hasattr(param, "__len__"):
            param = [param]
        if hasattr(self.env, 'venv'):
            for i in range(len(self.env_params)):
                setattr(self.env.venv.get_attr('env')[0].env,
                        self.env_params[i],
                        param[i])

        elif hasattr(self.env, 'env') and \
                (isinstance(self.env.env, (pybullet_envs.gym_locomotion_envs.HopperBulletEnv)) or \
                 isinstance(self.env.env, (pybullet_envs.gym_locomotion_envs.Walker2DBulletEnv))):
            for i in range(len(self.env_params)):
                pybullet_server = self.env.env.robot._p
                pybullet_server.setGravity(0, 0, -9.8)
                bodyId = 0
                linkIndex = -1
                tmp = {self.env_params[i]: param[i]}
                pybullet_server.changeDynamics(bodyId, linkIndex,
                                               **tmp)
                # print("The new Lateral friction is: {}".format(pybullet_server.getDynamicsInfo(bodyId, linkIndex)[1]))

        else:
            for i in range(len(self.env_params)):
                setattr(self.env.get_attr('env')[0].env,
                        self.env_params[i],
                        param[i])

        # Do a rollout
        nsteps = 50
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

            if hasattr(self.env, 'env') and \
                    (isinstance(self.env.env, (pybullet_envs.gym_locomotion_envs.HopperBulletEnv)) or \
                     isinstance(self.env.env, (pybullet_envs.gym_locomotion_envs.Walker2DBulletEnv))):
                s_shape = self.env.env.observation_space.shape[0]
            else:
                s_shape = s.shape[1]

            tmp = list(s.reshape(s_shape))

            if type(a) is np.ndarray:
                # Necessary for some simulators
                if type(a[0]) is np.int64:
                    a = a.tolist()
                    tmp.extend(a)
                else:
                    if hasattr(self.env, 'env') and \
                            (isinstance(self.env.env, (pybullet_envs.gym_locomotion_envs.HopperBulletEnv)) or \
                             isinstance(self.env.env, (pybullet_envs.gym_locomotion_envs.Walker2DBulletEnv))):
                        at = a
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
                s = self.env.reset()
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
        if False:
            # Linear regression coefficients to predict next state
            sdim = env.observation_space.shape[0]
            adim = np.size(env.action_space.sample())
            # print(ep_history[:,0:sdim+adim+1])
            # print(ep_history[:,sdim+adim+1:])
            reg = LinearRegression().fit(
                ep_history[:, 0:sdim + adim + 1],
                ep_history[:, sdim + adim + 1:])
            sample = reg.coef_.reshape(-1)
        elif True:
            # Cross correlation between difference of states and actions
            sdim = self.env.observation_space.shape[0]
            adim = np.size(self.env.action_space.sample())
            sample = np.zeros((sdim, adim))
            tmp = ep_history[:, sdim + adim + 1:] - ep_history[:, 0:sdim]  # difference between states (s1-s)
            tmp2 = ep_history[:, sdim:sdim + adim]  # actions

            for i in range(sdim):
                for j in range(adim):
                    sample[i, j] = np.dot(tmp[:, i], tmp2[:, j]) / (N - 1)
                    # Add mean of absolut states changes and std to the summary statistics
            sample = sample.reshape(-1)
            sample = np.append(sample, np.mean(tmp, axis=0))
            sample = np.append(sample, np.std(tmp.tolist(), axis=0))

        return {'data': sample.reshape(-1)}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--problems', type=str, nargs='+',
                        default=['Pendulum-v0', \
                                 'MountainCarContinuous-v0', 'HopperBulletEnv-v0',
                                 'Walker2dBulletEnv-v0'],
                        help='Problems to test the algorithms on. Choices are CartPole-v1, Pendulum-v0, \
                            MountainCarContinuous-v0, Acrobot-v1, HopperBulletEnv-v0, Walker2DBulletEnv-v0')
    parser.add_argument('--algorithms', nargs='+',
                        help='Algorithms for comparison. Choices are: MDRFF, MDNN, Basic, CDELFI, SNPE',
                        default=['REJ-ABC'],
                        type=str)
    parser.add_argument("--logdir", help="Log results in file",
                        default="/home/rafaelpossas/Dev/results", type=str)


    args = parser.parse_args()
    return args

if __name__=='__main__':
    #default = ['MDRFF', 'MDNN', 'Basic', 'CDELFI', 'SNPE']
    args = parse_args()
    experiments = RunExperiments(logdir=args.logdir)
    results = experiments.runTests(args.algorithms, args.problems)
    print(results)