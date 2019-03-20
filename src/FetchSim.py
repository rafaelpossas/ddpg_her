
from delfi.simulator.BaseSimulator import BaseSimulator
import numpy as np
import gym
import delfi.distribution as dd
from delfi.summarystats import Identity
from delfi.generator import Default
from src.utils import helper
import pandas as pd
import pickle
class ExperienceGenerator(object):

    def __init__(self, filename="prior.pkl", prior=dd.Uniform(lower=[0.1], upper=[1.]), policy=None):
        self.filename = filename
        self.prior = prior
        self.policy = policy

    def gen_save(self, num_data=1000, env_id="FetchSlide-v1", dim=3):
        env = gym.make(env_id)
        env_params = ['friction']

        s = Identity()
        m = FetchSim(dim, self.policy, env, env_params)

        g = Default(model=m, prior=self.prior, summary=s)

        params, stats = g.gen(num_data)  # necessary for initiliasation
        helper.save({'params': params, 'data': stats}, self.filename)

class FetchSim(BaseSimulator):
    def __init__(self, dim=None, policy=None, env=None, env_params=[], gamma=1., seed=None, data=None):
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
        self.env = env.env
        self.policy = policy
        self.env_params = env_params
        self.gamma = gamma  # Parameter for discounted reward
        self.data = data
        self.load_from_disk = False
        # self.a = self.env.action_space.sample()

    def toggle_load_from_disk(self):
        self.load_from_disk = True

    def get_closest_param_idx(self, param, param_arr):
        diff_arr = list(map(lambda x: np.abs(x - param), param_arr))
        return diff_arr.index(min(diff_arr)[0])

    def gen_single(self, param):
        # Update parameters of the simulator

        if self.load_from_disk and self.data is not None:
            idx = self.get_closest_param_idx(param, self.data['params'])
            #print("True param: {}, Sampled Param: {}".format(param, data['params'][idx]))
            return {'data': self.data['data'][idx]}

        self.env.sim.model.geom_friction[22][0] = param
        self.env.sim.model.geom_friction[23][0] = param

        # Do a rollout
        n_eps = 10

        running_reward = 0.
        running_discounted = 0.

        ep_history = {'state': [], 'action': [], 'reward': [], 'next_state': []}
        total_eps = 0
        self.env.seed(0)
        while total_eps < n_eps:
            state = self.env.reset()
            cur_ep = {'state': [], 'action': [], 'reward': [], 'next_state': []}
            episode_done = False

            for steps in range(50):
                # Pick an action given the policy
                a = self.policy.get_actions(state['observation'], state['achieved_goal'], state['desired_goal'])

                next_state, r, _, info = self.env.step(a)
                #self.env.render()

                cur_ep['state'].append(state['observation'])
                cur_ep['action'].append(a)
                cur_ep['reward'].append(r)
                cur_ep['next_state'].append(next_state['observation'])

                state = next_state

                running_reward += r
                running_discounted = running_discounted + self.gamma * r

                episode_done = bool(info['is_success'])

                if episode_done:
                    break

            for idx in range(len(cur_ep['state'])):
                ep_history['state'].append(cur_ep['state'][idx])
                ep_history['action'].append(cur_ep['action'][idx])
                ep_history['reward'].append(cur_ep['reward'][idx])
                ep_history['next_state'].append(cur_ep['next_state'][idx])

            total_eps += 1

        ep_history = pd.DataFrame.from_dict(ep_history)
        n = len(ep_history)

        # Cross correlation between difference of states and actions
        sdim = ep_history.loc[0, 'state'].shape[0]
        adim = self.env.action_space.shape[0]
        state_difference = np.array(list((ep_history.loc[:, 'next_state'].values - ep_history.loc[:, 'state']).values))
        # state_difference = np.array([distance.euclidean(ep_history.loc[:, 'next_state'].values[idx],
        #                                        ep_history.loc[:, 'state'].values[idx]) for idx in range(len(ep_history))])[:, np.newaxis]
        actions = np.array(list(ep_history.loc[:, 'action'].values))
        sample = np.zeros((sdim, adim))

        for i in range(sdim):
            for j in range(adim):
                sample[i, j] = np.dot(state_difference[:, i], actions[:, j]) / (n - 1)
                # Add mean of absolut states changes and std to the summary statistics

        sample = sample.reshape(-1)
        #sample = []
        #sample.append(np.mean(state_difference, axis=0))
        #sample = np.array(sample)
        # sample = np.append(sample, np.mean(np.abs(state_difference), axis=0))
        # sample = np.append(sample, np.std(np.abs(actions), axis=0))

        # ind = np.isnan(sample)
        # sample[ind] = 0.
        if np.isnan(sample).any():
            print('Nan')
        return {'data': sample.reshape(-1)}


if __name__ == "__main__":
    policy_file = "logs/FetchSlide-v1/fixed_0.1_200_epochs/policy_best.pkl"
    example = 1
    # Load policy.
    env_id = "FetchSlide-v1"
    dim = 4
    env = gym.make(env_id)
    p = dd.Uniform(lower=[0.1], upper=[1.])

    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)

    x_plot, y_plot = helper.plot_robustness(policy=policy, env=env.env, env_params=['friction'], ntests=5,
                                            distribution=p, true_param=0.1)