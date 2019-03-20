import numpy as np
import sys
import copy
import pandas as pd
import torch
from src.envs.slide import FetchSlideEnv
from scipy.spatial import distance
from src.models.MixtureDensityNetwork import MixtureDensityNetwork
from src.utils import helper

def run_sims_from_prior(prior_fn, render=False, wait_n_steps = 50,
                        reset_friction_every=10,
                        num_datapoints=10,
                        verbose=True):

    env = FetchSlideEnv()
    force_applied = False
    measuring_steps = 0
    sim_log = []
    cur_sim = []
    idx = 1
    stats_log = []
    #env.sim.model.body_mass[2] = 0.5

    if render:
        env.render()

    friction = prior_fn()
    env.reset(friction=friction)

    while len(sim_log) < num_datapoints:
        if not force_applied:
            if verbose:
                print("--------------------------")
                print("Initial Position: {}, Friction Coefficient: {}".format(env.sim.data.qpos[0:3], env.get_friction()))
            initial_pos = copy.deepcopy(env.sim.data.qpos[0:3])
            measuring_steps = 0
            env.apply_force()
            force_applied = True


        if force_applied:
            measuring_steps += 1
            if measuring_steps == wait_n_steps:
                final_pos = copy.deepcopy(env.sim.data.qpos[0:3])
                dist = distance.euclidean(initial_pos, final_pos)
                if verbose:
                    print("Final Position: {}".format(env.sim.data.qpos[0:3]))
                    print("Distance: {} ".format(dist))
                    print("--------------------------\n")
                final_res = np.concatenate(([env.get_friction()], initial_pos, final_pos, [dist]))
                sim_log.append(final_res)
                cur_sim.append(final_res)
                measuring_steps = 0
                force_applied = False

                if idx % reset_friction_every == 0:
                    stats_log.append(calculate_stats(cur_sim, friction, metric_idx=-1))
                    cur_sim = []
                    friction = prior_fn()

                env.reset(friction=friction)
                idx += 1
        if render:
            env.render()
        env.step()

    log_df = pd.DataFrame(sim_log, columns=['friction', 'initial_x', 'initial_y', 'initial_z', 'final_x', 'final_y', 'final_z', 'distance'])
    stats_df = pd.DataFrame(stats_log, columns=['friction', 'mean', 'var'])
    log_df.to_pickle('log_{}_sims.pickle'.format(num_datapoints))
    stats_df.to_pickle('stats_{}_sims.pickle'.format(num_datapoints))
    sys.exit(0)


def calculate_stats(arr, friction, metric_idx=-1):
    var = np.var(np.array(arr)[:, metric_idx])
    mean = np.mean(np.array(arr)[:, metric_idx])
    return [friction, mean, var]

def load_sims_from_prior(stats_file_name="stats_100000_sims.pickle"):
    df = pd.read_pickle(stats_file_name)
    y_train = df.iloc[:, 0][:, np.newaxis]
    x_train = df.iloc[:, 1][:, np.newaxis]
    return df, x_train, y_train

def train_mdn(x_train, y_train):
    network = MixtureDensityNetwork([20], 2)
    network.fit(x_train, y_train, batch_size=1000, epochs=5000)
    return network

if __name__ == "__main__":
    low = 0.1
    high = 1.0
    torch.manual_seed(1234)
    np.random.seed(1234)
    def uniform_prior():
        return np.random.uniform(low, high, 1)[0]

    #run_sims_from_prior(uniform_prior, render=True)
    df, x_train, y_train = load_sims_from_prior()

    network = MixtureDensityNetwork([20], 2)
    network.fit(x_train, y_train, batch_size=1000, epochs=30000)
    helper.save(network, "friction_mdn.pkl")
    x_obs = torch.from_numpy(np.float32(np.array([0.15])))
    posterior = network.get_mog(x_obs)
    print(posterior.gen())
    print(posterior.gen())
    print(posterior.gen())


