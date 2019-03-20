import os
import sys

py_path = os.path.split(os.getcwd())[0]
if py_path not in sys.path:
    sys.path.append(py_path)

import click
import numpy as np
import pickle

from src.utils import logger
from src.utils.util import set_global_seeds
import src.config as config
from src.buffer.episode_rollout import EpisodeRollout


@click.command()
@click.argument('policy_file', type=str)
@click.option('--seed', type=int, default=10)
@click.option('--n_test_rollouts', type=int, default=1000)
@click.option('--render', type=int, default=1)
def main(policy_file, seed, n_test_rollouts, render):
    set_global_seeds(seed)

    # Load policy.
    with open(policy_file, 'rb') as f:
        policy = pickle.load(f)
    env_name = policy.info['env_name']

    # Prepare params.
    params = config.DEFAULT_PARAMS
    if env_name in config.DEFAULT_ENV_PARAMS:
        params.update(config.DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in
    params['env_name'] = env_name
    params = config.prepare_params(params)
    config.log_params(params, logger=logger)

    dims = config.configure_dims(params)

    eval_params = {
        'exploit': True,
        'use_target_net': params['test_with_polyak'],
        'compute_Q': True,
        'rollout_batch_size': 1,
        'render': bool(render),
    }

    for name in ['max_episode_steps', 'gamma', 'noise_eps', 'random_eps']:
        eval_params[name] = params[name]

    evaluator = EpisodeRollout(params['make_env'], policy, dims, logger, **eval_params)
    evaluator.seed(seed)
    friction_arr = [0.05, 0.6, 0.1, 0.8, 0.2, 1.0, 0.25]

    # Run evaluation.
    evaluator.clear_history()
    friction_idx = 0
    all_episodes = []
    for _ in range(n_test_rollouts):

        # if friction_idx == len(friction_arr):
        #     friction_idx = 0
        # if _ % 5 == 0:
        #     evaluator.seed(seed)
        #     friction = friction_arr[friction_idx]
        #     friction_idx += 1
        #     evaluator.set_physics(param=friction)
        episode = evaluator.generate_rollouts()
        all_episodes.append(episode)


    # record logs
    for key, val in evaluator.logs('test'):
        logger.record_tabular(key, np.mean(val))
    logger.dump_tabular()


if __name__ == '__main__':
    main()
