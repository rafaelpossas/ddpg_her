import numpy as np
from scipy import stats
from src.fabio.MDN import MDNN, MDRFF
from delfi.generator import Default
import delfi.distribution as dd
from scipy.stats import multivariate_normal

class BayesSim(object):
    def __init__(self,
                 env=None,
                 model=None,
                 obs=None,
                 generator=Default,
                 n_components=1,
                 seed=None,
                 verbose=True,
                 prior_norm=False,
                 #init_norm=False,
                 pilot_samples=50,
                 ):

        self.env = env
        self.generator = generator # generates the data
        self.generator.proposal = None
        self.obs = obs # observation
        self.n_components = n_components # number of components for the mdn
        self.seed = seed
        self.verbose = verbose
        self.round = 0
        self.model = model

        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

        # gets the dimensionality from the simulator by generating one sample
        params, stats = self.generator.gen(1, skip_feedback=True, verbose=False)
        dim = params.shape[1]

        # MDN model
        if model is None:
            self.model = MDNN(ncomp=self.n_components, outputd=dim,
                              inputd=stats.shape[1], nhidden=2, nunits=[24, 24])


        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        # parameters for z-transform of params
        if prior_norm:
            # z-transform for params based on prior
            self.params_mean = self.generator.prior.mean
            self.params_std = self.generator.prior.std
        else:
            # parameters are set such that z-transform has no effect
            self.params_mean = np.zeros((params.shape[1],))
            self.params_std = np.ones((params.shape[1],))

        # parameters for z-transform for stats
        if pilot_samples is not None and pilot_samples != 0:
            # determine via pilot run
            if seed is not None:  # reseed generator for consistent inits
                self.generator.reseed(self.gen_newseed())
            self.pilot_run(pilot_samples)
        else:
            # parameters are set such that z-transform has no effect
            self.stats_mean = np.zeros((stats.shape[1],))
            self.stats_std = np.ones((stats.shape[1],))

    def gen(self, n_samples, prior_mixin=0, verbose=None):
        """Generate from generator and z-transform

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        n_reps : int
            Number of repeats per parameter
        verbose : None or bool or str
            If None is passed, will default to self.verbose
        """
        verbose = self.verbose if verbose is None else verbose
        params, stats = self.generator.gen(n_samples, prior_mixin=prior_mixin, verbose=verbose)

        # z-transform params and stats
        #params = (params - self.params_mean) / self.params_std
        #stats = (stats - self.stats_mean) / self.stats_std
        return params, stats

    def run(self, n_train=500, epochs=1000, n_rounds=2):
        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        epochs: int
            Number of epochs used for neural network training

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of posteriors
            posterior after each round
        """

        logs = []
        trn_datasets = []
        posteriors = []

        for r in range(n_rounds):  # start at 1
            self.round += 1

            # if round > 1, set new proposal distribution before sampling
            if self.round > 1:
                # posterior becomes new proposal prior
                posterior = self.predict(self.obs)
                self.generator.proposal = posterior.project_to_gaussian()
            # number of training examples for this round
            if type(n_train) == list:
                try:
                    n_train_round = n_train[self.round - 1]
                except:
                    n_train_round = n_train[-1]
            else:
                n_train_round = n_train

            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(r) if self.verbose else False
            trn_data = self.gen(n_train_round, verbose=verbose)
            _, log = self.model.train(x_data=trn_data[1].T,
                                      y_data=trn_data[0].T,
                                      nepoch=epochs)

            logs.append({'loss': log})
            trn_datasets.append(trn_data)

            try:
                posteriors.append(self.predict(self.obs.reshape(1, -1)))
            except:
                posteriors.append(None)
                print('analytic correction for proposal seemingly failed!')
                break

        return logs, trn_datasets, posteriors



    def predict(self, x, threshold=0.005):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        x = x.reshape(1, -1)
        if self.generator.proposal is None:
            # no correction necessary
            return self.model.predict_mog(x)[0]  # via super
        else:
            # mog is posterior given proposal prior
            mog = self.model.predict_mog(x)[0]  # via super
            mog.prune_negligible_components(threshold=threshold)

            # compute posterior given prior by analytical division step
            if isinstance(self.generator.prior, dd.Uniform):
                posterior = mog / self.generator.proposal
            elif isinstance(self.generator.prior, dd.Gaussian):
                posterior = (mog * self.generator.prior) / \
                    self.generator.proposal
            else:
                raise NotImplemented
            return posterior

    def gen_newseed(self):
        """Generates a new random seed"""
        if self.seed is None:
            return None
        else:
            return self.rng.randint(0, 2**31)

    def pilot_run(self, n_samples):
        """Pilot run in order to find parameters for z-scoring stats
        """
        verbose = '(pilot run) ' if self.verbose else False
        params, stats = self.generator.gen(n_samples, verbose=verbose)
        self.stats_mean = np.nanmean(stats, axis=0)
        self.stats_std = np.nanstd(stats, axis=0)