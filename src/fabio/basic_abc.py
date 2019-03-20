import numpy as np


def rejection_abc_posterior_sampler(y, prior_sampler, likelihood_sampler, distance_function, eps,
                                    n_samples=10000, n_iter=10000, verbose=True):
    
    posterior_samples = []
    calls_history = []
    
    for i in range(n_iter):
        
        theta = prior_sampler()
        x = likelihood_sampler(theta)
        
        if distance_function(x, y) < eps:
            posterior_samples.append(theta)
            calls_history.append(likelihood_sampler.calls)
            if verbose:
                print('Accepted %d samples - currently used %d simulation calls' % (len(posterior_samples), likelihood_sampler.calls))
            
        if len(posterior_samples) >= n_samples:
            break
    
    return np.array(posterior_samples), np.array(calls_history)
    

# def mcmc_abc_posterior_sampler(y, prior_sampler, prior_density, likelihood_sampler, distance_function, eps,
#                                proposal_sampler, proposal_density, n_iter=1000):
    
#     theta = rejection_abc_posterior_sampler(y, prior_sampler, likelihood_sampler, distance_function, eps,
#                                             n_samples=1, n_iter=10000)
    
#     posterior_samples = [theta]
    
#     for i in range(n_iter):
        
#         theta_proposal = proposal_sampler(theta)
#         x = likelihood_sampler(theta_proposal)
#         u = np.random.rand()
#         mh_ratio = prior_density(theta_proposal) * proposal_density(theta, theta_proposal) / (prior_density(theta) * proposal_density(theta_proposal, theta))
        
#         if u < mh_ratio and distance_function(x, y) < eps:
#             posterior_samples.append(theta_proposal)
#         else:
#             posterior_samples.append(theta)
            
#     return np.array(posterior_samples)


def gaussian_mcmc_abc_posterior_sampler(y, prior_sampler, prior_density, likelihood_sampler, distance_function, eps,
                                        proposal_std, n_iter=1000, verbose=True):
    
    theta = rejection_abc_posterior_sampler(y, prior_sampler, likelihood_sampler, distance_function, eps,
                                            n_samples=1, n_iter=10000, verbose=True)[0][0]
    
    posterior_samples = []
    calls_history = []
    
    for i in range(n_iter):
        
        theta_proposal = theta + np.random.randn(theta.shape[0]) * proposal_std
        x = likelihood_sampler(theta_proposal)
        u = np.random.rand()
        mh_ratio = prior_density(theta_proposal) / prior_density(theta)
        
        if u < mh_ratio and distance_function(x, y) < eps:
            posterior_samples.append(theta_proposal)
            theta = theta_proposal
            print('Sample %d accepted' % i) if verbose else None
        else:
            posterior_samples.append(theta)
            print('Sample %d rejected' % i) if verbose else None
        calls_history.append(likelihood_sampler.calls)
            
    return np.array(posterior_samples), np.array(calls_history)