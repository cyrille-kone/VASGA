import pymc3 as pm
import numpy as np
import theano.tensor as tt
import arviz as az


def get_mcmc_sample(X, y, size=5):
    """ MCMC sampling """ 
    prior_location = 0
    prior_scale = 1
    with pm.Model() as model:
        # Prior on the interceptt
        alpha = pm.Normal('alpha', mu=prior_location, sigma=prior_scale)

        # Prior on the coefficient 
        beta = pm.Normal('beta', mu=prior_location, sigma=prior_scale, shape=size)

        # Prior on std
        log_sigma = pm.Normal('log_sigma', mu=prior_location, sigma=prior_scale)

        # linear model
        mu = alpha + pm.math.dot(X, beta)

        # likelihood
        likelihood = pm.Normal('likelihood', mu=mu, sigma=np.exp(log_sigma), observed=y)

        # MCMC sampling
        trace = pm.sample(2000)

    return trace
