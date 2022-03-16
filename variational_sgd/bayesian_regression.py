import pymc3 as pm
import numpy as np
import theano.tensor as tt
import arviz as az


def get_mcmc_sample(X, y, size=5):
    prior_location = 0
    prior_scale = 1
    with pm.Model() as model:
        # Prior sur l'intercept
        alpha = pm.Normal('alpha', mu=prior_location, sigma=prior_scale)

        # Prior sur les coefficients de la régression linéaire
        beta = pm.Normal('beta', mu=prior_location, sigma=prior_scale, shape=size)

        # Prior sur l'écart type des résultats
        log_sigma = pm.Normal('log_sigma', mu=prior_location, sigma=prior_scale)

        # Régression linéaire
        mu = alpha + pm.math.dot(X, beta)

        # Vraissemblance
        likelihood = pm.Normal('likelihood', mu=mu, sigma=np.exp(log_sigma), observed=y)

        # MCMC
        trace = pm.sample(2000)

    return trace