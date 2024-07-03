import pymc as pm
import pymc.sampling.jax as pmj
import pytensor.tensor as pt
import arviz as az
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def gen_simdata(N, d, C, seed):
    rng = np.random.default_rng(seed=seed)
    Z = rng.multinomial(1, [1 / C] * C, size=N)
    c = np.squeeze(np.concatenate([np.where(Z[i, :] == 1) for i in range(N)]))
    s_0 = rng.lognormal(mean=0, sigma=1.2)
    u = sp.stats.multivariate_normal(mean=np.zeros(C), cov=np.eye(C) * s_0 ** 2, seed=rng).rvs()
    sgm_0 = rng.lognormal(mean=0, sigma=1.2)
    eps = rng.normal(loc=0, scale=sgm_0, size=N)
    be_0 = np.append(rng.uniform(low=-10, high=10, size=2), rng.uniform(low=0.9, high=1.3))
    X = np.stack([np.ones(N), np.zeros(N), rng.lognormal(mean=0, sigma=1.5, size=N) + 50], axis=1)
    for i in range(C):
        idx = np.where(c == i)
        X[idx, 1] = i % 2
    y = X.dot(be_0) + Z.dot(u) + eps

    return y, X, Z, be_0, sgm_0, s_0, u

def run_analysis(y, X, Z):
    N, d = X.shape
    C = Z.shape[1]
    nu_sgm = 2
    A_sgm = 1000
    nu_s = 2
    A_s = 1000
    m = np.zeros(d)
    c = np.where(Z == 1)[1]

    with pm.Model():
        σ = pm.HalfStudentT('sgm', nu=nu_sgm, sigma=A_sgm)
        s = pm.HalfStudentT('s', nu=nu_s, sigma=A_s)
        u = pm.Normal('u', mu=0, sigma=s, size=C)
        V = s**2*Z.dot(Z.T) + σ**2*np.eye(N)
        iV = pm.math.matrix_inverse(V)
        r = pm.Beta('r', alpha=0.01, beta=0.01*N)
        g = 1/r - 1
        Λ = pt.linalg.matrix_dot(X.T, iV, X)/g
        β = pm.MvNormal('be', mu=m, tau=Λ)
        μ = pm.math.matmul(X, β) + u[c]
        pm.Normal('y', mu=μ, sigma=σ, observed=y)

        idata = pmj.sample_numpyro_nuts(draws=1000, chains=4, tune=4000,
                                        target_accept=0.8) #, chain_method="vectorized"
        return idata

if __name__ == "__main__":
    N = 512
    d = 3
    C = 64
    y, X, Z, be_0, sgm_0, s_0, u = gen_simdata(N, d, C, 54)

    idata = run_analysis(y, X, Z)

    az.plot_energy(idata)
    az.plot_trace(idata, filter_vars="regex", var_names=['be', 's', 'u', 'r'])

    fig, ax = plt.subplots()
    plt.plot(be_0)
    plt.plot(np.mean(idata.posterior['be'].to_numpy(), axis=(0, 1)), linestyle="dashed")

    fig, ax = plt.subplots()
    plt.plot(u)
    plt.plot(np.mean(idata.posterior['u'].to_numpy(), axis=(0, 1)), linestyle="dashed")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    posterior_summary = az.summary(idata, round_to=2, filter_vars="regex", var_names=['be', 's', 'u', 'r'])
    print(posterior_summary)

    print('Completed')

'''
General Design Bayesian Generalized Linear Mixed Models (2006; Y. Zhao, J. Staudenmayer, B. A. Coull, M. P. Wand)
Prior Distributions for Objective Bayesian Analysis (2018; Guido Consonni, Dimitris Fouskakis, Brunero Liseo, Ioannis Ntzoufras)
Default Bayesian model determination methods for generalised linear mixed models (2010; Antony M. Overstall, Jonathan J. Forster)
On the safe use of prior densities for Bayesian model selection (2023; F. Llorente, L. Martino, E. Curbelo, J. Lopez-Santiago, D. Delgado)
Simple Marginally Noninformative Prior Distributions for Covariance Matrices (2013; Alan Huang, M. P. Wand)
A Reference Bayesian Test for Nested Hypotheses and its Relationship to the Schwarz Criterion (1995; Robert E. Kass, Larry Wasserman)
Bayes Factors (1995; Robert E. Kass, Adrian E. Raftery)
Bayesian variable and link determination for generalised linear models (2003; Ioannis Ntzoufras, Petros Dellaportas, Jonathan J. Forster)
Minimum Description Length Model Selection Criteria for Generalized Linear Models (2003; Mark H. Hansen, Bin Yu)
Adaptive Bayesian Criteria in Variable Selection for Generalized Linear Models (2007; Xinlei Wang, Edward I. George)
Misinformation in the conjugate prior for the linear model with implications for free-knot spline modelling (2006; Christopher J. Paciorek)
Variance Prior Forms for High-Dimensional Bayesian Variable Selection (2019; Gemma E. Moran, Veronika Rocková, Edward I. George)
Mixtures of g-Priors in Generalized Linear Models (2018; Yingbo Li, Merlise A. Clyde)
Hyper-g Priors for Generalized Linear Models (2011; Daniel Sabanés Bové, Leonhard Held)
Mixtures of g Priors for Bayesian Variable Selection (2008; Feng Liang, Rui Paulo, German Molina, Merlise A. Clyde, Jim O. Berger)
Conjugate Bayesian analysis of the Gaussian distribution (2007; Kevin P. Murphy)
'''