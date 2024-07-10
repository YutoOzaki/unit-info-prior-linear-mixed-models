import pymc as pm
import pymc.sampling.jax as pmj
import pytensor
import pytensor.tensor as pt
from pytensor.tensor import TensorVariable
from pymc.distributions.distribution import Continuous
from pytensor.tensor.random.op import RandomVariable
from typing import Optional, Tuple, List
import arviz as az
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

class MnNonLocalRV(RandomVariable):
    name: str = "mvnonlocal"
    ndim_supp: int = 1
    ndims_params: List[int] = [1, 2]
    signature: '(),() -> ()'
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("mvnonlocal", "\\operatorname{mvnonlocal}")

    def _supp_shape_from_params(self, dist_params, param_shapes):
        return dist_params[0].shape[0]

    def _infer_shape(self, size, dist_params):
        return dist_params[0].shape

    @classmethod
    def rng_fn(
            cls,
            rng: np.random.RandomState,
            mu: np.ndarray,
            Lambda: np.ndarray,
            size: Tuple[int, ...],
    ) -> np.ndarray:
        g = sp.stats.multivariate_normal(mean=mu, tau=Lambda, seed=rng)
        iLambda = np.linalg.inv(Lambda)
        f = sp.stats.multivariate_t(df=2, shape=iLambda, seed=rng)
        C = Lambda[1, 1] * iLambda[1, 1]

        Y = f.rvs(size)
        r = (Y[1] ** 2) * Lambda[1, 1] / C * g.pdf(Y) / (4 * d * f.pdf(Y))
        u = np.random.uniform(0, 1)
        while np.all(u >= r):
            Y = f.rvs()
            r = (Y[1] ** 2) * Lambda[1, 1] / C * g.pdf(Y) / (4 * d * f.pdf(Y))
            u = np.random.uniform(0, 1)

        return Y

mvnonlocal = MnNonLocalRV()

class MvNonLocal(Continuous):
    rv_op = mvnonlocal

    @classmethod
    def dist(cls, mu=0, Lambda=None, **kwargs):
        mu = pt.as_tensor_variable(mu)
        Lambda = pt.as_tensor_variable(Lambda)
        return super().dist([mu, Lambda], **kwargs)

    def logp(value: TensorVariable, mu: TensorVariable, Lambda: TensorVariable) -> TensorVariable:
        z = value - mu
        _, logdetLambda = pt.nlinalg.slogdet(Lambda)

        iLambda = np.linalg.inv(Lambda)
        C = Lambda[1, 1] * iLambda[1, 1]

        return (pm.math.log(z[1] ** 2 * Lambda[1, 1]) - pm.math.log(C)
                - 0.5 * logdetLambda - 0.5 * pt.linalg.matrix_dot(z.T, Lambda, z))

    def random(mu: np.ndarray | float, Lambda: np.ndarray | float,
                        rng: Optional[np.random.Generator] = None,
                        size: Optional[Tuple[int]] = None) -> np.ndarray | float:
        g = sp.stats.multivariate_normal(mean=mu, tau=Lambda, seed=rng)
        iLambda = np.linalg.inv(Lambda)
        f = sp.stats.multivariate_t(df=2, shape=iLambda, seed=rng)
        C = Lambda[1, 1] * iLambda[1, 1]

        Y = f.rvs(size)
        r = (Y[1] ** 2) * Lambda[1, 1] / C * g.pdf(Y) / (4 * d * f.pdf(Y))
        u = np.random.uniform(0, 1)
        while np.all(u >= r):
            Y = f.rvs()
            r = (Y[1] ** 2) * Lambda[1, 1] / C * g.pdf(Y) / (4 * d * f.pdf(Y))
            u = np.random.uniform(0, 1)

        return Y

    def support(rv: TensorVariable, size: TensorVariable, mu: TensorVariable,
                               Lambda: TensorVariable) -> TensorVariable:
        return mu + np.sqrt(2) / pt.sqrt(pt.diagonal(Lambda))

def gen_simdata(N, d, C, seed=None):
    rng = np.random.default_rng(seed=seed)

    Z = rng.multinomial(1, [1/C]*C, size=N)
    r, c = np.where(Z == 1)
    sortorder = np.hstack([r[c == i] for i in range(C)])
    Z = Z[sortorder, :]

    s_0 = rng.lognormal(mean=0, sigma=1.2)
    u = sp.stats.multivariate_normal(mean=np.zeros(C), cov=np.eye(C)*s_0**2, seed=rng).rvs()
    sgm_0 = rng.lognormal(mean=0, sigma=1.2)
    eps = rng.normal(loc=0, scale=sgm_0, size=N)
    be_0 = np.append(rng.uniform(low=-10, high=10, size=2), rng.uniform(low=0.9, high=1.3))
    X = np.stack([np.ones(N), np.zeros(N), rng.lognormal(mean=0, sigma=1.5, size=N) + 50], axis=1)
    _, c = np.where(Z == 1)
    for i in range(C):
        idx = np.where(c == i)
        X[idx, 1] = i % 2

    y = X.dot(be_0) + Z.dot(u) + eps

    testfisherinfoconstruction(X, Z, sgm_0, s_0)

    return y, X, Z, be_0, sgm_0, s_0, u

def testfisherinfoconstruction(X, Z, sgm, s):
    N, d = X.shape
    _, C = Z.shape

    R = sgm**2 * np.eye(N)
    D = s**2 * np.eye(C)
    V = Z.dot(D).dot(Z.T) + R
    iV = np.linalg.inv(V)
    A = X.T.dot(iV).dot(X)

    a = sgm**2
    b = s**2
    Λ = np.zeros([d, d])
    offset = 0
    for i in range(C):
        n = sum(Z[:, i] == 1)
        X_i = X[offset:(offset + n), :]
        iV_i = 1/a * np.eye(n) - b/(a * (a + b*n)) * np.ones([n, n])
        Λ += np.linalg.multi_dot([X_i.T, iV_i, X_i])
        offset += n

    print("Valid Fisher information matrix construction: {}".format(np.allclose(A, Λ)))

def run_analysis(y, X, Z):
    N, d = X.shape
    _, C = Z.shape
    nu_sgm = 2
    A_sgm = 1000
    nu_s = 2
    A_s = 1000
    m = np.zeros(d)
    _, c = np.where(Z == 1)

    with pm.Model():
        σ = pm.HalfStudentT('sgm', nu=nu_sgm, sigma=A_sgm)
        s = pm.HalfStudentT('s', nu=nu_s, sigma=A_s)
        u = pm.Normal('u', mu=0, sigma=s, size=C)
        r = pm.Beta('r', alpha=0.01, beta=0.01*N)
        g = 1/r - 1

        a = σ**2
        b = s**2
        #'''
        Λ = pytensor.shared(np.zeros([d, d]))
        offset = 0
        for i in range(C):
            n = sum(Z[:, i] == 1)
            X_i = X[offset:(offset + n), :]
            iV_i = 1/a * np.eye(n) - b/(a*(a + b*n)) * np.ones([n, n]) # manually construct precision matrices to avoid using pm.math.matrix_inverse(V) and its slowness
            Λ += pt.linalg.matrix_dot(X_i.T, iV_i, X_i)
            offset += n
        Λ = Λ/g
        #'''

        '''
        offset = 0
        X_list = [None]*C
        n_list = np.zeros(C)
        offset_list = np.zeros(C)
        for i in range(C):
            n = sum(Z[:, i] == 1)
            X_list[i] = X[offset:(offset + n), :]
            n_list[i] = n
            offset_list[i] = offset
            offset += n

        def oneStep(n, offset, Λ_tm, X, a, b):
            X_i = X[offset:(offset + n), :]
            iV_i = 1/a * pt.eye(n) - b/(a*(a + b*n)) * pt.ones(shape=[n, n])
            Λ = Λ_tm + pt.linalg.matrix_dot(X_i.T, iV_i, X_i)
            return Λ

        Λ_ini = pt.as_tensor_variable(np.zeros([d, d]))
        n_tv = pt.as_tensor_variable(n_list.astype('int32'))
        offset_tv = pt.as_tensor_variable(offset_list.astype('int32'))
        Λ, _ = pytensor.scan(fn=oneStep, outputs_info=Λ_ini,
                             sequences=[n_tv, offset_tv],
                             non_sequences=[pt.as_tensor_variable(X), a, b],
                             strict=True)
        Λ = Λ[-1]/g
        '''
        β = MvNonLocal('be', mu=m, Lambda=Λ)
        #β = pm.MvNormal('be', mu=m, cov=Λ)

        μ = pm.math.matmul(X, β) + u[c]
        pm.Normal('y', mu=μ, sigma=σ, observed=y)

        idata = pmj.sample_numpyro_nuts(draws=1000, chains=4, tune=9000, target_accept=0.8)
        #idata = pm.sample(draws=1000, chains=4, tune=4000, target_accept=0.8, cores=1)
        return idata

if __name__ == "__main__":
    N = 512
    d = 3
    C = 64
    y, X, Z, be_0, sgm_0, s_0, u = gen_simdata(N, d, C, 90)

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