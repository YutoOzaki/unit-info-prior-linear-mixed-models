import pymc as pm
import pytensor
import pytensor.tensor as pt
import nutpie
from pytensor.tensor import TensorVariable
from typing import Optional, Tuple
import arviz as az
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime


def gen_simdata(N, d, C, seed=None):
    rng = np.random.default_rng(seed=seed)

    Z = rng.multinomial(1, [1 / C] * C, size=N)
    r, c = np.where(Z == 1)
    sortorder = np.hstack([r[c == i] for i in range(C)])
    Z = Z[sortorder, :]

    s_0 = rng.lognormal(mean=0, sigma=1.2)
    u = sp.stats.multivariate_normal(mean=np.zeros(C), cov=np.eye(C) * s_0 ** 2, seed=rng).rvs()
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

    R = sgm ** 2 * np.eye(N)
    D = s ** 2 * np.eye(C)
    V = Z.dot(D).dot(Z.T) + R
    iV = np.linalg.inv(V)
    A = X.T.dot(iV).dot(X)

    a = sgm ** 2
    b = s ** 2
    Λ = np.zeros([d, d])
    offset = 0
    for i in range(C):
        n = sum(Z[:, i] == 1)
        X_i = X[offset:(offset + n), :]
        iV_i = 1 / a * np.eye(n) - b / (a * (a + b * n)) * np.ones([n, n])
        Λ += np.linalg.multi_dot([X_i.T, iV_i, X_i])
        offset += n

    print("Valid Fisher information matrix construction (1): {}".format(np.allclose(A, Λ)))

    offset = 0
    n_list = np.zeros(C)
    offset_list = np.zeros(C)
    for i in range(C):
        n = sum(Z[:, i] == 1)
        n_list[i] = n
        offset_list[i] = offset
        offset += n

    def oneStep(n, offset, Λ_tm, X, a, b):
        X_i = X[offset:(offset + n), :]
        iV_i = 1 / a * pt.eye(n) - b / (a * (a + b * n)) * pt.ones(shape=[n, n])
        Λ = Λ_tm + pt.linalg.matrix_dot(X_i.T, iV_i, X_i)
        return Λ

    Λ_ini = pt.as_tensor_variable(np.zeros([d, d]))
    n_tv = pt.as_tensor_variable(n_list.astype('int32'))
    offset_tv = pt.as_tensor_variable(offset_list.astype('int32'))
    Λ, _ = pytensor.scan(fn=oneStep, outputs_info=Λ_ini,
                         sequences=[n_tv, offset_tv],
                         non_sequences=[pt.as_tensor_variable(X), a, b],
                         strict=True)
    Λ = Λ[-1]

    print("Valid Fisher information matrix construction (2): {}".format(np.allclose(A, Λ.eval())))


def logp_nonlocal(value: TensorVariable, mu: TensorVariable, Lambda: TensorVariable) -> TensorVariable:
    m_0 = 0

    z = value - mu
    logdetLambda = pm.math.logdet(Lambda)
    iLambda = pm.math.matrix_inverse(Lambda)
    C = Lambda[1, 1] * iLambda[1, 1]

    return (pm.math.log((value[1] - m_0) ** 2 * Lambda[1, 1]) - pm.math.log(C)
            + 0.5 * logdetLambda - 0.5 * pt.linalg.matrix_dot(z.T, Lambda, z))


def random_nonlocal(mu: np.ndarray | float, Lambda: np.ndarray | float,
                    rng: Optional[np.random.Generator] = None,
                    size: Optional[Tuple[int]] = None) -> np.ndarray | float:
    m_0 = 0

    d = mu.shape[0]
    iLambda = np.linalg.inv(Lambda)
    g = sp.stats.multivariate_normal(mean=mu, cov=iLambda, seed=rng)
    f = sp.stats.multivariate_t(df=2, loc=mu, shape=iLambda, seed=rng)
    C = Lambda[1, 1] * iLambda[1, 1]

    Y = f.rvs()
    u = np.random.uniform(0, 4 * d * f.pdf(Y))
    p = ((Y[1] - m_0) ** 2) * Lambda[1, 1] / C * g.pdf(Y)
    while u > p:
        Y = f.rvs()
        u = np.random.uniform(0, 4 * d * f.pdf(Y))
        p = ((Y[1] - m_0) ** 2) * Lambda[1, 1] / C * g.pdf(Y)

    return Y


def support_point_nonlocal(rv: TensorVariable, size: TensorVariable, mu: TensorVariable,
                           Lambda: TensorVariable) -> TensorVariable:
    return mu + np.sqrt(2) / pt.sqrt(pt.diagonal(Lambda))


def logPowNormal(y, mu, sigma, beta):
    return beta*(-0.5*pm.math.log(2*np.pi) - pm.math.log(sigma) - 0.5*(y - mu)**2/sigma**2)


nu_sgm = 2
A_sgm = 1000
nu_s = 2
A_s = 1000
def run_analysis(y, X, Z, thermobeta):
    N, d = X.shape
    _, C = Z.shape
    _, c = np.where(Z == 1)
    m_be = np.zeros(d)

    with pm.Model() as mymodel:
        σ = pm.HalfStudentT('sgm', nu=nu_sgm, sigma=A_sgm)
        s = pm.HalfStudentT('s', nu=nu_s, sigma=A_s)
        u = pm.Normal('u', mu=0, sigma=s, size=C)
        r = pm.Beta('r', alpha=0.01, beta=0.01 * N)
        g = 1/r - 1

        a = σ ** 2
        b = s ** 2

        # ''' scan version
        offset = 0
        n_list = np.zeros(C)
        offset_list = np.zeros(C)
        for i in range(C):
            n = sum(Z[:, i] == 1)
            n_list[i] = n
            offset_list[i] = offset
            offset += n

        def oneStep(n, offset, Λ_tm, X, a, b):
            X_i = X[offset:(offset + n), :]
            iV_i = 1 / a * pt.eye(n) - b / (a * (a + b * n)) * pt.ones(shape=[n, n])
            Λ = Λ_tm + pt.linalg.matrix_dot(X_i.T, iV_i, X_i)
            return Λ

        Λ_ini = pt.as_tensor_variable(np.zeros([d, d]))
        n_tv = pt.as_tensor_variable(n_list.astype('int32'))
        offset_tv = pt.as_tensor_variable(offset_list.astype('int32'))
        Λ, _ = pytensor.scan(fn=oneStep, outputs_info=Λ_ini,
                             sequences=[n_tv, offset_tv],
                             non_sequences=[pt.as_tensor_variable(X), a, b],
                             strict=True)

        Λ = Λ[-1] / g
        # '''

        # '''
        β = pm.CustomDist('be', m_be, Λ, logp=logp_nonlocal,
                          random=random_nonlocal, support_point=support_point_nonlocal,
                          signature='(n),(n,n)->(n)')
        # '''

        μ = pm.math.matmul(X, β) + u[c]
        #pm.Normal('y', mu=μ, sigma=σ, observed=y)**thermobeta
        pm.Potential("y", logPowNormal(y, μ, σ, thermobeta))

    # '''
    compiled_model = nutpie.compile_pymc_model(mymodel)
    idata = nutpie.sample(compiled_model=compiled_model, draws=1000, tune=9000, chains=4)
    return idata
    # '''


if __name__ == "__main__":
    # test data
    N = 256
    d = 3
    C = 32
    y, X, Z, be_0, sgm_0, s_0, u = gen_simdata(N, d, C)

    # stepping-stone algorithm variables
    K = 10
    al = 0.25
    thermobeta = np.linspace(0, 1, K + 1) ** (1/al)
    y_pt = pt.vector()
    mu_pt = pt.vector()
    sgm_pt = pt.scalar()
    be_pt = pt.scalar()
    logpowlik = logPowNormal(y_pt, mu_pt, sgm_pt, be_pt)
    _, c = np.where(Z == 1)

    # inference
    lnr = np.zeros(K)
    for k in range(K):
        if thermobeta[k] > 0:
            print('{}: analysis start (k = {}/{})'.format(datetime.now(), k, K))
            idata = run_analysis(y, X, Z, thermobeta[k])
            print('{}: analysis end (k = {}/{})'.format(datetime.now(), k, K))

            be_pos = idata.posterior['be'].to_numpy()
            numchain, numsample, d = be_pos.shape
            be_pos = be_pos.reshape([numchain*numsample, d])
            sgm_pos = idata.posterior['sgm'].to_numpy().reshape([numchain*numsample])
            u_pos = idata.posterior['u'].to_numpy().reshape([numchain*numsample, C])
        else:
            numchain = 4
            numsample = 1000
            M = numchain*numsample
            m_be = np.zeros(d)
            sgm_pos = A_sgm*np.abs(np.random.standard_t(df=nu_sgm, size=M))
            s = A_s*np.abs(np.random.standard_t(df=nu_s, size=M))
            u_pos = np.vstack([np.random.normal(loc=0, scale=s[i], size=C) for i in range(M)])
            g = 1/np.random.beta(a=0.01, b=0.01*N, size=M) - 1
            ZZ = Z.dot(Z.T)
            I = np.eye(N)
            Lmd = [g[i]*np.linalg.inv(X.T.dot(np.linalg.inv(s[i]*ZZ + sgm_pos[i]*I)).dot(X)) for i in range(M)]
            be_pos = np.vstack([sp.stats.multivariate_normal(mean=m_be, cov=Lmd[i]).rvs() for i in range(M)])

        l_theta = np.zeros(numchain*numsample)
        dbe = thermobeta[k + 1] - thermobeta[k]
        for m in range(numchain*numsample):
            mu_pos = X.dot(be_pos[m, :]) + u_pos[m, c]
            l_theta[m] += np.sum(logpowlik.eval({y_pt: y, mu_pt: mu_pos, sgm_pt: sgm_pos[m], be_pt: dbe}))
        lnr[k] = np.max(l_theta) + np.log(np.sum(np.exp(l_theta - np.max(l_theta)))) - np.log(numchain*numsample)

    lnZ = np.sum(lnr)
    print("lnZ (Stepping-stone sampling): {}".format(lnZ))

    az.plot_energy(idata)
    plt.savefig("/Users/yutoozaki/Desktop/energy.png")
    az.plot_trace(idata, var_names=['be', 's', 'sgm', 'u', 'r'])
    plt.savefig("/Users/yutoozaki/Desktop/trace.png")

    fig, ax = plt.subplots()
    plt.plot(be_0)
    plt.plot(np.mean(idata.posterior['be'].to_numpy(), axis=(0, 1)), linestyle="dashed")
    plt.savefig("/Users/yutoozaki/Desktop/be.png")

    fig, ax = plt.subplots()
    plt.plot(u)
    plt.plot(np.mean(idata.posterior['u'].to_numpy(), axis=(0, 1)), linestyle="dashed")
    plt.savefig("/Users/yutoozaki/Desktop/u.png")

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    posterior_summary = az.summary(idata, round_to=2, filter_vars="regex", var_names=['be', 's', 'u', 'r'])
    print(posterior_summary)

    print('sgm_0 = {}, s_0 = {}'.format(sgm_0, s_0))
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