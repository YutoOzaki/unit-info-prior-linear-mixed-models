import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def nonlocalpdf(x, mu=0, sigma=1):
    S = (x - mu)**2/sigma**2
    p_normal = (2*np.pi)**(-0.5) * (sigma**2)**(-0.5) * np.exp(-0.5*S)
    return S * p_normal

def mvnonlocalpdf(x, mu, Lambda, q):
    d = mu.shape[0]
    z = x - mu[:, np.newaxis]
    S = (z.T.dot(Lambda)*z.T).sum(axis=1)
    lnp_normal = (-0.5*d)*np.log(2*np.pi) \
               - (-0.5)*np.linalg.slogdet(Lambda)[1] \
               + (-0.5*S)

    qq = np.array(np.meshgrid(q, q)).T.reshape(-1, 2)
    Lambda_q = Lambda[qq[:, 0], qq[:, 1]].reshape([len(q), len(q)])
    Q = (z[q, :].T.dot(Lambda_q)*z[q, :].T).sum(axis=1)

    iLambda = np.linalg.inv(Lambda)
    iLambda_q = iLambda[qq[:, 0], qq[:, 1]].reshape([len(q), len(q)])
    C = np.trace(Lambda_q.dot(iLambda_q))

    lnp_nonlocal = lnp_normal + np.log(Q) - np.log(C)
    return np.exp(lnp_nonlocal), lnp_nonlocal

# Case - univariate
mu = np.random.uniform(-3, 3)
sigma = np.random.uniform(0.01, 2)
x = np.linspace(-9, 9, 4096)
p = nonlocalpdf(x, mu, sigma)
A = sp.integrate.trapezoid(p, x)
proper = np.abs(1 - A) < 1e-5

g = 4*sp.stats.t(df=2, loc=mu, scale=sigma).pdf(x)
covered = np.all(g >= p)

fig, ax = plt.subplots()
plt.plot(x, p)
plt.plot(x, sp.stats.norm(loc=mu, scale=sigma).pdf(x))
plt.plot(x, g)

# Case - multivariate
d = 3
Lambda = sp.stats.wishart(df=d+2, scale=np.eye(d)).rvs()
mu = np.random.uniform(low=-3, high=3, size=d)
K = 10000
M = 10000
a = -9
b = 9
q = ([1], [0, 2], [0, 1, 2])
mvt = sp.stats.multivariate_t(loc=mu, shape=np.linalg.inv(Lambda))

for i in range(len(q)):
    A = 0
    covered_mv = True
    for k in range(K):
        x = np.random.uniform(a, b, (d, M))
        p = mvnonlocalpdf(x, mu, Lambda, q[i])[0]
        g = (4*d)*mvt.pdf(x.T)
        covered_mv = covered_mv & np.all(g > p)

        A += np.sum(p)
    A = A/(K*M)*((b - a)**d)
    print('Monte Carlo integral of pdf: {} (q = {}, MVT covered = {})'.format(A, q[i], covered_mv))

print("Completed")

'''
On the use of non-local prior densities in Bayesian hypothesis tests (2010; Valen E. Johnson, David Rossell)
'''