import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def mvnloglik(y, mu, Lmd):
    d = y.shape[0]
    z = y - mu
    return -0.5*d*np.log(2*np.pi) + 0.5*np.linalg.slogdet(Lmd)[1] - 0.5*z.T.dot(Lmd).dot(z)

def numhess(be_eval, genmodel, M):
    eps = 1e-6
    H = np.zeros((be_eval.shape[0], be_eval.shape[0]))

    for m in range(M):
        y = genmodel.rvs()

        for i in range(be_eval.shape[0]):
            for j in range(be_eval.shape[0]):
                be_p = be_eval.copy()

                if i == j:
                    be_p[i] = be_eval[i] + eps
                    l_1 = mvnloglik(y, X.dot(be_p), Lmd)
                    l_2 = mvnloglik(y, X.dot(be_eval), Lmd)
                    be_p[i] = be_eval[i] - eps
                    l_3 = mvnloglik(y, X.dot(be_p), Lmd)
                    H[i, j] = H[i, j] + (l_1 - 2 * l_2 + l_3) / (eps ** 2)
                else:
                    be_p[i] = be_eval[i] + eps
                    be_p[j] = be_eval[j] + eps
                    l_1 = mvnloglik(y, X.dot(be_p), Lmd)
                    be_p[i] = be_eval[i] - eps
                    be_p[j] = be_eval[j] + eps
                    l_2 = mvnloglik(y, X.dot(be_p), Lmd)
                    be_p[i] = be_eval[i] + eps
                    be_p[j] = be_eval[j] - eps
                    l_3 = mvnloglik(y, X.dot(be_p), Lmd)
                    be_p[i] = be_eval[i] - eps
                    be_p[j] = be_eval[j] - eps
                    l_4 = mvnloglik(y, X.dot(be_p), Lmd)
                    H[i, j] = H[i, j] + (l_1 - l_2 - l_3 + l_4) / (4 * eps ** 2)
    return H / M

# Case 1
d = 8
y = np.random.normal(size=d)
mu = np.random.normal(size=d)
Lmd = sp.stats.wishart(df=d+2, scale=np.eye(d)).rvs()
l1 = mvnloglik(y, mu, Lmd)
l2 = sp.stats.multivariate_normal(mean=mu, cov=np.linalg.solve(Lmd, np.eye(d))).logpdf(y)
print((l1, l2, l1 - l2))

# Case 2 - prep
N = 64
C = 4
Z = np.random.multinomial(n=1, pvals=np.repeat(1/C, C), size=N)
c = np.where(Z == 1)[1]
X = np.vstack((np.ones(N), np.zeros(N), np.random.lognormal(mean=0, sigma=1.1, size=N)+50)).T
for i in range(C):
    X[c == i, 1] = i % 2
be = np.array((np.random.uniform(-10, 10), np.random.uniform(-10, 10), np.random.uniform(0.8, 1.2)))
sgm = np.random.uniform(0.1, 5)
s = np.random.uniform(0.1, 5)
u = np.random.normal(loc=0, scale=s, size=C)
Sgm = Z.dot(s**2*np.eye(C)).dot(Z.T) + sgm**2*np.eye(N)
Lmd = np.linalg.inv(Sgm)
mu = X.dot(be)

# Case 6
genmodel = sp.stats.multivariate_normal(mean=mu, cov=Sgm)
H = 0
for m in range(2048):
    y = genmodel.rvs()
    s = X.T.dot(Lmd).dot(y - mu)
    s = s[:, np.newaxis]
    H = H + s.dot(s.T)
H = H/2048
I = X.T.dot(Lmd).dot(X)
fig, ax = plt.subplots()
plt.plot(H.flatten())
plt.plot(I.flatten(), linestyle=':')

genmodel = sp.stats.multivariate_normal(mean=np.random.normal(loc=1, scale=10, size=N), cov=Sgm)
H = 0
for m in range(2048):
    y = genmodel.rvs()
    s = X.T.dot(Lmd).dot(y - mu)
    s = s[:, np.newaxis]
    H = H + s.dot(s.T)
H = H/2048
I = X.T.dot(Lmd).dot(X)
fig, ax = plt.subplots()
plt.plot(H.flatten())
plt.plot(I.flatten(), linestyle=':')

# Case 2 - test: true model evaluated at the ML parameter
H = numhess(be_eval=be, genmodel=sp.stats.multivariate_normal(mean=mu, cov=Sgm), M=2048)
I = -X.T.dot(Lmd).dot(X)
fig, ax = plt.subplots()
plt.plot(H.flatten())
plt.plot(I.flatten(), linestyle=':')

# Case 3 - test: true model evaluated at the non-ML parameter
H = numhess(be_eval=np.zeros(be.shape[0]), genmodel=sp.stats.multivariate_normal(mean=mu, cov=Sgm), M=2048)
I = -X.T.dot(Lmd).dot(X)
fig, ax = plt.subplots()
plt.plot(H.flatten())
plt.plot(I.flatten(), linestyle=':')

# Case 4 - test: true model evaluated at the non-ML parameter
Sgm_a = np.random.lognormal(0, 1)*Z.dot(Z.T) + np.random.lognormal(0, 1)*np.eye(N)
H = numhess(be_eval=np.zeros(be.shape[0]), genmodel=sp.stats.multivariate_normal(mean=mu, cov=Sgm_a), M=2048)
I = -X.T.dot(Lmd).dot(X)
fig, ax = plt.subplots()
plt.plot(H.flatten())
plt.plot(I.flatten(), linestyle=':')

# Case 5 - test: misspecified model
X_a = np.hstack((X, np.random.gamma(shape=0.5, scale=1.2, size=N)[:, np.newaxis]))
be_a = np.append(be, np.random.normal())
mu_a = X_a.dot(be_a)
Sgm_a = np.random.lognormal(0, 1)*Z.dot(Z.T) + np.random.lognormal(0, 1)*np.eye(N)
H = numhess(be_eval=np.zeros(be.shape[0]), genmodel=sp.stats.multivariate_normal(mean=mu_a, cov=Sgm_a), M=2048)
I = -X.T.dot(Lmd).dot(X)
fig, ax = plt.subplots()
plt.plot(H.flatten())
plt.plot(I.flatten(), linestyle=':')

# Case 6 - test: misspecified model
H = numhess(be_eval=be, genmodel=sp.stats.gamma(a=2, scale=np.random.uniform(low=0.1, high=3, size=N)), M=2048)
I = -X.T.dot(Lmd).dot(X)
fig, ax = plt.subplots()
plt.plot(H.flatten())
plt.plot(I.flatten(), linestyle=':')

print("Completed")