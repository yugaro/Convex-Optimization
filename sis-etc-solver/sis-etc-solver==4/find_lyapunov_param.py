import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rc
import cvxpy as cp
from sklearn.neural_network import MLPRegressor
np.random.seed(0)
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)
mpl.rcParams['axes.xmargin'] = 0
mpl.rcParams['axes.ymargin'] = 0

# number of nodes (max: 50)
n = 50

# preparation
INF = 1e9
epsilon = 1e-15
In = np.identity(n)
On = np.zeros((n, n))

# matrix of recovery rates
D_base_max = 1.8
D_base_min = 1.5
D = np.diag(np.sort((D_base_max - D_base_min) *
                    np.random.rand(n) + D_base_min)[::-1])

# matrix of infection rates (air route matrix in case of the passengers and flights)
B = pd.read_csv('./data/US_Airport_Ad_Matrix.csv',
                index_col=0, nrows=n, usecols=[i for i in range(n + 1)]).values


def lyapunov_param_solver(B, D):
    # define SVM
    mlp = MLPRegressor(activation='relu',
                       early_stopping=False,
                       alpha=0.000100,
                       max_iter=500,)

    # define target vector
    target = (B - D).dot(np.ones(n)) * 0.001
    # print(target)

    # model fitting
    mlp.fit(B - D, target)

    # calculate Lyapunov param
    p = mlp.predict(B - D).dot(np.linalg.inv(B - D))

    print(p)

    # add bias and implement normalization
    # p += np.abs(p.min()) + 1
    p /= np.linalg.norm(p)

    # caluculate rc
    rc = (B.T - D).dot(p)

    print(mlp.score(B - D, target))

    return p, rc


def analyse_theta(p, B, D, K, L, G, H):
    # define variables of the state of nodes
    x = cp.Variable(n)

    # define parameter of theorem 1
    s = (K + L.T).dot(In - G).dot(p)
    S = np.diag(s)
    Q = S + 1 / 2 * np.diag(p).dot(L.T).dot(In - G).dot(G + H)
    r = ((B - D.T) + (K + L.T).dot(H)).dot(p)

    # define constraint in theorem 1
    if np.all(Q):
        constranit_theta = [x @ Q @ x - r.T @ x <= 0,
                            0 <= x, x <= 1]
    else:
        constranit_theta = [- r.T @ x <= 0,
                            0 <= x, x <= 1]

    # define objective function in theorem 1
    theta = p.T @ x

    # solve program of theorem 1 and caluculate theta*
    prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
    prob_theta.solve()

    return prob_theta.value


if __name__ == '__main__':

    # design Lyapunov parameter
    p, rc = lyapunov_param_solver(B, D)
    print(p)
    print(p.dot(B - D))
    theta = analyse_theta(p, B, D, On, On, On, On)
    print(theta)
