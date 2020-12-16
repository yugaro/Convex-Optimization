import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.neural_network import MLPRegressor

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
B_plus = [B[i][j] for i in range(n) for j in range(n) if B[i][j] != 0]
print(np.array(B_plus).max() / (10 * D_base_max))
print(np.array(B_plus).min() / (10 * D_base_max))

# load parameters of the event-triggered controller
K = np.load('./data/matrix/K.npy')
print(K / (10 * D_base_max))

L = np.load('./data/matrix/L.npy')
L_plus = [L[i][j] for i in range(n) for j in range(n) if L[i][j] != 0]
print(np.array(L_plus).max() / (10 * D_base_max))
print(np.array(L_plus).min() / (10 * D_base_max))

sigma = np.load('./data/matrix/sigma.npy')
eta = np.load('./data/matrix/eta.npy')


def lyapunov_param_solver(B, D):
    # define SVM
    mlp = MLPRegressor(activation='relu', alpha=0.0001, max_iter=500)

    # define target vector
    target = np.ones(n) * 0.001

    # model fitting
    mlp.fit(B - D, target)

    # calculate Lyapunov param
    p = mlp.predict(B - D).dot(np.linalg.inv(B - D))

    # add bias and implement normalization
    p += np.abs(p.min()) + 1
    p /= np.linalg.norm(p)

    # caluculate rc
    rc = (B.T - D).dot(p)

    return p, rc


def analyse_theta(p, B, D, K, L, G, H):
    # define variables of the state of nodes
    x = cp.Variable(n)

    # define parameter of theorem 1
    s = (K + L.T).dot(In - G).dot(p)
    S = np.diag(s)
    Q = S + 1 / 2 * np.diag(p).dot(L.T).dot(In - G).dot(G + H)
    Q = (Q.T + Q) / 2
    r = ((B - D.T) + (K + L.T).dot(H)).dot(p)
    print(Q)

    # define constraint in theorem 1
    constranit_theta = [x @ Q @ x - r.T @ x <= 0,
                        0 <= x, x <= 1]
    # if np.all(Q):
    #    constranit_theta = [x @ Q @ x - r.T @ x <= 0,
    #                        0 <= x, x <= 1]
    # else:
    #    constranit_theta = [- r.T @ x <= 0,
    #                        0 <= x, x <= 1]

    # define objective function in theorem 1
    theta = p.T @ x

    # solve program of theorem 1 and caluculate theta*
    prob_theta = cp.Problem(cp.Maximize(theta), constranit_theta)
    prob_theta.solve(method='sdp-relax', solver=cp.MOSEK)

    return prob_theta.value

if __name__ == "__main__":

    # caluculate eigenvalue and eigenvector
    eigenvalue, eigenvector = np.linalg.eig(B - D)
    print('eigenvalue (B - D)\n', eigenvalue)

    # caluculate eigenvalue and eigenvector
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if B[i][j] != 0:
                A[i][j] = 1
    eigenvalue, eigenvector = np.linalg.eig(A)
    print('eigenvalue A \n', np.array(eigenvalue).max())

    # caluculate spectral radius (B)
    spectral_radius = np.linalg.norm(B, 2)
    print('spectral radius B \n', spectral_radius)

    # caluculate spectral radius (B - D)
    spectral_radius = np.linalg.norm((B - D), 2)
    print('spectral radius (B - D)\n', spectral_radius)

    # caluculate Lyapunov paramters
    p, rc = lyapunov_param_solver(B, D)

    print('spectral radius D^{-1}B\n', np.linalg.norm(np.linalg.inv(D).dot(B), 2))

    # calculate thetastar
    thetastar = analyse_theta(p, B, D, K, L, np.diag(sigma), np.diag(eta))
    # thetastar = analyse_theta(np.ones(n), B, D, On, On, On, On)
    print('theta*\n', thetastar)
