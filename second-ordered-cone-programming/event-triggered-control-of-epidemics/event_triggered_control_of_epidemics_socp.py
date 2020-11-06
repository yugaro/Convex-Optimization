import numpy as np
import cvxpy as cp
import math
from matplotlib import pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font', **{'family': "sans-serif"})
params = {'text.latex.preamble': [r'\usepackage{amsmath}']}
plt.rcParams.update(params)


# preparation
n = 5
m = 3
INF = 1e9
np.random.seed(2)
I = np.ones(n)
I_n = np.identity(n)
O = np.zeros(n)
O_n = np.zeros((n, n))
O_n31 = np.zeros(3 * n + 1)
epsilon = 1e-9

# Problem data.
W = np.random.randint(0, 2, (m, n))
d = np.array([0.1, 0.070, 0.060])
d_table = []
for i in range(W.shape[0]):
    flag = 0
    d_table += [d[i]]
    for j in range(W.shape[1]):
        if W[i][j] == 1:
            flag += 1
    d[i] *= flag
A = np.random.rand(n, n)
B = np.diag(np.random.rand(n)) / 2
D = np.diag(np.random.rand(n))
A2 = np.random.rand(n, n)
D2 = np.diag(np.random.rand(n)) * 40
for i in range(n):
    A[i][i] = 0
    for j in range(n):
        if A[i][j] == 0:
            A2[i][j] = 0
        while A[i][j] < A2[i][j] or A[i][j] - A2[i][j] >= 0.1:
            A2[i][j] = np.random.rand()


def lyapunob_parameter_solver():
    # define variable
    p_v = cp.Variable(n)
    # create constraints
    sq_constraints = [0.1 <= p_v]
    gamma = np.ones(n) * (-1)
    # create objective func and solve least squares method
    sq_objective = cp.sum_squares((B.dot(A) - D) @ p_v - gamma)
    sq_prob = cp.Problem(cp.Minimize(sq_objective), sq_constraints)
    sq_prob.solve()
    # return lyapunob parameter and r
    p = np.array(p_v.value)
    p = p / np.linalg.norm(p)
    r = (B.dot(A) - D).dot(p)
    return p, r


def p_min_supp_w(p):
    p_min = []
    for i in range(m):
        tmp_p = INF
        for j in range(n):
            if W[i][j] == 1 and tmp_p > p[j]:
                tmp_p = p[j]
        p_min += [tmp_p]
    return p_min


def trigger_parameter_solver_gp(p, K, L):
    # define variable
    sigma_v = cp.Variable(n, pos=True)
    eta_v = cp.Variable(n, pos=True)
    r_v = cp.Variable(n, pos=True)
    s_v = cp.Variable(n, pos=True)
    t1 = cp.Variable(n, pos=True)
    t2 = cp.Variable(1, pos=True)

    # create constraints
    # gp_constraints0 = [eta_v[i] == 1e-20 for i in range(n)]

    # # sigma <= 1 - epsilon
    gp_constraints1 = [sigma_v[i] <= 1 - epsilon for i in range(n)]

    # # eta <= 1 - epsilon
    gp_constraints2 = [eta_v[i] <= 1 - epsilon for i in range(n)]

    # # sigma + 1 / t1 <= 1
    gp_constraints3 = [sigma_v[i] + 1 / t1[i] <= 1 for i in range(n)]

    # # s * t1 <= c1
    c1 = []
    for i in range(n):
        for j in range(n):
            tmp1 = p[i] * K[i][i]
            if i != j:
                tmp1 += p[j] * B[j][j] * L[j][i]
            c1 += [tmp1]
    gp_constraints4 = [s_v[i] * t1[i] <= c1[i] for i in range(n)]

    # # c2 + c1 * eta <= r (i in C)
    # # c1 * eta <= r (i notin C)
    gp_constraints5 = []
    for i in range(n):
        for j in range(n):
            c2 = - p[i] * D[i][i]
            if i != j:
                c2 += p[j] * B[j][j] * A[j][i]
        if c2 >= 0:
            gp_constraints5 += [c2 + c1[i] * eta_v[i] <= r_v[i]]
        elif c2 < 0:
            gp_constraints5 += [c1[i] * eta_v[i] <= r_v[i]]

    # # (sum p^2 / s) * (sum r^2 / s) <= t2
    tmp3 = 0
    tmp4 = 0
    for i in range(n):
        tmp3 += (p[i] ** 2) * (s_v[i] ** -1)
        tmp4 += (r_v[i] ** 2) * (s_v[i] ** -1)
    gp_constraints6 = [tmp3 * tmp4 <= t2]

    # # t2 ** 0.5 + sum p * r / s <= 2 * p_m * d
    p_m = p_min_supp_w(p)
    gp_constraints7 = []
    for i in range(m):
        tmp5 = 0
        for j in range(n):
            tmp5 += p[i] * r_v[i] / s_v[i]
        gp_constraints7 += [t2**0.5 + tmp5 <= 2 * p_m[i] * d[i]]

    # # configure gp constraints
    gp_constraints_t = gp_constraints1 + gp_constraints2 + \
        gp_constraints3 + gp_constraints4 + \
        gp_constraints5 + gp_constraints6 + gp_constraints7

    # create objective funcition and solve GP (trigger)
    gp_f = 1
    for i in range(n):
        gp_f *= sigma_v[i] * eta_v[i]
    gp_prob = cp.Problem(cp.Maximize(gp_f), gp_constraints_t)
    gp_prob.solve(gp=True)
    print("GP status (trigger) :", gp_prob.status)

    # get value of sigma and eta
    sigma = np.array(sigma_v.value)
    eta = np.array(eta_v.value)
    return sigma, eta


def gain_parameter_solver_gp(p, r):
    # define varialbe
    K_v = cp.Variable(n, pos=True)
    L_v = cp.Variable((n, n), pos=True)
    s_v = cp.Variable(n, pos=True)
    t1 = cp.Variable(1, pos=True)
    t2 = cp.Variable(1, pos=True)
    t3 = cp.Variable(1, pos=True)
    s2_v = cp.Variable(n, pos=True)

    # create constraints
    # # K <= D2
    gp_constraints1 = [K_v[i] <= D2[i][i] for i in range(n)]

    # # L <= A2
    gp_constraints2 = []
    for i in range(n):
        for j in range(n):
            if i != j:
                gp_constraints2 += [L_v[i][j] <= A2[i][j]]

    # # s + pK + sum(pBL) <= pD2 + sum(PBA2)
    gp_constraints3 = []
    for i in range(n):
        tmp1 = 0
        tmp2 = 0
        for j in range(n):
            if i != j:
                tmp1 += p[j] * B[j][j] * L_v[j][i]
                tmp2 += p[j] * B[j][j] * A2[j][i]
        gp_constraints3 += [s_v[i] + p[i] *
                            K_v[i] + tmp1 <= p[i] * D2[i][i] + tmp2]

    # # sum(p^2/s) <= t1
    # # sum(r^2/s) <= t2
    tmp3 = 0
    tmp4 = 0
    for i in range(n):
        tmp3 += (s_v[i]**-1) * (p[i]**2)
        tmp4 += (s_v[i]**-1) * (r[i]**2)
    gp_constraints4 = [tmp3 <= t1]
    gp_constraints5 = [tmp4 <= t2]

    # # t1^(1/2) t2^(1/2) + sum( p /s * sum(tmp74)) <= 2p_min d + tmp73
    s_max = np.ones(n) * -1
    for i in range(n):
        tmp5 = p[i] * D2[i][i]
        for j in range(n):
            if i != j:
                tmp5 += p[j] * B[j][j] * A2[j][i]
        s_max[i] = tmp5
    gp_constraints6 = []
    p_m = p_min_supp_w(p)
    for k in range(m):
        tmp71 = 0
        tmp73 = 0
        for i in range(n):
            tmp74 = - p[i] * D[i][i]
            for j in range(n):
                tmp74 += p[j] * B[j][j] * A[j][i]
            if tmp74 >= 0:
                tmp71 += p[i] * tmp74 * s_v[i]**(-1)
            elif tmp74 < 0:
                tmp73 -= p[i] * tmp74 / s_max[i]
        gp_constraints6 += [(t1**0.5) @ (t2**0.5) +
                            tmp71 <= 2 * p_m[k] * d[k] + tmp73]

    # # configure gp constraints
    gp_constraints_g = gp_constraints1 + gp_constraints2 + gp_constraints3 + \
        gp_constraints4 + gp_constraints5 + gp_constraints6

    # create objective func and solve GP (gain)
    gp_f = 1
    for i in range(n):
        gp_f *= K_v[i]
        for j in range(n):
            if i != j:
                gp_f *= L_v[i][j]
    gp_prob = cp.Problem(cp.Maximize(gp_f), gp_constraints_g)
    gp_prob.solve(gp=True)
    print("GP status (gain):", gp_prob.status)

    # get value of K and L
    K = D2 - np.diag(K_v.value)
    L = A2 - np.array(L_v.value)
    for i in range(n):
        L[i][i] = 0
    return K, L


def data_info(p, K, L, sigma, eta):
    print("W\n", W)
    print("d\n", d_table)
    print("A\n", A)
    print("B\n", B)
    print("D\n", D)
    print("p\n", p)
    print("K\n", K)
    print("L\n", L)
    print("sigma\n", sigma)
    print("eta\n", eta)


def event_trigger_func(x, xk, sigma, eta):
    if math.fabs(x - xk) > sigma * x + eta:
        return 1
    else:
        return 0


def plot_data(K, L, sigma, eta, cost, choice):
    # define time and gap
    N = 150000
    h = 0.00001

    # define propotion of infected pepole
    x = np.zeros([N, n])
    # x0 = np.random.rand(n)
    x0 = np.array([0.05, 0.2, 0.7, 0.9, 0.8])
    x[0] = x0
    xk = x0

    # define event and objective list
    event = np.zeros([N, n])
    d_table_list = np.zeros([N, m])
    d_table_list[0] = d_table
    delta_trans = np.zeros([N - 1, n])
    a_trans = np.zeros([N - 1, n])

    # collect transition data of propotion of infected pepole and triggerring event
    for k in range(N - 1):
        for j in range(n):
            d_table_list[k + 1] = d_table
            # # choice 1 has no control input
            if choice == 1:
                x[k + 1] = x[k] + h * \
                    (-D.dot(x[k]) + (I_n - np.diag(x[k])
                                     ).dot(B).dot(A).dot(x[k]))
                delta_trans[k] = np.zeros(n)
                a_trans[k] = np.zeros(n)
            # # choice 2 has normal feedback control
            elif choice == 2:
                if event_trigger_func(x[k][j], xk[j], 0, 0) == 1:
                    xk[j] = x[k][j]
                    event[k + 1][j] = 1
                x[k + 1] = x[k] + h * (-(D + K.dot(np.diag(xk))).dot(x[k]) + (
                    I_n - np.diag(x[k])).dot(B).dot(A - L.dot(np.diag(xk))).dot(x[k]))
                delta_trans[k] = K.dot(xk)
                a_trans[k] = L.dot(xk)
            # # choice 3 has event triggered control
            elif choice == 3:
                if event_trigger_func(x[k][j], xk[j], sigma[j], eta[j]) == 1:
                    xk[j] = x[k][j]
                    event[k + 1][j] = 1
                x[k + 1] = x[k] + h * (-(D + K.dot(np.diag(xk))).dot(x[k]) + (
                    I_n - np.diag(x[k])).dot(B).dot(A - L.dot(np.diag(xk))).dot(x[k]))
                delta_trans[k] = K.dot(xk)
                a_trans[k] = L.dot(xk)

    # plot data
    # # subplot 1 is transition data of propotion of infected pepole
    plt.figure(figsize=(16, 9.7))
    # plt.subplot(2, 2, 1)
    plt.plot()
    for i in range(x.shape[1]):
        plt.plot(x.T[i], label=r'$x_{%d}(t)$' % (i + 1))
    plt.plot(d_table_list.T[0], linestyle="dashdot")
    plt.plot(d_table_list.T[1], linestyle="dashdot")
    plt.plot(d_table_list.T[2], linestyle="dashdot")
    plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0),
               borderaxespad=0, fontsize=35)
    plt.xlabel('Time', fontsize=35)
    # plt.ylabel('Proportion of infected people')
    plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=35)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=35)
    plt.title(r'Transition of $x_i(t)$', fontsize=40)
    plt.grid()

    # # subplot 2 is transition data of triggerring event
    # plt.subplot(2, 2, 2)
    plt.figure(figsize=(18, 9.7))
    for i in range(event.shape[1]):
        plt.plot(event.T[i], label=r'event$_%d$' % (i + 1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=30)
    plt.xlabel('Time', fontsize=35)
    plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=35)
    plt.yticks([0, 1], fontsize=35)
    plt.title('Triggering event', fontsize=40)
    plt.grid()

    # plt.subplot(2, 2, 3)
    plt.figure(figsize=(18, 9.7))
    for i in range(n):
        plt.plot(delta_trans.T[i], label=r'$\delta%d(t)$' % (i + 1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=30)
    plt.xlabel('Time', fontsize=35)
    plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=35)
    plt.yticks(fontsize=35)
    plt.title(r'Transition of $\delta_i(t)$', fontsize=40)
    plt.grid()
    # plt.subplot(2, 2, 4)
    plt.figure(figsize=(18, 9.7))
    for i in range(n):
        plt.plot(
            a_trans.T[i], label=r'$\sum_{j\in \mathcal{N}_i} a_{%dj}(t)$' % (i + 1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=30)
    plt.xlabel('Time', fontsize=35)
    plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=35)
    plt.yticks(fontsize=35)
    plt.title(
        r'Transition of $\sum_{j\in \mathcal{N}_i} a_{ij}(t)$', fontsize=40)

    # plt.suptitle('GP-GP')
    # # plot collected data
    plt.tight_layout()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    p, r = lyapunob_parameter_solver()
    cost = []
    K, L = gain_parameter_solver_gp(p, r)
    sigma, eta = trigger_parameter_solver_gp(p, K, L)
    data_info(p, K, L, sigma, eta)
    plot_data(K, L, sigma, eta, cost, choice=3)
