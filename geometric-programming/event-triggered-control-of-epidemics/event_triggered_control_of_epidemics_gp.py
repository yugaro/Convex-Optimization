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
D2 = np.diag(np.random.rand(n)) * 20
for i in range(n):
    A[i][i] = 0
    for j in range(n):
        if A[i][j] == 0:
            A2[i][j] = 0
        while A[i][j] < A2[i][j] or A[i][j] - A2[i][j] >= 0.1:
            A2[i][j] = np.random.rand()
weight = 100


def lyapunob_parameter_solver():
    # define variable
    p_v = cp.Variable(n)
    # create constraints
    sq_constraints = [1 <= p_v]
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


def trigger_parameter_solver_socp_socp(p, K, L):
    # define variable
    r_v = cp.Variable(n)
    s_v = cp.Variable(1, pos=True)
    sigma_v = cp.Variable(n, pos=True)
    eta_v = cp.Variable(n, pos=True)

    # create constraints
    # # 0 < s
    co_constraints1 = [epsilon <= s_v]

    # # 0 <= sigma < 1
    co_constraints2 = [0 <= sigma_v, sigma_v <= (1 - epsilon) * I]

    # # 0 <= eta < 1
    co_constraints3 = [0 <= eta_v, eta_v <= (1 - epsilon) * I]

    # # (A^T B - D) p + diag(p^T K + p^T BL) eta == r
    co_constraints4 = [
        (A.T.dot(B) - D).dot(p) + np.diag(p.T.dot(K) + p.T.dot(B).dot(L)) @ eta_v == r_v]

    # # diag(p^T K + p^T BL)(1 - sigma) >= s_v * I
    co_constraints5 = [
        np.diag(p.T.dot(K) + p.T.dot(B).dot(L)) @ (1 - sigma_v) >= s_v * I]

    # # ||p||・||r|| <= - p^T r + 2p_min ds
    co_constraints6 = []
    p_m = p_min_supp_w(p)
    for i in range(m):
        co_constraints6 += [np.linalg.norm(p) * cp.norm2(r_v) <= - p.T
                            @ r_v + 2 * p_m[i] * d[i] * s_v]

    # # configure cone constraints
    co_constraints_t = co_constraints1 + co_constraints2 + \
        co_constraints3 + co_constraints4 + co_constraints5 + co_constraints6

    # create objective func and solve SOCP
    # co_f = np.sum(K) + np.sum(L) + weight * \
    #     cp.sum(1 - sigma_v) + weight * cp.sum(1 - eta_v)
    co_f = I.dot(K).dot(I) + I.dot(L).dot(I) + \
        weight * I @ (1 - sigma_v) + weight * I @ (1 - eta_v)
    co_prob_t = cp.Problem(cp.Minimize(co_f), co_constraints_t)
    co_prob_t.solve(solver=cp.MOSEK)
    print("SOCP status (trigger parameter):", co_prob_t.status)

    # get value of sigma and eta
    sigma = np.array(sigma_v.value)
    eta = np.array(eta_v.value)
    return sigma, eta, co_prob_t.value


def gain_parameter_solver_socp_socp(p, sigma, eta):
    # define variable
    r_v = cp.Variable(n)
    s_v = cp.Variable(1, pos=True)
    K_v = cp.Variable((n, n), pos=True)
    L_v = cp.Variable((n, n), pos=True)

    # create constraints
    # # 0 < s
    co_constraints1 = [epsilon <= s_v]

    # # 0 <= K, K <= D2
    # co_constraints2 = [0 <= K_v, K_v <= D2]
    co_constraints2 = [K_v[i][i] <= D2[i][i] for i in range(n)]
    co_constraints2_2 = [K_v[i][j] == 0 for i in range(
        n) for j in range(n) if i != j]

    # # 0 <= L , L <= A2
    co_constraints3 = [L_v[i][j] <= A2[i][j]
                       for i in range(n) for j in range(n) if i != j]
    co_constraints3_2 = [L_v[i][i] == 0 for i in range(n)]

    # # (A^T B - D) p + diag(p^T K + p^T BL)eta == r
    co_constraints4 = [
        (A.T.dot(B) - D).dot(p) + cp.diag(p.T @ K_v + p.T.dot(B) @ L_v) @ eta == r_v]

    # # diag(p^T K p^T BL)(1 - sigma) >= s
    co_constraints5 = [
        cp.diag(p.T @ K_v + p.T.dot(B) @ L_v) @ (1 - sigma) >= s_v * I]

    # # ||p||・||r|| <= - p^T r + 2 p_min ds
    co_constraints6 = []
    p_m = p_min_supp_w(p)
    for i in range(m):
        co_constraints6 += [np.linalg.norm(p) * cp.norm2(r_v) <= - p.T
                            @ r_v + 2 * p_m[i] * d[i] * s_v]

    # # configure constraints
    co_constraints_g = co_constraints1 + co_constraints2 + \
        co_constraints3 + co_constraints4 + co_constraints5 + \
        co_constraints6 + co_constraints2_2 + co_constraints3_2

    # # create objective function and solve
    # co_f = cp.sum(K_v) + cp.sum(L_v) + weight * \
    #     np.sum(1 - sigma) + weight * np.sum(1 - eta)
    co_f = I @ K_v @ I + I @ L_v @ I + \
        weight * I.dot(1 - sigma) + weight * I.dot(1 - eta)
    co_prob_g = cp.Problem(cp.Minimize(co_f), co_constraints_g)
    co_prob_g.solve(solver=cp.MOSEK)
    print("SOCP status (gain parameter):", co_prob_g.status)

    # get value of K and L
    K = np.array(K_v.value)
    L = np.array(L_v.value)
    return K, L, co_prob_g.value


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
    print("weight\n", weight)


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
            # # choice 2 has normal feedback control
            elif choice == 2:
                if event_trigger_func(x[k][j], xk[j], 0, 0) == 1:
                    xk[j] = x[k][j]
                    event[k + 1][j] = 1
                x[k + 1] = x[k] + h * (-(D + K.dot(np.diag(xk))).dot(x[k]) + (
                    I_n - np.diag(x[k])).dot(B).dot(A - L.dot(np.diag(xk))).dot(x[k]))
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
    plt.figure(figsize=(18, 9.7))
    # plt.subplot(2, 2, 1)
    plt.plot()
    for i in range(x.shape[1]):
        plt.plot(x.T[i], label=r'$x_{%d}(t)$' % (i + 1))
    plt.plot(d_table_list.T[0], linestyle="dashdot")
    plt.plot(d_table_list.T[1], linestyle="dashdot")
    plt.plot(d_table_list.T[2], linestyle="dashdot")
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=20)
    plt.xlabel('Time', fontsize=30)
    # plt.ylabel('Proportion of infected people')
    plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=30)
    plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=30)
    plt.title(r'Transition of $x_i(t)$', fontsize=35)
    plt.grid()
    # # subplot 2 is transition data of triggerring event
    # plt.subplot(2, 2, 2)
    plt.figure(figsize=(18, 9.7))
    for i in range(event.shape[1]):
        plt.plot(event.T[i], label=r'event$_%d$' % (i + 1))
    plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
               borderaxespad=0, fontsize=20)
    plt.xlabel('Time', fontsize=30)
    plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=30)
    plt.yticks([0, 1], fontsize=30)
    plt.title('Triggering event', fontsize=35)
    plt.grid()

    if choice == 1 or choice == 2 or choice == 3:
        # plt.subplot(2, 2, 3)
        plt.figure(figsize=(18, 9.7))
        for i in range(n):
            plt.plot(delta_trans.T[i], label=r'$\delta%d(t)$' % (i + 1))
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
                   borderaxespad=0, fontsize=20)
        plt.xlabel('Time', fontsize=30)
        plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=30)
        plt.yticks(fontsize=30)
        plt.title(r'Transition of $\delta_i(t)$', fontsize=35)
        plt.grid()

        # plt.subplot(2, 2, 4)
        plt.figure(figsize=(18, 9.7))
        for i in range(n):
            plt.plot(
                a_trans.T[i], label=r'$\sum_{j\in \mathcal{N}_i} a_{%dj}(t)$' % (i + 1))
        plt.legend(bbox_to_anchor=(1, 1), loc='upper right',
                   borderaxespad=0, fontsize=20)
        plt.xlabel('Time', fontsize=30)
        plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=30)
        plt.yticks(fontsize=30)
        plt.title(
            r'Transition of $\sum_{j\in \mathcal{N}_i} a_{ij}(t)$', fontsize=35)
        plt.grid()

    if choice == 3:
        plt.figure()
        plt.plot(cost)
        plt.xlabel('times', fontsize=30)
        plt.ylabel('cost', fontsize=30)
        plt.xticks([0, 30000, 60000, 90000, 120000, 150000], fontsize=30)
        plt.yticks(fontsize=30)
        plt.title('Transition of cost (weight:%d)' % weight, fontsize=35)
        plt.grid()
    # # plot collected data
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    p, r = lyapunob_parameter_solver()
    cost = []
    K = D2
    L = A2
    sigma = np.array(
        [0.41002, 0.41002, 0.21004, 0.51001, 0.21004])
    eta = np.ones(n) * 0

    which_solver = np.random.randint(0, 2, 5000)
    for i in range(which_solver.shape[0]):
        if which_solver[i] == 1:
            K, L, g_cost = gain_parameter_solver_socp_socp(p, sigma, eta)
            cost += [g_cost]
        elif which_solver[i] == 0:
            sigma, eta, t_cost = trigger_parameter_solver_socp_socp(p, K, L)
            cost += [t_cost]
        print("epochs:", i)

    data_info(p, K, L, sigma, eta)
    plot_data(K, L, sigma, eta, cost, choice=3)
    
# weight 100: [0.41002242, 0.4100224, 0.21004395, 0.51001162, 0.21004402]
# times 0 to 5000 → start [0.41002, 0.41002, 0.21004, 0.51001, 0.21004]
