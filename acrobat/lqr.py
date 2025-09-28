import numpy as np
from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.signal import cont2discrete

def controllability(A,B):
    n = A.shape[0]
    r = B.shape[1]
    C = np.zeros((n, n*r))
    product = B
    for i in range(n-1):
        C[:, i*r:((i+1)*r)] = product
        product = A @ product
    if np.linalg.matrix_rank(C) == n:
        return C, True
    else:
        return C, False

def lqr(A,B,Q,R):
    P = solve_continuous_are(A,B,Q,R)
    K = np.linalg.inv(R) @ B.T @ P
    #u = -K @ x
    return K


def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return K, P

# If you start from continuous-time (A,B), discretize first:
# (C,D are dummy here because cont2discrete needs them)
def c2d(A, B, dt):
    C = np.zeros((A.shape[0], A.shape[1]))
    D = np.zeros((A.shape[0], B.shape[1]))
    Ad, Bd, _, _, _ = cont2discrete((A, B, C, D), dt, method="zoh")
    return Ad, Bd
