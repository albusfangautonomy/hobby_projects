import numpy as np
from scipy.linalg import solve_continuous_are

def controllability(A,B):
    n = A.shape[0]
    r = B.shape[1]
    C = np.array((n, n*r))
    product = B
    for i in range(n-1):
        C[n, i*r:((i+1)*r)] = product
        product = A @ product
    if np.linalg.matrix_rank(C) == n:
        return C, True
    else:
        return C, False

def lqr(A,B,Q,R):
    P = solve_continuous_are(A,B,Q,R)
    K = np.linalg.inv(R) * B.T * P
    #u = -K @ x
    return K


