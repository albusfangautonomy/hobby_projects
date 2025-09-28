import casadi as ca
q1, q2 = ca.SX.sym('q1'), ca.SX.sym('q2')
dq1, dq2 = ca.SX.sym('dq1'), ca.SX.sym('dq2')
q = ca.vertcat(q1,q2)
dq = ca.vertcat(dq1,dq2)

state = ca.vertcat(q,dq)

u = ca.SX.sym('u')
tau = ca.vertcat(0,u)
B = ca.vertcat(0,1)

# Parameters
m1   = ca.SX.sym('m1')
m2   = ca.SX.sym('m2')
l1   = ca.SX.sym('l1')         # length of link 1
l2   = ca.SX.sym('l2')         # length of link 2
lc1  = ca.SX.sym('lc1')        # COM distance on link 1
lc2  = ca.SX.sym('lc2')        # COM distance on link 2
I1   = ca.SX.sym('I1')         # planar inertia about COM for link 1
I2   = ca.SX.sym('I2')         # planar inertia about COM for link 2
g    = ca.SX.sym('g')

p = ca.vertcat(m1,m2,l1,l2,lc1,lc2,I1,I2,g)

# --------------------------
# Kinematics
# --------------------------
# World frame: x right, y up. Angles measured from +x axis (CCW).
# Link 1 COM
x1 = lc1*ca.sin(q1)
y1 = -lc1*ca.cos(q1)

# Link 2 COM (relative to base)
x2 = l1*ca.sin(q1) + lc2*ca.sin(q1+q2)
y2 = -(l1*ca.cos(q1) + lc2*ca.cos(q1+q2))

pos1 = ca.vertcat(x1,y1)
pos2 = ca.vertcat(x2,y2)

# Velocities via Jacobian * qdot
J1 = ca.jacobian(pos1, q)      # 2x2
J2 = ca.jacobian(pos2, q)      # 2x2
v1 = J1 @ dq
v2 = J2 @ dq

# --------------------------
# Energies & Lagrangian
# --------------------------

T = (1/2) * m1 * (v1[0]**2 + v1[1]**2) + (1/2) * m2 * (v2[0]**2 + v2[1]**2)
U = m1*g*pos1[1] + m2*g*pos2[1]
L = T - U


# --------------------------
# Euler–Lagrange to manipulator form: M(q) ddq + C(q,dq) dq + G(q) = tau
# d/dt(∂L/∂dq) - ∂L/∂q = tau
# We compute:
#   H(q)      = ∂(∂L/∂dq)/∂dq       (mass matrix)
#   Cqdot     = ∂(∂L/∂dq)/∂q * dq   (Coriolis/centrifugal-like term times dq)
#   GradL_q   = ∂L/∂q               (note: G = ∂V/∂q, so GradL_q = ∂T/∂q - G)
# Then: H*ddq + Cqdot - GradL_q = tau  =>  ddq = H^{-1}(tau - Cqdot + GradL_q)
dldq = ca.jacobian(L, q).T
dldqdot = ca.jacobian(L, dq).T
H = ca.jacobian(dldqdot, dq)
# Cqdot = ca.jacobian(dldqdot, q) @ dq
GradL_q = dldq
print(H)
# print(GradL_q)
# print(Cqdot)
G_vec = ca.jacobian(U,q).T
C11 = -2*m2*l1*lc2*ca.sin(q2)*dq2
C12 = -m2*l1*lc2*ca.sin(q2)*dq2
C21 = m2*l1*lc2*ca.sin(q2)*dq1
C22 = 0
# Construct the matrix
C = ca.vertcat(
    ca.horzcat(C11, C12),
    ca.horzcat(C21, C22)
)
Cqdot = C @ dq
T_g1 = -m1*g*lc1*ca.sin(q1)-m2*g*(l1*ca.sin(q1)+lc2*ca.sin(q1+q2))
T_g2 = -m2*g*lc2*ca.sin(q1+q2)
T_g = ca.vertcat(T_g1, T_g2)
ddq = ca.solve(H, (B*u - Cqdot + T_g))

# --------------------------
# State-space form xdot = f(x,u)
# --------------------------
# generalized_x = ca.vertcat(q, dq)
# xdot = ca.vertcat(dq, ddq)
f = ca.vertcat(dq,ddq)

A_sym = ca.jacobian(f, state)
B_sym = ca.jacobian(f, u)

# CasADi functions so we can evaluate with numbers later
A_fun = ca.Function("A_fun", [state, u, p], [A_sym])
B_fun = ca.Function("B_fun", [state, u, p], [B_sym])

# print("A linearized= \n", A_sym)
# print("B linearized= \n", B_sym)



def linearize(m1, m2, l1, l2, g):
    import numpy as np
    z_eq = [float(ca.pi), 0, 0, 0]
    u_eq = 0.0
    lc1_param = l1/2.0
    I1 = m1 * (lc1_param **2)
    lc2_param = l2/2.0
    I2 = m2 * (lc2_param **2)
    p_in = [m1,m2,l1,l2,l1/2.0,l2/2.0,I1, I2, g]
    A = A_fun(z_eq, u_eq, p_in)
    B = B_fun(z_eq, u_eq, p_in)

    return A.full(), B.full()

def angle_wrap_around_pi(err):
    """Wrap angle error to (-pi, pi]."""
    import numpy as np
    return (err + np.pi) % (2*np.pi) - np.pi