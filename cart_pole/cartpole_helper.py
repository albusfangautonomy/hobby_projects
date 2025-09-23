import casadi as ca
import numpy as np

x = ca.SX.sym("x")
theta = ca.SX.sym("theta")
xdot = ca.SX.sym("xdot")
thetadot = ca.SX.sym("thetadot")
f_x = ca.SX.sym("f_x")
u_input = f_x
B = ca.vertcat(1,0)

mp = ca.SX.sym("mp")
mc = ca.SX.sym("mc")
l = ca.SX.sym("l")
g = ca.SX.sym("g")

p = ca.vertcat(mp, mc, l, g)

q = ca.vertcat(x,theta)
qdot = ca.vertcat(xdot, thetadot)
state = ca.vertcat(q, qdot)
state_eq = ca.vertcat(0, ca.pi, 0, 0)

# --------------------------
# Kinematics
# --------------------------
# World frame: x right, y up. Angles measured from +x axis (CCW).
# cart
xc = x
yc = 0

# pole
xp = x + l * (ca.sin(theta))
yp = l * (-ca.cos(theta))


posc = ca.vertcat(xc,yc)
posp = ca.vertcat(xp,yp)


dxcdq = ca.jacobian(posc, q)
vc = dxcdq @ qdot
dxpdq = ca.jacobian(posp, q)
vp = dxpdq @ qdot

Tc = 0.5 * mc * (vc[0]**2 + vc[1]**2)
Tp = 0.5 * mp * (vp[0]**2 + vp[1]**2)
T = Tc + Tp

U = mp * g * posp[1]   # = -mp*g*l*cos(theta)

L = T - U

dLdqdot = ca.jacobian(L, qdot).T
dLdq = ca.jacobian(L, q).T
M = ca.jacobian(dLdqdot, qdot)
Cqdot = ca.jacobian(dLdqdot, q) @ qdot
GradL_q = ca.jacobian(L, q).T
ddq = ca.solve(M, (B *u_input - Cqdot + GradL_q))

f = ca.vertcat(xdot, thetadot, ddq)
A_sym = ca.jacobian(f, state)
B_sym = ca.jacobian(f, u_input)
# print("A system is: \n", A_sym.str())
# print("Linearized B system is \n", B_sym.str())
# Substitute the equilibrium (x=0, theta=pi, xdot=0, thetadot=0, u=0)
print ("M = \n", M)
print("Cqdot = ", Cqdot)
print("tau_g = \n", GradL_q)
z_eq = ca.SX([0.0, np.pi, 0.0, 0.0])  # SX vector
A_eq = ca.substitute(A_sym, state, z_eq)
A_eq = ca.substitute(A_eq, f_x, ca.SX(0.0))

B_eq = ca.substitute(B_sym, state, z_eq)
B_eq = ca.substitute(B_eq, f_x, ca.SX(0.0))

print("A_eq is \n", A_eq)
print("B_eq is \n", B_eq)


# CasADi functions so we can evaluate with numbers later
A_fun = ca.Function("A_fun", [state, u_input, p], [A_sym])
B_fun = ca.Function("B_fun", [state, u_input, p], [B_sym])

def linearize(mp_val, mc_val, l_val, g_val):
    """Return (A,B) evaluated at upright equilibrium and u=0."""
    param = [mp_val, mc_val, l_val, g_val]
    z_eq = [0.0, float(ca.pi), 0.0, 0.0]
    u_eq = 0.0
    A = A_fun(z_eq, u_eq, param)
    B = B_fun(z_eq, u_eq, param)
    return A.full(), B.full()

def angle_wrap_around_pi(err):
    """Wrap angle error to (-pi, pi]."""
    import numpy as np
    return (err + np.pi) % (2*np.pi) - np.pi


    