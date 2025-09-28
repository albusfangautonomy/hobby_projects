# README — Cart-Pole (Inverted Pendulum on a Cart) Symbolic Dynamics & Linearization with CasADi

This README explains the provided Python script that derives the **cart-pole** dynamics symbolically using **CasADi**, assembles the **manipulator form**  $M(q)\ddot q + C(q,\dot q)\dot q + G(q) = Bu$, and **linearizes** the nonlinear system about the upright equilibrium to obtain $A$ and $B$ matrices for control design (e.g., LQR).

---

## Contents

- [Problem Setup](#problem-setup)
- [State, Parameters, Inputs](#state-parameters-inputs)
- [Kinematics](#kinematics)
- [Energies and Lagrangian](#energies-and-lagrangian)
- [Euler–Lagrange → Manipulator Form](#eulerlagrange--manipulator-form)
- [What the Code Computes (and Why It Works)](#what-the-code-computes-and-why-it-works)
- [Explicit Forms (Cart-Pole)](#explicit-forms-cartpole)
- [State-Space Form and Linearization](#state-space-form-and-linearization)
- [API (Functions Exposed)](#api-functions-exposed)
- [Quick Start](#quick-start)
- [Notes on Signs & Conventions](#notes-on-signs--conventions)
- [Angle Wrapping Helper](#angle-wrapping-helper)

---

## Problem Setup

We model a point-mass **pole** (mass $m_p$, length $l$) hinged at the top of a **cart** (mass $m_c$) that moves horizontally along the $x$ axis. Gravity $g$ acts in $-y$. The generalized coordinates are:

$$
q = \begin{bmatrix} x \\ \theta \end{bmatrix}, \qquad
\dot q = \begin{bmatrix} \dot x \\ \dot\theta \end{bmatrix}.
$$

**Angle convention:** $\theta$ is measured **CCW from the +x axis**. With this choice, the **upright** configuration corresponds to $\theta=\pi$.

The control input $u$ is a **horizontal force** applied to the cart.

---

## State, Parameters, Inputs

- **State vector** (code variable `state`):

  $$
  \begin{bmatrix} x & \theta & \dot x & \dot\theta \end{bmatrix}^\top
  $$

- **Parameters** (`p`):

  $$
  m_p,\; m_c,\; l,\; g
  $$

- **Input mapping**:

  $$
  B=\begin{bmatrix}1\\0\end{bmatrix}, \quad \tau = B\,u
  $$
  (the force acts only on the cart’s generalized coordinate $x$).

---

## Kinematics

World frame: $x$ to the right, $y$ up.

- Cart position: $$ (x_c,y_c)=(x,0)$$.
- Pole mass position (end of a massless rod of length $l$):

  $$
  x_p = x + l\sin\theta, \qquad
  y_p = -\,l\cos\theta.
  $$

Velocities are obtained via the Jacobians:

$$
v_c = \frac{\partial (x_c,y_c)}{\partial q}\,\dot q,\qquad
v_p = \frac{\partial (x_p,y_p)}{\partial q}\,\dot q.
$$

---

## Energies and Lagrangian

- **Kinetic energy**

  $$
  T = \tfrac12 m_c\|v_c\|^2 + \tfrac12 m_p\|v_p\|^2.
  $$

- **Potential energy** (zero at $y=0$):

  $$
  U = m_p g\,y_p = -\,m_p g l \cos\theta.
  $$

- **Lagrangian**:

  $$
  L = T - U.
  $$

---

## Euler–Lagrange → Manipulator Form

Euler–Lagrange with generalized forces $\tau = Bu$:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} \;=\; \tau.
$$

After expanding and regrouping, this yields the **manipulator form**:

$$
M(q)\,\ddot q \;+\; C(q,\dot q)\,\dot q \;+\; G(q) \;=\; B\,u,
$$

where
- $M(q)$ is the **inertia (mass) matrix**,
- $C(q,\dot q)\dot q$ is the **Coriolis/centrifugal** vector,
- $G(q)$ is the **gravity** vector.

---

## What the Code Computes (and Why It Works)

The code uses CasADi to compute the following symbolic objects:

```python
dLdqdot = ca.jacobian(L, qdot).T         # ∂L/∂q̇
dLdq    = ca.jacobian(L, q).T            # ∂L/∂q
M       = ca.jacobian(dLdqdot, qdot)     # M(q) = ∂/∂q̇ (∂L/∂q̇)
Cqdot   = ca.jacobian(dLdqdot, q) @ qdot # C(q, q̇) q̇  (Christoffel contraction)
GradL_q = ca.jacobian(L, q).T            # ∂L/∂q = ∂T/∂q - ∂U/∂q
ddq     = ca.solve(M, (B*u_input - Cqdot + GradL_q))
```
### Euler–Lagrange Derivation in Manipulator Form

We start from the **Euler–Lagrange equation**:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q} - \frac{\partial L}{\partial q} = \tau
$$

---

Expanding the time derivative term:

$$
\frac{d}{dt}\frac{\partial L}{\partial \dot q} =
\frac{\partial}{\partial q}\left(\frac{\partial L}{\partial \dot q}\right)\dot q +
\frac{\partial}{\partial \dot q}\left(\frac{\partial L}{\partial \dot q}\right)\ddot q
$$

---

Substituting this back gives:

$$
M(q)\ddot q + C(q, \dot q)\dot q - \frac{\partial L}{\partial q} = \tau
$$

---

Since:

$$
\frac{\partial L}{\partial q} = \frac{\partial T}{\partial q} - \frac{\partial U}{\partial q}
$$

and

$$
\frac{\partial U}{\partial q} = G(q)
$$

We can rewrite:

$$
-\frac{\partial L}{\partial q} = -\frac{\partial T}{\partial q} + G(q)
$$

---

Thus, the **manipulator equation** becomes:

$$
M(q)\ddot q + C(q, \dot q)\dot q + G(q) = \tau
$$

which is equivalent to:

$$
M(q)\ddot q + C(q, \dot q)\dot q - \frac{\partial L}{\partial q} = \tau
$$

---

The code solves for **forward dynamics** acceleration:

$$
\ddot q = M(q)^{-1}\left(Bu - C(q, \dot q)\dot q + \frac{\partial L}{\partial q}\right)
$$

---

The alternative version shown in the comments:

$$
\ddot q = M(q)^{-1}\left(Bu - C(q, \dot q)\dot q - G(q)\right)
$$

is mathematically identical — it simply expresses the gravity vector $G(q)$ explicitly rather than using $\frac{\partial L}{\partial q}$.
