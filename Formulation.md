## High-Level MINLP Contact Scheduler for Go1 Box Pushing

We formulate a high-level mixed-integer nonlinear program (MINLP) that explicitly schedules robot--box contact modes and generates desired pushing references for the existing whole-body MPPI controller.

The overall control architecture is

$$
\text{MINLP/L2O contact scheduler}
\rightarrow
\text{desired pushing pose}
\rightarrow
\text{whole-body MPPI}
\rightarrow
\text{MuJoCo}.
$$

The MINLP does not optimize the full Go1 joint dynamics. Instead, it uses a reduced-order planar model of the box and a kinematic model of the robot reference motion. The resulting pushing pose is tracked by the existing low-level MPPI controller.

---

### 1. Box State

Let the planar box state at stage $k$ be

$x_k^o =
\begin{bmatrix}
p_k^\top &
\psi_k &
v_k^\top &
\omega_k
\end{bmatrix}^\top,$

where

$
p_k =
\begin{bmatrix}
p_{x,k}\\
p_{y,k}
\end{bmatrix}
\in\mathbb{R}^2
$

is the box position,

$
\psi_k\in\mathbb{R}
$

is the box yaw angle,

$
v_k =
\begin{bmatrix}
v_{x,k}\\
v_{y,k}
\end{bmatrix}
\in\mathbb{R}^2
$

is the translational velocity, and

$
\omega_k\in\mathbb{R}
$

is the angular velocity.

The task goal is a desired planar box position

$
p_g\in\mathbb{R}^2.
$

At every receding-horizon solve, the initial conditions are fixed to the
measured box and robot states:

$$
x_0^o=\hat{x}_0^o,
\qquad
q_0=\hat q_0.
$$

---

### 2. Candidate Contact Modes

Let

$$
\mathcal S
=
\{\mathrm{rear},\mathrm{left},\mathrm{right}\}
$$

denote the set of candidate box faces that can be selected for pushing.

For each face $s\in\mathcal S$, define

- $r_s^0\in\mathbb{R}^2$: midpoint of the face in the box frame,
- $n_s\in\mathbb{R}^2$: outward unit normal of the face,
- $t_s\in\mathbb{R}^2$: unit tangent vector along the face,
- $\ell_s>0$: half-length of the face.

Introduce the binary contact-scheduling variable

$
z_{s,k}\in\{0,1\},
$

where

$
z_{s,k}=1
$

means that face $s$ is selected for pushing at stage $k$.

Introduce a free/repositioning mode $z_{\mathrm{free},k}\in\{0,1\}$. Exactly
one mode is active at each stage:

$
z_{\mathrm{free},k}+\sum_{s\in\mathcal S} z_{s,k}
= 1.
$

The free mode allows the robot to temporarily leave contact while repositioning.

---

### 3. Continuous Contact Location

To allow the box to rotate, the contact point should not be fixed at the center of each face.

Introduce a continuous contact-location variable

$
\xi_{s,k}\in\mathbb{R}.
$

The contact point in the box frame is

$
r_{s,k}
=
r_s^0
+
\xi_{s,k} t_s.
$

The contact location is activated only when face $s$ is selected:

$
-\ell_s z_{s,k}
\le
\xi_{s,k}
\le
\ell_s z_{s,k}.
$

Therefore,

$
z_{s,k}=0
\quad\Longrightarrow\quad
\xi_{s,k}=0.
$

---

### 4. Contact Force

Let

$f_{s,k}\ge 0$

denote the normal pushing-force magnitude associated with face $s$.

The force is activated through

$
0
\le
f_{s,k}
\le
f_{\max} z_{s,k}.
$

The rotation matrix from the box frame to the world frame is

$$
R(\psi_k)
=
\begin{bmatrix}
\cos\psi_k & -\sin\psi_k\\
\sin\psi_k & \cos\psi_k
\end{bmatrix}.
$$

Since $n_s$ is defined as the outward face normal, the force applied to the box is

$$
F_{s,k} = -f_{s,k} R(\psi_k)n_s.
$$

Thus,

$
z_{s,k}=0
\quad\Longrightarrow\quad
F_{s,k}=0.
$

---

### 5. Contact Torque

The contact vector in world coordinates is

$$
\bar r_{s,k}
=
R(\psi_k) r_{s,k}.
$$

The corresponding planar torque applied to the box is

$$
\tau_{s,k}
=
\bar r_{s,k}
\times
F_{s,k},
$$

where for two-dimensional vectors,

$$
\begin{bmatrix}
a_x\\a_y
\end{bmatrix}
\times
\begin{bmatrix}
b_x\\b_y
\end{bmatrix}
=
a_x b_y-a_y b_x.
$$

Equivalently,

$$
\tau_{s,k}
=
\bar r_{s,k}^{x} F_{s,k}^{y}
-
\bar r_{s,k}^{y} F_{s,k}^{x}.
$$

Because planar rotations preserve the cross product,

$$
\tau_{s,k}
=
r_{s,k}
\times
\left(-f_{s,k}n_s\right).
$$

For example, consider the rear face of a rectangular box with half-length $a$:

$$
r_{\mathrm{rear}}^0
=
\begin{bmatrix}
-a\\
0
\end{bmatrix},
\qquad
n_{\mathrm{rear}}
=
\begin{bmatrix}
-1\\
0
\end{bmatrix},
\qquad
t_{\mathrm{rear}}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}.
$$

Then

$$
r_{\mathrm{rear},k}
=
\begin{bmatrix}
-a\\
\xi_{\mathrm{rear},k}
\end{bmatrix},
$$

and

$$
\tau_{\mathrm{rear},k}
=
-\xi_{\mathrm{rear},k} f_{\mathrm{rear},k}.
$$

Hence,

$$
\xi_{\mathrm{rear},k}=0
$$

produces a straight push, whereas

$$
\xi_{\mathrm{rear},k}\neq 0
$$

produces both translation and rotation.

---

### 6. Reduced-Order Box Dynamics

We use a planar rigid-body model for the high-level optimizer.

The translational dynamics are

$$
p_{k+1}
=
p_k
+
\Delta t\,v_k,
$$

and

$$
v_{k+1}
=
v_k
+
\frac{\Delta t}{m}
\left(
\sum_{s\in\mathcal S}F_{s,k}
-
c_v v_k
\right),
$$

where

- $m$ is the box mass,
- $c_v>0$ is a translational damping coefficient.

The rotational dynamics are

$$
\psi_{k+1}
=
\psi_k
+
\Delta t\,\omega_k,
$$

and

$$
\omega_{k+1}
=
\omega_k
+
\frac{\Delta t}{I}
\left(
\sum_{s\in\mathcal S}\tau_{s,k}
-
c_\omega\omega_k
\right),
$$

where

- $I$ is the planar moment of inertia,
- $c_\omega>0$ is a rotational damping coefficient.

The damping terms provide a reduced-order approximation of the box--ground interaction. The full contact physics are still evaluated in MuJoCo.

---

### 7. Robot Reference Dynamics

Let

$$
q_k
=
\begin{bmatrix}
q_{x,k}\\
q_{y,k}
\end{bmatrix}
\in\mathbb{R}^2
$$

denote the desired planar body position for the Go1 robot.

We use the simple kinematic model

$$
q_{k+1}
=
q_k
+
\Delta t\,u_k,
$$

where

$$
u_k\in\mathbb{R}^2
$$

is the desired planar reference velocity.

Impose

$$
\|u_k\|_2
\le
v_r^{\max}.
$$

This model is not intended to represent the full Go1 dynamics. It only imposes high-level reachability and prevents the desired pushing reference from moving instantaneously between different sides of the box.

---

### 8. Desired Pushing Pose

For each candidate face, define the desired robot body position

$$
q_{s,k}^{\mathrm{push}}
=
p_k
+
R(\psi_k)
\left(
r_{s,k}
+
d_r n_s
\right),
$$

where $ d_r>0$

is an offset that places the Go1 body outside the box.

If face $s$ is selected, the robot reference should remain close to the corresponding pushing pose:

$$
z_{s,k}=1
\quad\Longrightarrow\quad
q_k
\approx
q_{s,k}^{\mathrm{push}}.
$$

A big-$M$ formulation is

$$
-\epsilon
-
M(1-z_{s,k})
\le
q_k-q_{s,k}^{\mathrm{push}}
\le
\epsilon
+
M(1-z_{s,k}),
$$

applied componentwise.

Therefore,

$$
z_{s,k}=1
\quad\Longrightarrow\quad
\left\|
q_k-q_{s,k}^{\mathrm{push}}
\right\|_\infty
\le
\epsilon.
$$

---

### 9. Desired Robot Heading

For an active face $s$, the desired pushing direction is

$$
d_{s,k}^{\mathrm{push}}
=
-R(\psi_k)n_s.
$$

Rather than introducing an `atan2` term inside the MINLP, the desired robot heading can be computed after solving the optimization problem.

Let

$
\alpha_s
$

denote the fixed orientation of the outward normal $n_s$ in the box frame. Then the desired robot heading is

$
\phi_{r,k}^{\mathrm{ref}}
=
\psi_k+\alpha_s+\pi.
$

The pair

$
\left(
q_k^{\mathrm{ref}},
\phi_{r,k}^{\mathrm{ref}}
\right)
$

is passed to the existing whole-body MPPI controller.

---

### 10. Contact-Switching Variables

To discourage rapid switching between pushing faces, introduce auxiliary variables

$
\eta_{s,k}\ge0.
$

Impose

$
\eta_{s,k}
\ge
z_{s,k}-z_{s,k-1},
$

and

$
\eta_{s,k}
\ge
z_{s,k-1}-z_{s,k}.
$

Therefore,

$
\eta_{s,k}
\ge
\left|
z_{s,k}-z_{s,k-1}
\right|.
$

A switching penalty can then be included in the objective.

---

### 11. Objective Function

The objective is designed to move the box toward the target while penalizing excessive robot-reference motion, contact force, angular motion, and contact-mode switching:

$
\begin{aligned}
J
=
&
\sum_{k=0}^{H-1}
\Bigg[
(p_k-p_g)^\top Q_p(p_k-p_g)
+
q_v\|v_k\|_2^2
+
q_\omega\omega_k^2
\\
&
\qquad\qquad
+
u_k^\top R_u u_k
+
r_f
\sum_{s\in\mathcal S}
f_{s,k}^2
+
\rho_{\mathrm{sw}}
\sum_{s\in\mathcal S}
\eta_{s,k}
\Bigg]
\\
&
+
(p_H-p_g)^\top
Q_f
(p_H-p_g).
\end{aligned}
$

The box orientation does not need to be assigned a desired terminal value in the first version. Instead, rotation is used as an intermediate mechanism for steering the box toward the desired planar position.

---

### 12. Complete MINLP

The resulting optimization problem is

$
\boxed{
\begin{aligned}
\min_{\substack{
x_{0:H}^o,\,
q_{0:H},\,
u_{0:H-1},\\
f_{s,0:H-1},\,
\xi_{s,0:H-1},\,
z_{s,0:H-1},\,
z_{\mathrm{free},0:H-1},\,
\eta_{s,1:H-1}
}}
\quad
&
J
\\[1mm]
\text{s.t.}\quad
&
p_{k+1}
=
p_k+\Delta t\,v_k,
\\
&
v_{k+1}
=
v_k+
\frac{\Delta t}{m}
\left(
\sum_sF_{s,k}
-c_vv_k
\right),
\\
&
\psi_{k+1}
=
\psi_k+\Delta t\,\omega_k,
\\
&
\omega_{k+1}
=
\omega_k+
\frac{\Delta t}{I}
\left(
\sum_s\tau_{s,k}
-c_\omega\omega_k
\right),
\\
&
F_{s,k}
=
-f_{s,k}R(\psi_k)n_s,
\\
&
r_{s,k}
=
r_s^0+\xi_{s,k}t_s,
\\
&
\tau_{s,k}
=
[R(\psi_k)r_{s,k}]
\times
F_{s,k},
\\
&
0
\le
f_{s,k}
\le
f_{\max}z_{s,k},
\\
&
-\ell_sz_{s,k}
\le
\xi_{s,k}
\le
\ell_sz_{s,k},
\\
&
z_{\mathrm{free},k}+\sum_s z_{s,k}
=1,
\\
&
z_{s,k}
\in
\{0,1\},
\\
&
z_{\mathrm{free},k}
\in
\{0,1\},
\\
&
q_{k+1}
=
q_k+\Delta t\,u_k,
\\
&
\|u_k\|_2
\le
v_r^{\max},
\\
&
-\epsilon-M(1-z_{s,k})
\le
q_k-q_{s,k}^{\mathrm{push}}
\\
&
\hspace{28mm}
\le
\epsilon+M(1-z_{s,k}),
\\
&
\eta_{s,k}
\ge
z_{s,k}-z_{s,k-1},
\\
&
\eta_{s,k}
\ge
z_{s,k-1}-z_{s,k},
\\
&
\eta_{s,k}\ge0.
\end{aligned}
}
$

with

$
q_{s,k}^{\mathrm{push}}
=
p_k+
R(\psi_k)
\left(
r_{s,k}+d_rn_s
\right).
$

---

### 13. Why This Is a MINLP

The formulation contains binary variables

$
z_{s,k}\in\{0,1\},
$

as well as nonlinear continuous terms including

$
f_{s,k}\cos\psi_k,
\qquad
f_{s,k}\sin\psi_k,
\qquad
\xi_{s,k}f_{s,k}.
$

Therefore, the problem is a mixed-integer nonlinear program.

---

### 14. Connection to the L2O Architecture

Let the problem parameters be collected in

$
\theta
=
\left(
x_0^o,\,
q_0,\,
p_g
\right).
$

The L2O model first predicts the contact schedule

$
\hat z_{\phi}(\theta).
$

It may also predict an initial continuous solution

$
x_{\omega}^0(\theta).
$

Once the discrete schedule is fixed,

$
z=\hat z_{\phi}(\theta),
$

the MINLP reduces to a continuous nonlinear program:

$
\min_x
J
\left(
x,\hat z_{\phi}(\theta);
\theta
\right)
$

subject to the nonlinear box dynamics, contact-force equations, contact-location equations, and pushing-pose constraints.

The continuous solution can then be refined using a fixed number of differentiable SQP layers:

$
\theta
\rightarrow
\hat z_{\phi}(\theta)
\rightarrow
x_{\omega}^0(\theta)
\rightarrow
x_{\psi}^{K}.
$

Thus, the overall learned solution map is

$
\boxed{
\theta
\mapsto
\left(
\hat z_{\phi}(\theta),
x_{\psi}^{K}
\left(
x_{\omega}^0(\theta),
\hat z_{\phi}(\theta);
\theta
\right)
\right).
}
$

---

### 15. Interface With Whole-Body MPPI

The high-level optimizer outputs the active pushing face and desired robot pushing pose:

$
\left(
z_{s,k},
q_k^{\mathrm{ref}},
\phi_{r,k}^{\mathrm{ref}}
\right).
$

These references are passed to the existing whole-body MPPI controller:

$
\boxed{
\left(
q_k^{\mathrm{ref}},
\phi_{r,k}^{\mathrm{ref}}
\right)
\rightarrow
\text{whole-body MPPI}
\rightarrow
u_k^{\mathrm{joint}}
\rightarrow
\text{MuJoCo}.
}
$

The high-level MINLP determines **where and when the robot should push**, while the low-level MPPI controller determines **how the quadruped executes that pushing motion**.
