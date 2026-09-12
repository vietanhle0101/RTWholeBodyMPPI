# High-Level MINLP Contact Scheduler for Go1 Box Pushing

## Purpose and architecture

The high-level planner chooses when and where Go1 contacts one rectangular box.
It predicts planar box motion, then sends only the **first future** Go1
reference to the existing MPPI controller:

$$
\text{MINLP contact scheduler}\rightarrow(q_1^{\rm ref},\phi_{r,0}^{\rm ref},u_0)
\rightarrow\text{whole-body MPPI}\rightarrow\text{MuJoCo}.
$$

MPPI remains responsible for joint-level execution. The planned contact force
is a reduced-order prediction variable; it is **not** directly commanded to
MuJoCo or MPPI.

## State, modes, and geometry

The box state is

$$
x_k^o=[p_{x,k},p_{y,k},\psi_k,v_{x,k},v_{y,k},\omega_k]^\top,
$$

and $q_k=[q_{x,k},q_{y,k}]^\top$ is the desired Go1 planar body position.
Each solve fixes $x_0^o=\hat x_0^o$ and $q_0=\hat q_0$ from MuJoCo.
The task goal is planar position $p_g\in\mathbb R^2$; terminal box yaw is
unconstrained in this first version.

Candidate faces are $\mathcal S=\{\mathrm{rear},\mathrm{left},\mathrm{right}\}$.
For face $s$, $r_s^0$, $n_s$, $t_s$, and $\ell_s$ are respectively its
box-frame midpoint, outward normal, tangent, and half-length. Binary variables
$z_{s,k}$ select pushing faces and $z_{\rm free,k}$ selects repositioning:

$$
z_{\rm free,k}+\sum_{s\in\mathcal S}z_{s,k}=1,
\qquad z_{s,k},z_{\rm free,k}\in\{0,1\}.
$$

The continuous contact location and normal-force magnitude obey

$$
r_{s,k}=r_s^0+\xi_{s,k}t_s,\quad
-\ell_s z_{s,k}\le\xi_{s,k}\le\ell_s z_{s,k},\quad
0\le f_{s,k}\le f_{\max}z_{s,k}.
$$

With $R(\psi)$ the planar rotation matrix, predicted force and torque are

$$
F_{s,k}=-f_{s,k}R(\psi_k)n_s,\qquad
\tau_{s,k}=[R(\psi_k)r_{s,k}]\times F_{s,k}.
$$

Thus yaw creates nonlinear world-frame forces, while off-center contact
($\xi_{s,k}\ne0$) generates yaw torque.

## Dynamics and contact pose

The reduced-order dynamics are

$$
\begin{aligned}
p_{k+1}&=p_k+\Delta t\,v_k, &
v_{k+1}&=v_k+\frac{\Delta t}{m}\left(\sum_sF_{s,k}-c_vv_k\right),\\
\psi_{k+1}&=\psi_k+\Delta t\,\omega_k, &
\omega_{k+1}&=\omega_k+\frac{\Delta t}{I}\left(\sum_s\tau_{s,k}-c_\omega\omega_k\right),\\
q_{k+1}&=q_k+\Delta t\,u_k, & \|u_k\|_2&\le v_r^{\max}.
\end{aligned}
$$

For an active face, the desired Go1 pushing pose is outside the box:

$$
q_{s,k}^{\rm push}=p_k+R(\psi_k)(r_{s,k}+d_rn_s),
$$

enforced componentwise with a big-$M$ tolerance:

$$
-\epsilon-M(1-z_{s,k})\le q_k-q_{s,k}^{\rm push}
\le\epsilon+M(1-z_{s,k}).
$$

The Go1 heading is computed after solving, avoiding `atan2` in the MINLP:

$$
\phi_{r,k}^{\rm ref}=\operatorname{atan2}(-[R(\psi_k)n_s]_y,
                                             -[R(\psi_k)n_s]_x).
$$

## Objective

With switching auxiliaries $\eta_{s,k}\ge|z_{s,k}-z_{s,k-1}|$, solve

$$
\begin{aligned}
\min\;J={}&\sum_{k=0}^{H-1}\big[
(p_k-p_g)^\top Q_p(p_k-p_g)+q_v\|v_k\|^2+q_\omega\omega_k^2\\
&\qquad+u_k^\top R_u u_k+r_f\sum_sf_{s,k}^2
+\rho_{\rm sw}\sum_s\eta_{s,k}\big]\\
&+(p_H-p_g)^\top Q_f(p_H-p_g).
\end{aligned}
$$

subject to the dynamics, mode, force, contact-location, and pushing-pose
constraints above. This is an MINLP because it includes binary modes and terms
such as $f\cos\psi$, $f\sin\psi$, and $\xi f$.

## Optional multi-rate form

The implementation supports fine dynamics timestep $\Delta t_d$ and coarser
contact timestep $\Delta t_c=m\Delta t_d$. Box/robot states integrate at
$\Delta t_d$; $z$, $f$, $\xi$, and $\eta$ are constant inside each contact
block. Setting $m=1$ recovers the single-rate formulation. The current
warm-start implementation requires the replan period to equal $\Delta t_c$.

## Practical safeguards

- If the heuristic-selected pushing pose is beyond the horizon's reachable
  distance, command a straight-line free-mode approach before solving.
- Shift the previous solution forward one contact block as a warm start.
- Accept a time-limited BONMIN incumbent only if it is integer-feasible and
  satisfies all constraints; otherwise use a nominal safe fallback.
- Run BONMIN in a quiet worker process with an external hard deadline.

## Future replacement

An L2O model may predict $\hat z(\hat x_0^o,\hat q_0,p_g)$ and a continuous
warm start. With $z$ fixed, the MINLP becomes a continuous NLP that can be
refined by unrolled SQP. The MPPI-facing interface remains unchanged.
