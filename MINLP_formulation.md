# Go1 Box-Pushing MINLP

## Role

`MinlpContactScheduler` is a receding-horizon planar contact planner for one
Go1 and one rectangular box. It outputs only the first future body reference
to the existing whole-body MPPI controller:

$$
(q_1^{\rm ref},\phi_r^{\rm ref},u_0)\quad\longrightarrow\quad\text{MPPI}.
$$

The planned force predicts box motion; it is not a MuJoCo actuator command.
MPPI remains responsible for whole-body execution.

## Variables and modes

The box state and Go1 planar reference are

$$
x_k^o=[p_{x,k},p_{y,k},\psi_k,v_{x,k},v_{y,k},\omega_k]^\top,
\qquad q_k=[q_{x,k},q_{y,k}]^\top.
$$

The initial box state and Go1 position are measured from MuJoCo. The goal is
the box planar position $p_g$; box terminal yaw is unconstrained.

Candidate faces are $\mathcal S=\{\mathrm{rear},\mathrm{left},
\mathrm{right}\}$. Each contact block selects one face or free mode:

$$
z_{\rm free,b}+\sum_{s\in\mathcal S}z_{s,b}=1,
\qquad z_{\rm free,b},z_{s,b}\in\{0,1\}.
$$

For face midpoint $r_s^0$, outward normal $n_s$, tangent $t_s$, and half
length $\ell_s$, the continuous contact variables satisfy

$$
r_{s,b}=r_s^0+\xi_{s,b}t_s,\quad
|\xi_{s,b}|\le\ell_s z_{s,b},\quad
0\le f_{s,b}\le f_{\max}z_{s,b}.
$$

Free mode permits repositioning without a contact-pose constraint.

## Dynamics and contact geometry

Let $R(\psi)$ be the planar rotation matrix. At fine dynamics step $k$, for
block $b(k)$,

$$
F_k=\sum_s-f_{s,b(k)}R(\psi_k)n_s,\qquad
\tau_k=\sum_s[R(\psi_k)r_{s,b(k)}]\times[-f_{s,b(k)}R(\psi_k)n_s].
$$

The box uses damped rigid-body Euler integration and Go1 reference motion uses
a speed-limited integrator:

$$
\begin{aligned}
p_{k+1}&=p_k+\Delta t_d v_k,&
v_{k+1}&=v_k+\frac{\Delta t_d}{m}(F_k-c_vv_k),\\
\psi_{k+1}&=\psi_k+\Delta t_d\omega_k,&
\omega_{k+1}&=\omega_k+\frac{\Delta t_d}{I}(\tau_k-c_\omega\omega_k),\\
q_{k+1}&=q_k+\Delta t_d u_k,&\|u_k\|_2&\le v_r^{\max}.
\end{aligned}
$$

When face $s$ is active, Go1's reference point must be near its standoff pose:

$$
q_k\approx p_k+R(\psi_k)(r_{s,b(k)}+d_rn_s).
$$

This is enforced componentwise with a $\pm\epsilon$ big-$M$ implication. The
yaw-dependent force and off-centre torque make the problem nonlinear; the
face binaries make it an MINLP.

## Objective and continuity

The objective penalizes running/terminal box-goal error, box linear and yaw
motion, Go1 reference speed, and squared force. Switching auxiliaries obey

$$
\eta_{s,b}\ge |z_{s,b}-z_{s,b-1}|.
$$

For $b=0$, $z_{s,-1}$ is the one-hot face issued by the preceding replan (or
all zero for free mode). This cross-replan penalty prevents a first-stage face
change from being free. Its weight is `initial_contact_switch`.

## Timing, handoff, and safeguards

Discrete contact variables, force, and location are constant for one contact
period $\Delta t_c$. Box and robot-reference dynamics may integrate at a finer
$\Delta t_d$, where $\Delta t_c=m\Delta t_d$. The replan period must equal
$\Delta t_c$; the default is 0.2 s with five blocks (one-second lookahead).

For an active face, the raw Go1 heading is the inward face normal:

$$
\phi_{\rm target}=\operatorname{atan2}(-[R(\psi)n_s]_y,
                                         -[R(\psi)n_s]_x).
$$

Free mode targets its planned velocity direction, retaining the prior yaw at
near-zero speed. Before MPPI receives either heading, it is slew-limited:

$$
\phi_{\rm cmd}=\phi_{\rm last}+
\operatorname{clip}\!\left(\operatorname{wrap}(\phi_{\rm target}-\phi_{\rm last}),
[-\dot\phi_{\max}\Delta t_c,\dot\phi_{\max}\Delta t_c]\right).
$$

The default $\dot\phi_{\max}=1.25$ rad/s limits a 5 Hz command change to
0.25 rad. Raw and commanded yaw are logged.

If the selected standoff pose is outside the short-horizon reachable distance,
a free-mode straight-line approach is issued. The solver warm-starts from a
shifted previous solution. A time-limited BONMIN result is accepted only when
integer-feasible and constraint-feasible; otherwise a nominal fallback is
used. BONMIN runs quietly in a worker with an external deadline.

## Replacement path

An L2O model can predict the mode sequence and continuous warm start. Fixing
the predicted binaries converts this to an NLP that unrolled SQP can refine;
the `ContactSchedule` and MPPI reference interface need not change.
