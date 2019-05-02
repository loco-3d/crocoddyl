import numpy as np
from crocoddyl import (DifferentialActionModelLQR, IntegratedActionModelEuler, ShootingProblem, SolverBoxDDP,
                       SolverBoxKKT, SolverDDP, SolverKKT)
from crocoddyl.qpsolvers import quadprogWrapper

# --- TEST DDP vs KKT LQR ---
# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
np.set_printoptions(linewidth=400, suppress=True)
np.random.seed(220)

nq = 4
nu = 2
nv = nq

dmodel = DifferentialActionModelLQR(nq, nu)
model = IntegratedActionModelEuler(dmodel, withCostResiduals=False)
nx = model.nx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero() + 2, [model] * T, model)
xs = [m.State.zero() for m in problem.runningModels + [problem.terminalModel]]
us = [np.zeros(m.nu) for m in problem.runningModels]
x0ref = problem.initialState

xs[0] = np.random.rand(nx)
xs[1] = np.random.rand(nx)
us[0] = np.random.rand(nu)
# xs[1] = model.calc(data,xs[0],us[0])[0].copy()

ddpbox = SolverBoxDDP(problem)
ddpbox.qpsolver = quadprogWrapper
ddp = SolverDDP(problem)

ddp.setCandidate(xs, us)
ddp.computeDirection()
xddp, uddp, costddp = ddp.forwardPass(stepLength=1)

# The argmin value is barely within the limits of the control (one dimension is on limit)
ddpbox.ul = np.array([
    min(np.minimum(uddp[0], min(us[0]))),
] * nu)
ddpbox.uu = np.array([
    max(np.maximum(uddp[0], max(us[0]))),
] * nu)

# ddpbox.ul = np.array([-3.,]*nu)
# ddpbox.uu = np.array([3.,]*nu)

ddpbox.setCandidate(xs, us)
ddpbox.computeDirection()
xddp_box, uddp_box, costddp_box = ddpbox.forwardPass(stepLength=1)
# The solution of the boxddp should be same as normal ddp.
assert (np.allclose(xddp[0], xddp_box[0], atol=1e-9))
assert (np.allclose(xddp[1], xddp_box[1], atol=1e-9))
assert (np.allclose(uddp[0], uddp_box[0], atol=1e-9))

if min(uddp[0]) < min(us[0]):
    limit_id = np.where(uddp[0] == min(uddp[0]))[0]
    limit_lower = True
else:
    limit_id = np.where(uddp[0] == max(uddp[0]))[0]
    limit_type = False

# Make smaller the limits. This should force the u dimension which was on limit
# to be clamped at the new limits.
ddpbox.ul = np.array([
    min(np.minimum(uddp[0], min(us[0]))),
] * nu) + 1e-3
ddpbox.uu = np.array([
    max(np.maximum(uddp[0], max(us[0]))),
] * nu) - 1e-3

ddpbox.setCandidate(xs, us)
ddpbox.computeDirection()
xddp_box, uddp_box, costddp_box = ddpbox.forwardPass(stepLength=1)

# Check that the new uddp_box is clamped at the new limit
# Check that all other values of uddp_box are the same as the previous uddp
if limit_lower:
    assert (uddp_box[0][limit_id] == ddpbox.ul[limit_id])
    ok_range = range(nu)
    ok_range.pop(limit_id)
    assert (np.allclose(uddp[0][ok_range], uddp_box[0][ok_range], atol=1e-9))
else:
    assert (uddp_box[0][limit_id] == ddpbox.uu[limit_id])
    ok_range = range(nu)
    ok_range.pop(limit_id)
    assert (np.allclose(uddp[0][ok_range], uddp_box[0][ok_range], atol=1e-9))

# KKT vs Box KKT
kkt = SolverKKT(problem)
xkkt, ukkt, donekkt = kkt.solve(maxiter=2, init_xs=xs, init_us=us)

boxkkt = SolverBoxKKT(problem)
xkkt_box, ukkt_box, donekkt_box = boxkkt.solve(maxiter=2, init_xs=xs, init_us=us)

assert (np.allclose(xkkt[0], xkkt_box[0], atol=1e-9))
assert (np.allclose(xkkt[1], xkkt_box[1], atol=1e-9))
assert (np.allclose(ukkt[0], ukkt_box[0], atol=1e-9))

# BOX DDP VS BOX KKT
xkkt_box, ukkt_box, donekkt_box = boxkkt.solve(maxiter=2,
                                               init_xs=xs,
                                               init_us=us,
                                               qpsolver=quadprogWrapper,
                                               ul=ddpbox.ul,
                                               uu=ddpbox.uu)

assert (np.allclose(xkkt_box[0], xddp_box[0], atol=1e-9))
assert (np.allclose(xkkt_box[1], xddp_box[1], atol=1e-9))
assert (np.allclose(ukkt_box[0], uddp_box[0], atol=1e-3))
