import copy
import unittest

import numpy as np
from crocoddyl import (ActionModelLQR, ActionModelUnicycle, ActionModelUnicycleVar, ShootingProblem, SolverDDP,
                       SolverFDDP, SolverKKT)
from numpy.linalg import eig, inv, norm


class ShootingProblemTest(unittest.TestCase):
    MODEL = ActionModelUnicycle()

    def setUp(self):
        # Creating the model data
        self.model = self.MODEL
        self.data = self.model.createData()

        # Creating shooting problem
        self.problem = ShootingProblem(self.model.State.zero(), [self.model, self.model], self.model)

    def test_trajectory_dimension(self):
        # Getting the trajectory dimension of the shooting problem
        xs = [m.State.zero() for m in self.problem.runningModels + [self.problem.terminalModel]]
        xs_dim = len(np.concatenate(xs[:-1]))
        x_dim = sum([m.nx for m in self.problem.runningModels])

        # Checking the trajectory dimension
        self.assertTrue(xs_dim == x_dim, "Trajectory dimension is wrong.")

    def test_control_dimension(self):
        # Getting the control dimension of the shooting problem
        us = [np.zeros(m.nu) for m in self.problem.runningModels]
        us_dim = len(np.concatenate(us))
        u_dim = sum([m.nu for m in self.problem.runningModels])

        # Checking the control dimension
        self.assertTrue(us_dim == u_dim, "Control dimension is wrong.")


class SolverKKTTest(unittest.TestCase):
    MODEL = ActionModelUnicycle()

    def setUp(self):
        # Creating the model
        self.model = self.MODEL
        self.data = self.model.createData()

        # Defining the shooting problem
        self.problem = ShootingProblem(self.model.State.zero(), [self.model, self.model], self.model)

        # Creating the KKT solver
        self.kkt = SolverKKT(self.problem)

        # Setting up a warm-point
        xs = [m.State.zero() for m in self.problem.runningModels + [self.problem.terminalModel]]
        us = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.kkt.setCandidate(xs, us)

    def test_dimension_of_kkt_problem(self):
        # Compute the KKT matrix
        self.kkt.calc()

        # Getting the dimension of the KKT problem
        nxu = len(filter(lambda x: x > 0, eig(self.kkt.kkt)[0]))
        nx = len(filter(lambda x: x < 0, eig(self.kkt.kkt)[0]))

        # Checking the dimension of the KKT problem
        self.assertTrue(nxu == self.kkt.nx + self.kkt.nu, "Dimension of decision variables is wrong.")
        self.assertTrue(nx == self.kkt.nx, "Dimension of state variables is wrong.")

    def test_hessian_is_symmetric(self):
        # Computing the KKT matrix
        self.kkt.calc()

        # Checking the symmetricity of the Hessian
        self.assertTrue(np.linalg.norm(self.kkt.hess - self.kkt.hess.T) < 1e-9, "The Hessian isn't symmetric.")

    def test_search_direction(self):
        dxs, dus, ls = self.kkt.computeDirection()

        # Checking that the first primal variable ensures initial constraint
        self.assertTrue(
            np.linalg.norm(dxs[0] - self.problem.initialState) < 1e-9, "Initial constraint isn't guaranteed.")

        # Checking that primal variables ensures dynamic constraint (or its
        # linear approximation)
        LQR = isinstance(self.model, ActionModelLQR)
        h = 1 if LQR else 1e-6
        for i, _ in enumerate(dus):
            # Computing the next state
            xnext = self.model.calc(self.data, dxs[i] * h, dus[i] * h)[0] / h

            # Checking that the next primal variable is consistant with the
            # dynamics
            self.assertTrue(np.allclose(xnext, dxs, atol=10 * h),
                            "Primal variables doesn't ensure dynamic constraints.")


# --- TEST KKT ---
# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
NX = 1
NU = 1
model = ActionModelUnicycle()
data = model.createData()
LQR = isinstance(model, ActionModelLQR)

x = model.State.rand()
u = np.zeros([model.nu])

problem = ShootingProblem(model.State.zero(), [model, model], model)
kkt = SolverKKT(problem)
xs = [m.State.zero() for m in problem.runningModels + [problem.terminalModel]]
us = [np.zeros(m.nu) for m in problem.runningModels]

# Test dimensions of KKT calc.
kkt.setCandidate(xs, us)

# Test that the solution respect the dynamics (or linear approx of it).
dxs, dus, ls = kkt.computeDirection()
x0, x1, x2 = dxs
u0, u1 = dus
l0, l1, l2 = ls

# If LQR. test that a random solution respect the dynamics
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(m.nu) for m in problem.runningModels]
kkt.setCandidate(xs, us)
dxs, dus, ls = kkt.computeDirection()
assert (np.linalg.norm(xs[0] + dxs[0] - problem.initialState) < 1e-9)
for i, _ in enumerate(dus):
    if LQR:
        assert (np.linalg.norm(model.calc(data, xs[i] + dxs[i], us[i] + dus[i])[0] - (xs[i + 1] + dxs[i + 1])) < 1e-9)


# Test the optimality of the QP solution (underlying KKT).
def quad(a, Q, b):
    return .5 * np.dot(np.dot(Q, b).T, a)


cost = np.dot(.5 * np.dot(kkt.hess, kkt.primal) + kkt.grad, kkt.primal)
for i in range(1000):
    eps = 1e-1 * np.random.rand(*kkt.primal.shape)
    eps -= np.dot(np.dot(np.linalg.pinv(kkt.jac), kkt.jac), eps)  # project eps in null(jac)
    assert (cost <= np.dot(.5 * np.dot(kkt.hess, kkt.primal + eps) + kkt.grad, kkt.primal + eps))

# Check improvement model
_, _, done = kkt.solve(maxiter=0)
assert (not done)
x0s, u0s = kkt.xs, kkt.us
kkt.setCandidate(x0s, u0s)
dxs, dus, ls = kkt.computeDirection()
dv = kkt.tryStep(1)
x1s = [_x + dx for _x, dx in zip(x0s, dxs)]
u1s = [_u + du for _u, du in zip(u0s, dus)]
for xt, x1 in zip(kkt.xs_try, x1s):
    assert (norm(xt - x1) < 1e-9)
for ut, u1 in zip(kkt.us_try, u1s):
    assert (norm(ut - u1) < 1e-9)
d1, d2 = kkt.expectedImprovement()
if LQR:
    assert (d1 + d2 / 2 + problem.calc(x1s, u1s) < 1e-9)
d3 = np.dot(.5 * np.dot(kkt.hess, np.concatenate(dxs + dus)) + kkt.grad, np.concatenate(dxs + dus))
assert (d1 + d2 / 2 + d3 < 1e-9)

# Check stoping criteria
kkt.setCandidate(x1s, u1s)
kkt.calc()
dL, dF = kkt.stoppingCriteria()
if LQR:
    assert (dL + dF < 1e-9)
assert (abs(kkt.stoppingCriteria()[0] - sum((kkt.grad + np.dot(kkt.jacT, kkt.dual))**2)) < 1e-9)

xopt, uopt, done = kkt.solve(maxiter=200)
assert (done)
for i, _ in enumerate(uopt):
    assert (np.linalg.norm(model.calc(data, xopt[i], uopt[i])[0] - xopt[i + 1]) < 1e-9)

#  INTEGRATIVE TEST ###
#  INTEGRATIVE TEST ###
#  INTEGRATIVE TEST ###
T = 10
WITH_PLOT = not True


def disp(x):
    sc, delta = .1, .05
    a, b, th = x[:3]
    c, s = np.cos(th), np.sin(th)
    refs = []
    refs.append(plt.arrow(a - sc / 2 * c - delta * s, b - sc / 2 * s + delta * c, c * sc, s * sc, head_width=.03))
    refs.append(plt.arrow(a - sc / 2 * c + delta * s, b - sc / 2 * s - delta * c, c * sc, s * sc, head_width=.03))
    return refs


runcost = ActionModelUnicycle()
runcost.costWeights[1] = 10
termcost = ActionModelUnicycle()
termcost.costWeights[0] = 1000
problem = ShootingProblem(np.array([1, 0, 3]), [
    runcost,
] * T, termcost)
kkt = SolverKKT(problem)
xs, us, done = kkt.solve()
assert (norm(xs[-1]) < 1e-2)

if WITH_PLOT:
    import matplotlib.pyplot as plt
    for x in xs:
        disp(x)
    ax = max(np.concatenate([(abs(x[0]), abs(x[1])) for x in xs])) * 1.2
    plt.axis([-ax, ax, -ax, ax])
    plt.show()

model = ActionModelLQR(1, 1, driftFree=False)
data = model.createData()
# model = ActionModelUnicycle()
nx, nu = model.nx, model.nu

problem = ShootingProblem(model.State.zero() + 1, [model], model)
ddp = SolverDDP(problem)

xs = [m.State.zero() for m in problem.runningModels + [problem.terminalModel]]
us = [np.zeros(m.nu) for m in problem.runningModels]
# xs[0][:] = problem.initialState
xs[0] = np.random.rand(nx)
us[0] = np.random.rand(nu)
xs[1] = np.random.rand(nx)

ddp.setCandidate(xs, us)
ddp.computeDirection()
xnew, unew, cnew = ddp.forwardPass(stepLength=1)

# Check versus simple (1-step) DDP
ddp.problem.calcDiff(xs, us)
l0x = problem.runningDatas[0].Lx
l0u = problem.runningDatas[0].Lu
l0xx = problem.runningDatas[0].Lxx
l0xu = problem.runningDatas[0].Lxu
l0uu = problem.runningDatas[0].Luu
f0x = problem.runningDatas[0].Fx
f0u = problem.runningDatas[0].Fu
x1pred = problem.runningDatas[0].xnext

v1x = problem.terminalData.Lx
v1xx = problem.terminalData.Lxx

relin1 = np.dot(v1xx, x1pred - xs[1])
q0x = l0x + np.dot(f0x.T, v1x) + np.dot(f0x.T, relin1)
q0u = l0u + np.dot(f0u.T, v1x) + np.dot(f0u.T, relin1)
q0xx = l0xx + np.dot(f0x.T, np.dot(v1xx, f0x))
q0xu = l0xu + np.dot(f0x.T, np.dot(v1xx, f0u))
q0uu = l0uu + np.dot(f0u.T, np.dot(v1xx, f0u))

K0 = np.dot(inv(q0uu), q0xu.T)
k0 = np.dot(inv(q0uu), q0u)
assert (norm(K0 - ddp.K[0]) < 1e-9)
assert (norm(k0 - ddp.k[0]) < 1e-9)

v0x = q0x - np.dot(q0xu, k0)
v0xx = q0xx - np.dot(q0xu, K0)
assert (norm(v0xx - ddp.Vxx[0]) < 1e-9)

x0 = problem.initialState
u0 = us[0] - k0 - np.dot(K0, x0 - xs[0])
x1 = model.calc(data, x0, u0)[0]

assert (norm(unew[0] - u0) < 1e-9)
assert (norm(xnew[1] - x1) < 1e-9)

x0ref = problem.initialState
f0 = x1pred
l1x = v1x
l1xx = v1xx

# for n in [ 'x0ref','l0x','l0u','l0xx','l0xu','l0uu','f0x','f0u','f0','l1x','l1xx' ]:
#    print(n[0]+n[2:]+'['+n[1]+'] : '+str(float(locals()[n])))

# --- TEST DDP vs KKT LQR ---
# ---------------------------------------------------
# ---------------------------------------------------
# ---------------------------------------------------
np.random.seed(220)
model = ActionModelLQR(3, 2, driftFree=False)
# model = ActionModelUnicycle()
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

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

kkt.setCandidate(xs, us)
dxkkt, dukkt, lkkt = kkt.computeDirection()
xkkt, ukkt, donekkt = kkt.solve(maxiter=2)
assert (donekkt)

ddp.setCandidate(xs, us)
ddp.computeDirection()
xddp, uddp, costddp = ddp.forwardPass(stepLength=1)
assert (norm(xddp[0] - xkkt[0]) < 1e-9)
assert (norm(xddp[1] - xkkt[1]) < 1e-9)
assert (norm(uddp[0] - ukkt[0]) < 1e-9)

# Test step length
us = [np.random.rand(m.nu) for m in problem.runningModels]
xs = problem.rollout(us)
kkt.setCandidate(xs, us, isFeasible=True)
ddp.setCandidate(xs, us, isFeasible=False)
dxkkt, dukkt, lkkt = kkt.computeDirection()
step = .1
dvkkt = kkt.tryStep(step)
xkkt, ukkt = kkt.xs_try, kkt.us_try
dxddp, duddp, lddp = ddp.computeDirection()
dvddp = ddp.tryStep(step)
xddp, uddp = ddp.xs_try, ddp.us_try

assert (norm(xddp[0] - xkkt[0]) < 1e-9)
assert (norm(xddp[1] - xkkt[1]) < 1e-9)
assert (norm(uddp[0] - ukkt[0]) < 1e-9)

d1, d2 = kkt.expectedImprovement()
assert (abs(dvkkt - d1 * step - .5 * d2 * step**2) < 1e-9)

dd1, dd2 = ddp.expectedImprovement()
assert (abs(d1 - dd1) < 1e-9 and abs(d2 - dd2) < 1e-9)

# Test stopping criteria at optimum
ddp.tryStep(1)
ddp.setCandidate(ddp.xs_try, ddp.us_try)
kkt.setCandidate(ddp.xs_try, ddp.us_try)
ddp.computeDirection()
kkt.computeDirection()
assert (sum(ddp.stoppingCriteria()) < 1e-9)
assert (sum(kkt.stoppingCriteria()) < 1e-9)

# --- TEST DDP VS KKT NLP ---
model = ActionModelUnicycle()
nx = model.nx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero() + 2, [model] * T, model)
xs = [m.State.zero() for m in problem.runningModels + [problem.terminalModel]]
us = [np.zeros(m.nu) for m in problem.runningModels]
x0ref = problem.initialState
xs[0] = np.random.rand(nx)
us[0] = np.random.rand(nu)
xs[1] = np.random.rand(nx)
# xs[1] = model.calc(data,xs[0],us[0])[0].copy()

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

kkt.setCandidate(xs, us)
dxkkt, dukkt, lkkt = kkt.computeDirection()
xkkt = [_x + dx for _x, dx in zip(xs, dxkkt)]
ukkt = [_u + du for _u, du in zip(us, dukkt)]

ddp.setCandidate(xs, us)
ddp.computeDirection()
xddp, uddp, costddp = ddp.forwardPass(stepLength=1)
assert (norm(xddp[0] - xkkt[0]) < 1e-9)
assert (norm(uddp[0] - ukkt[0]) < 1e-9)
# Value predicted by the linearization of the transition model:
#      xlin = xpred + Fx dx + Fu du = f(xguess,ugess) + Fx (xddp-xguess) + Fu (uddp-ugess).
xddplin1 = model.calc(model.createData(), xs[0], us[0])[0]
xddplin1 += np.dot(problem.runningDatas[0].Fx, xddp[0] - xs[0]) + np.dot(problem.runningDatas[0].Fu, uddp[0] - us[0])
assert (norm(xddplin1 - xkkt[1]) < 1e-9)

# --- TEST DDP VS KKT NLP in T time---
'''
This test computes the KKT solution on one step, and compare it with the DDP solution
obtained from the linearized dynamics about the current guess xs,us.
'''
model = ActionModelUnicycle()
nx = model.nx
nu = model.nu
T = 20

problem = ShootingProblem(model.State.zero() + 2, [model] * T, model)
xs = [np.random.rand(nx) for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

kkt.setCandidate(xs, us)
dxkkt, dukkt, lkkt = kkt.computeDirection()
xkkt = [_x + dx for _x, dx in zip(xs, dxkkt)]
ukkt = [_u + du for _u, du in zip(us, dukkt)]

ddp.setCandidate(xs, us)
ddp.computeDirection()
xddp, uddp, costddp = ddp.forwardPass(stepLength=1)

assert (norm(problem.initialState[0] - xkkt[0]) < 1e-9)
assert (norm(problem.initialState[0] - xddp[0]) < 1e-9)
assert (norm(uddp[0] - ukkt[0]) < 1e-9)

# Forward pass with linear dynamics.
ulin = [np.nan] * T
xlin = xddp[:1] + [np.nan] * T
for t in range(T):
    # Value predicted by the linearization of the transition model:
    #      xlin = xpred + Fx dx + Fu du = f(xguess,ugess) + Fx (xddp-xguess) + Fu (uddp-ugess).
    ulin[t] = us[t] - ddp.k[t] - np.dot(ddp.K[t], model.State.diff(xs[t], xlin[t]))
    xlin[t + 1] = model.calc(model.createData(), xs[t], us[t])[0] + np.dot(
        problem.runningDatas[t].Fx, xlin[t] - xs[t]) + np.dot(problem.runningDatas[t].Fu, ulin[t] - us[t])
    assert (norm(ulin[t] - ukkt[t]) < 1e-9)
    assert (norm(xlin[t + 1] - xkkt[t + 1]) < 1e-9)

# --- DDP NLP solver ---

np.random.seed(220)
model = ActionModelLQR(1, 1, driftFree=False)
nx = model.nx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero() + 2, [model] * T, model)
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]
x0ref = problem.initialState

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

xkkt, ukkt, donekkt = kkt.solve(maxiter=2, init_xs=xs, init_us=us)
xddp, uddp, doneddp = ddp.solve(maxiter=2, init_xs=xs, init_us=us, regInit=0)
assert (donekkt)
assert (doneddp)
assert (norm(xkkt[0] - problem.initialState) < 1e-9)
assert (norm(xddp[0] - problem.initialState) < 1e-9)
for t in range(problem.T):
    assert (norm(ukkt[t] - uddp[t]) < 1e-9)
    assert (norm(xkkt[t + 1] - xddp[t + 1]) < 1e-9)

# --- DDP VERSUS KKT : integrative test ---
np.random.seed(220)
model = ActionModelUnicycle()
nx = model.nx
nu = model.nu
T = 20

problem = ShootingProblem(model.State.zero() + 2, [model] * T, model)
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]
x0ref = problem.initialState

ddp = SolverDDP(problem)
kkt = SolverKKT(problem)

ddp.th_stop = 1e-18
kkt.th_stop = 1e-18

xkkt, ukkt, donekkt = kkt.solve(maxiter=200, init_xs=xs, init_us=us)
xddp, uddp, doneddp = ddp.solve(maxiter=200, init_xs=xs, init_us=us, regInit=0)
assert (donekkt)
assert (doneddp)
assert (norm(xkkt[0] - problem.initialState) < 1e-9)
assert (norm(xddp[0] - problem.initialState) < 1e-9)
for t in range(problem.T):
    assert (norm(ukkt[t] - uddp[t]) < 1e-6)
    assert (norm(xkkt[t + 1] - xddp[t + 1]) < 1e-6)

# --- Test with manifold dynamics
model = ActionModelUnicycleVar()

nx = model.nx
ndx = model.ndx
nu = model.nu
T = 10

x0ref = np.array([-1, -1, 1, 0])
problem = ShootingProblem(x0ref, [model] * T, model)
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]

kkt = SolverKKT(problem)
kkt.setCandidate(xs, us)
kkt.computeDirection()


def xTOx3(x):
    return model.State.diff(model.State.zero(), x)


def x3TOx(x3):
    return model.State.integrate(model.State.zero(), x3)


xs = problem.rollout(us)

model3 = ActionModelUnicycle()
X = model3.State
data3 = model3.createData()

x0ref3 = xTOx3(x0ref)
problem3 = ShootingProblem(x0ref3, [model3] * T, model3)
xs3 = [xTOx3(_x) for _x in xs]
us3 = [_u.copy() for _u in us]

kkt3 = SolverKKT(problem3)
kkt3.setCandidate(xs3, us3)
kkt3.computeDirection()

kkt.setCandidate(xs, us)
kkt.computeDirection()

assert (norm(kkt3.primal[:3] - kkt.primal[:3]) < 1e-9)

kkt.th_stop = 1e-18
kkt3.th_stop = 1e-18
x3, u3, d3 = kkt3.solve(maxiter=100, init_xs=xs3, init_us=us3, verbose=False)
x4, u4, d4 = kkt.solve(maxiter=100, init_xs=xs, init_us=us, verbose=False)
assert (d3 and d4)

assert (norm(X.diff(x4[0], x0ref)) < 1e-9)
assert (norm(x3[0] - x0ref3) < 1e-9)
for t in range(T):
    assert (norm(u4[t] - u3[t]) < 1e-9)
    assert (norm(x4[t + 1] - x3TOx(x3[t + 1])) < 1e-9)
assert (norm(kkt.primal - kkt3.primal) < 1e-9)
# Duals are not equals as the jacobians are not the same.

# DDP test with manifold
T = 1

x0ref = np.array([-1, -1, 1, 0])
problem = ShootingProblem(x0ref, [model] * T, model)
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]

kkt = SolverKKT(problem)
ddp = SolverDDP(problem)
ddp.solve()

xkkt, ukkt, donekkt = kkt.solve(maxiter=200, init_xs=xs, init_us=us)
xddp, uddp, doneddp = ddp.solve(maxiter=200, init_xs=xs, init_us=us, regInit=0)
assert (donekkt)
assert (doneddp)
assert (norm(xkkt[0] - problem.initialState) < 1e-9)
assert (norm(xddp[0] - problem.initialState) < 1e-9)
for t in range(problem.T):
    assert (norm(ukkt[t] - uddp[t]) < 1e-6)
    assert (norm(xkkt[t + 1] - xddp[t + 1]) < 1e-6)

T = 10

x0ref = np.array([-1, -1, 1, 0])
problem = ShootingProblem(x0ref, [model] * T, model)
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]

kkt = SolverKKT(problem)
ddp = SolverDDP(problem)
ddp.solve()

kkt.th_stop = 1e-18
ddp.th_stop = 1e-18

xkkt, ukkt, donekkt = kkt.solve(maxiter=200, init_xs=xs, init_us=us)
xddp, uddp, doneddp = ddp.solve(maxiter=200, init_xs=xs, init_us=us, regInit=0)
assert (donekkt)
assert (doneddp)
assert (norm(xkkt[0] - problem.initialState) < 1e-9)
assert (norm(xddp[0] - problem.initialState) < 1e-9)
for t in range(problem.T):
    assert (norm(ukkt[t] - uddp[t]) < 1e-6)
    assert (norm(xkkt[t + 1] - xddp[t + 1]) < 1e-6)

del problem
# -------------------------------------------------------------------
# --- REG -----------------------------------------------------------
# -------------------------------------------------------------------
model = ActionModelLQR(1, 1, driftFree=False)
ndx = model.ndx
nu = model.nu
T = 1

problem = ShootingProblem(model.State.zero() + 1, [model] * T, model)
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]
x0ref = problem.initialState

kkt = SolverKKT(problem)
xkkt, ukkt, dkkt = kkt.solve(maxiter=2)
assert (dkkt)

kkt.setCandidate(xs, us)
dxs, dus, ls = kkt.computeDirection()

xs = [m.State.zero() for m in problem.runningModels + [problem.terminalModel]]
us = [np.zeros(nu) for m in problem.runningModels]
kkt.setCandidate(xs, us)
kkt.x_reg = .1
kkt.u_reg = 1
dxs_reg, dus_reg, ls_reg = kkt.computeDirection()

modeldmp = copy.copy(model)
modeldmp.Lxx[range(ndx), range(ndx)] += kkt.x_reg
modeldmp.Luu += kkt.u_reg
problemdmp = ShootingProblem(model.State.zero() + 1, [modeldmp] * T, modeldmp)
kktdmp = SolverKKT(problem)

kktdmp.setCandidate(kkt.xs, kkt.us)
dxs_dmp, dus_dmp, ls_dmp = kktdmp.computeDirection()

for t in range(T):
    assert (norm(dus_dmp[t] - dus_reg[t]) < 1e-9)
    assert (norm(dxs_dmp[t + 1] - dxs_reg[t + 1]) < 1e-9)

# --- REG UNICYCLE ---
model = ActionModelUnicycleVar()
data = model.createData()

nx = model.nx
ndx = model.ndx
nu = model.nu
T = 10

x0ref = np.array([-1, -1, 1, 0])
problem = ShootingProblem(x0ref, [model] * T, model)
xs = [m.State.zero() for m in problem.runningModels + [problem.terminalModel]]
us = [np.zeros(nu) for m in problem.runningModels]

kkt = SolverKKT(problem)
kkt.setCandidate(xs, us)
kkt.x_reg = 1
kkt.u_reg = 1
model.costWeights[0] = 0
model.costWeights[1] = 0
dxs, dus, ls = kkt.computeDirection()

modeldmp = copy.copy(model)
problemdmp = ShootingProblem(x0ref, [modeldmp] * T, modeldmp)
kktdmp = SolverKKT(problem)
kktdmp.setCandidate(xs, us)
modeldmp.costWeights[0] = kkt.x_reg
modeldmp.costWeights[1] = kkt.u_reg
dxs_dmp, dus_dmp, ls_dmp = kktdmp.computeDirection()

for t in range(T):
    assert (norm(dus_dmp[t] - dus[t]) < 1e-9)
    assert (norm(dxs_dmp[t + 1] - dxs[t + 1]) < 1e-9)

# --- DDP ---
model = ActionModelLQR(3, 3, driftFree=False)
nx = model.nx
nu = model.nu
T = 5

problem = ShootingProblem(model.State.zero() + 1, [model] * T, model)
xs = [m.State.rand() for m in problem.runningModels + [problem.terminalModel]]
us = [np.random.rand(nu) for m in problem.runningModels]
x0ref = problem.initialState

kkt = SolverKKT(problem)
ddp = SolverDDP(problem)

kkt.setCandidate(xs, us)
ddp.setCandidate(xs, us)


def deltaddp(solv):
    solv.computeDirection()
    xs_d, us_d, cost_d = solv.forwardPass(stepLength=1)
    dxs_d = [m.State.diff(x0, x) for m, x0, x in zip(solv.models(), xs, xs_d)]
    dus_d = [u - u0 for u0, u in zip(us, us_d)]
    return dxs_d, dus_d


dxs_k, dus_k, ls_k = kkt.computeDirection()
dxs_d, dus_d = deltaddp(ddp)
for t in range(T):
    assert (norm(dus_d[t] - dus_k[t]) < 1e-9)
    assert (norm(dxs_d[t + 1] - dxs_k[t + 1]) < 1e-9)

kkt.u_reg = 1
ddp.u_reg = 1

dxs_k, dus_k, ls_k = kkt.computeDirection()
dxs_d, dus_d = deltaddp(ddp)
for t in range(T):
    assert (norm(dus_d[t] - dus_k[t]) < 1e-9)
    assert (norm(dxs_d[t + 1] - dxs_k[t + 1]) < 1e-9)

kkt.x_reg = 1
ddp.x_reg = 1

dxs_k, dus_k, ls_k = kkt.computeDirection()
dxs_d, dus_d = deltaddp(ddp)
for t in range(T):
    assert (norm(dus_d[t] - dus_k[t]) < 1e-9)
    assert (norm(dxs_d[t + 1] - dxs_k[t + 1]) < 1e-9)

# --- REG INTEGRATIVE TEST ---

# -------------------------------------------------------------------
# --- test invalid direction computed in backward pass --------------
# -------------------------------------------------------------------
model = ActionModelLQR(3, 3, driftFree=False)
nx = model.nx
nu = model.nu
T = 5

runningModels = [model] * T
problem = ShootingProblem(model.State.zero() + 1, runningModels, model)

# Make artificially Quu=0 at T-1
Vxx_T = problem.terminalModel.Lxx
Fu_T_1 = problem.runningModels[T - 1].Fu
problem.runningModels[T - 2].Luu = -1. * np.dot(np.dot(Fu_T_1.T, Vxx_T), Fu_T_1)

ddp = SolverDDP(problem)
fddp = SolverFDDP(problem)

# Do not allowed to regularize the problem and solve it, so we can trigger an invalid direction computations
ddp.regMax = 1e-5
fddp.regMax = 1e-5

assert (not ddp.solve()[2])
assert (not fddp.solve()[2])


# -------------------------------------------------------------------
# ------------- test expected improvement without gaps --------------
# -------------------------------------------------------------------
NX, NU = 3, 3
T = 5
models = [ActionModelLQR(NX, NU, driftFree=True) for t in range(T + 1)]
problem = ShootingProblem(models[0].State.zero(), models[:-1], models[-1])

# Run the KKT and FDDP solver
kkt = SolverKKT(problem)
[xskkt, uskkt, donekkt] = kkt.solve(maxiter=1)
fddp = SolverFDDP(problem)
fddp.computeDirection()
fddp.tryStep(1.)

# Checks that FDDP solution is OK
for t in range(T):
    assert (np.allclose(fddp.us_try[t], kkt.us[t]))
    assert (np.allclose(fddp.xs_try[t + 1], kkt.xs[t + 1]))

# Checks the expecte improvement against the KKT solver
d1, d2 = fddp.expectedImprovement()
d1kkt, d2kkt = kkt.expectedImprovement()
assert (abs(d1 - d1kkt) < 1e-14 and abs(d2 - d2kkt) < 1e-14)


# -------------------------------------------------------------------
# ------------- test expected improvement against gaps --------------
# -------------------------------------------------------------------
NX, NU = 3, 3
T = 5
models = [ActionModelLQR(NX, NU, driftFree=False) for t in range(T + 1)]
problem = ShootingProblem(models[0].State.zero(), models[:-1], models[-1])

# Run the KKT and FDDP solver
kkt = SolverKKT(problem)
[xskkt, uskkt, donekkt] = kkt.solve(maxiter=1)
fddp = SolverFDDP(problem)
fddp.computeDirection()
fddp.tryStep(1.)

# Checks that FDDP solution is OK
for t in range(T):
    assert (np.allclose(fddp.us_try[t], kkt.us[t]))
    assert (np.allclose(fddp.xs_try[t + 1], kkt.xs[t + 1]))

# Checks the expecte improvement against the KKT solver
d1, d2 = fddp.expectedImprovement()
d1kkt, d2kkt = kkt.expectedImprovement()
assert (abs(d1 - d1kkt) < 1e-14 and abs(d2 - d2kkt) < 1e-14)

if __name__ == '__main__':
    unittest.main()
