import os
import sys

import numpy as np

import crocoddyl

WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

# The linear system This system consists of two cars of mass M1 and M1, respectively.
# They are connected by a spring and damper and rolling in a frictionless surface. The
# spring k is not tension when the gap between these cars (i.e., k0) is 1m. Our aims is
# to drive our first car to x1ref with zero velocity in both cars. Additionally, we would
# like to keep a distance between cars above dxref.

# Define linear dynamics and quadratic cost
dt = 1e-3
k, k0 = 100.0, 1
d = 10.0
m1, m2 = 15, 15
wx, wu = 1.0, 1e-5
dxref, x1ref = 0.1, 0.25
Fx = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [-k / m1, k / m1, -d / m1, d / m1],
        [-k / m2, k / m2, -d / m2, d / m2],
    ]
)
Fu = np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / m1, 0.0], [0.0, 1.0 / m2]])
fo = np.array([0.0, 0.0, k0 / m1, k0 / m2])
A, B, f = np.eye(4) + dt * Fx, dt * Fu, dt * fo
Q = wx * np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1e-5, 0.0],
        [0.0, 0.0, 0.0, 1e-5],
    ]
)
R = wu * np.eye(2)
N = np.zeros((4, 2))
q = np.array([dxref, -dxref, 0.0, 0.0])
r = np.zeros(2)
# Define terminal constraint
G, g = np.zeros((0, 6)), np.zeros((0, 6))
H, h = np.eye(4, 6), -np.array([x1ref, x1ref + dxref, 0.0, 0.0])
model = crocoddyl.ActionModelLQR(A, B, Q, R, N, f, q, r)
terminalModel = crocoddyl.ActionModelLQR(A, B, Q, R, N, G, H, f, q, r, g, h)

# Creating the shooting problem and the optimal-control solver
T = 50
x0 = np.array([0.0, 0.5, 0.0, 0.0])
problem_1 = crocoddyl.ShootingProblem(x0, [model] * T, terminalModel)
problem_2 = crocoddyl.ShootingProblem(x0, [model] * T, terminalModel)
solver = crocoddyl.SolverFDDP(problem_1, crocoddyl.DynamicsSolverType.MultiShoot)
if crocoddyl.WITH_ODYN:
    solverSQP = crocoddyl.SolverOdynSQP(problem_2)
solver.setCallbacks([crocoddyl.CallbackVerbose()])
if crocoddyl.WITH_ODYN:
    solverSQP.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the shooting problem
solver.solve()
if crocoddyl.WITH_ODYN:
    solverSQP.solve()

# Printing the terminal state
print("Target state:", -h)
print("Terminal state:", solver.xs[-1])

# Plots the controller performance
if WITHPLOT:
    import matplotlib.pyplot as plt

    ts = np.arange(0, (T + 1) * dt, dt)
    g = [x[1] - x[0] for x in solver.xs]
    x1 = [x[0] for x in solver.xs]
    x2 = [x[1] for x in solver.xs]
    dx1 = [x[2] for x in solver.xs]
    dx2 = [x[3] for x in solver.xs]
    # Gap regulation
    plt.figure(0, figsize=(10.7, 4.3))
    plt.subplot(1, 2, 1)
    plt.title("gap tracking: x1 - x2")
    plt.plot(ts, g)
    plt.xlabel("time (sec)")
    plt.legend()
    # Body positions
    plt.subplot(1, 2, 2)
    plt.title("body positions")
    plt.plot(ts, x1, label="x1")
    plt.plot(ts, x2, label="x2")
    plt.plot(ts[-1], x1ref, "bo")
    plt.plot(ts[-1], x1ref + dxref, "bo")
    plt.xlabel("time (sec)")
    plt.legend()
    plt.show()
