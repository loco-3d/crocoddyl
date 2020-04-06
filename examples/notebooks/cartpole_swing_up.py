# Display the solution
import numpy as np
from IPython.display import HTML

from cartpole_utils import animateCartpole
import crocoddyl


class DifferentialActionModelCartpole(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(4), 1, 6)  # nu = 1; nr = 6
        self.unone = np.zeros(self.nu)

        self.m1 = 1.
        self.m2 = .1
        self.l = .5
        self.g = 9.81
        self.costWeights = [1., 1., 0.1, 0.001, 0.001, 1.]  # sin, 1-cos, x, xdot, thdot, f

    def calc(self, data, x, u=None):
        if u is None: u = model.unone
        # Getting the state and control variables
        y, th, ydot, thdot = np.asscalar(x[0]), np.asscalar(x[1]), np.asscalar(x[2]), np.asscalar(x[3])
        f = np.asscalar(u[0])

        # Shortname for system parameters
        m1, m2, l, g = self.m1, self.m2, self.l, self.g
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        m = m1 + m2
        mu = m1 + m2 * s**2
        xddot = (f + m2 * c * s * g - m2 * l * s * thdot**2) / mu
        thddot = (c * f / l + m * g * s / l - m2 * c * s * thdot**2) / mu
        data.xout = np.matrix([xddot, thddot]).T

        # Computing the cost residual and value
        data.r = np.matrix(self.costWeights * np.array([s, 1 - c, y, ydot, thdot, f])).T
        data.cost = .5 * np.asscalar(sum(np.asarray(data.r)**2))

    def calcDiff(self, data, x, u=None):
        # Advance user might implement the derivatives
        pass


# Creating the DAM for the cartpole
cartpoleDAM = DifferentialActionModelCartpole()
cartpoleData = cartpoleDAM.createData()
cartpoleDAM = model = DifferentialActionModelCartpole()

# Using NumDiff for computing the derivatives. We specify the
# withGaussApprox=True to have approximation of the Hessian based on the
# Jacobian of the cost residuals.
cartpoleND = crocoddyl.DifferentialActionModelNumDiff(cartpoleDAM, True)

# Getting the IAM using the simpletic Euler rule
timeStep = 5e-2
cartpoleIAM = crocoddyl.IntegratedActionModelEuler(cartpoleND, timeStep)

# Creating the shooting problem
x0 = np.array([0., 3.14, 0., 0.])
T = 50

terminalCartpole = DifferentialActionModelCartpole()
terminalCartpoleDAM = crocoddyl.DifferentialActionModelNumDiff(terminalCartpole, True)
terminalCartpoleIAM = crocoddyl.IntegratedActionModelEuler(terminalCartpoleDAM)

terminalCartpole.costWeights[0] = 100
terminalCartpole.costWeights[1] = 100
terminalCartpole.costWeights[2] = 1.
terminalCartpole.costWeights[3] = 0.1
terminalCartpole.costWeights[4] = 0.01
terminalCartpole.costWeights[5] = 0.0001
problem = crocoddyl.ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)

# Solving it using DDP
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])
ddp.solve([], [], 300)

HTML(animateCartpole(ddp.xs).to_html5_video())
