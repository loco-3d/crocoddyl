# Display the solution
import numpy as np
from IPython.display import HTML

from cartpole_utils import animateCartpole
from crocoddyl import *


class DifferentialActionModelCartpole:
    def __init__(self):
        self.State = StateVector(4)
        self.nq, self.nv = 2, 2
        self.nx = 4
        self.ndx = 4
        self.nout = 2
        self.nu = 1
        self.ncost = 6
        self.unone = np.zeros(self.nu)

        self.m1 = 1.
        self.m2 = .1
        self.l = .5
        self.g = 9.81
        self.costWeights = [1., 1., 0.1, 0.001, 0.001, 1.]  # sin,1-cos,x,xdot,thdot,f

    def createData(self):
        return DifferentialActionDataCartpole(self)

    def calc(model, data, x, u=None):
        if u is None: u = model.unone
        # Getting the state and control variables
        x, th, xdot, thdot = x
        f, = u

        # Shortname for system parameters
        m1, m2, l, g = model.m1, model.m2, model.l, model.g
        s, c = np.sin(th), np.cos(th)

        # Defining the equation of motions
        m = m1 + m2
        mu = m1 + m2 * s**2
        xddot = (f + m2 * c * s * g - m2 * l * s * thdot**2) / mu
        thddot = (c * f / l + m * g * s / l - m2 * c * s * thdot**2) / mu
        data.xout[:] = [xddot, thddot]

        # Computing the cost residual and value
        data.costResiduals[:] = [s, 1 - c, x, xdot, thdot, f]
        data.costResiduals[:] *= model.costWeights
        data.cost = .5 * sum(data.costResiduals**2)
        return data.xout, data.cost

    def calcDiff(model, data, x, u=None, recalc=True):
        # Advance user might implement the derivatives
        pass


class DifferentialActionDataCartpole:
    def __init__(self, model):
        self.cost = np.nan
        self.xout = np.zeros(model.nout)

        nx, nu, ndx, nout = model.nx, model.nu, model.ndx, model.nout
        self.costResiduals = np.zeros([model.ncost])
        self.g = np.zeros([ndx + nu])
        self.L = np.zeros([ndx + nu, ndx + nu])
        self.F = np.zeros([nout, ndx + nu])

        self.Lx = self.g[:ndx]
        self.Lu = self.g[ndx:]
        self.Lxx = self.L[:ndx, :ndx]
        self.Lxu = self.L[:ndx, ndx:]
        self.Luu = self.L[ndx:, ndx:]
        self.Fx = self.F[:, :ndx]
        self.Fu = self.F[:, ndx:]


# Creating the DAM for the cartpole
cartpoleDAM = DifferentialActionModelCartpole()
cartpoleData = cartpoleDAM.createData()
cartpoleDAM = model = DifferentialActionModelCartpole()

# Using NumDiff for computing the derivatives. We specify the
# withGaussApprox=True to have approximation of the Hessian based on the
# Jacobian of the cost residuals.
cartpoleND = DifferentialActionModelNumDiff(cartpoleDAM, withGaussApprox=True)

# Getting the IAM using the simpletic Euler rule
timeStep = 5e-2
cartpoleIAM = IntegratedActionModelEuler(cartpoleND, timeStep)

# Creating the shooting problem
x0 = np.array([0., 3.14, 0., 0.])
T = 50

terminalCartpole = DifferentialActionModelCartpole()
terminalCartpoleDAM = DifferentialActionModelNumDiff(terminalCartpole, withGaussApprox=True)
terminalCartpoleIAM = IntegratedActionModelEuler(terminalCartpoleDAM)

terminalCartpole.costWeights[0] = 100
terminalCartpole.costWeights[1] = 100
terminalCartpole.costWeights[2] = 1.
terminalCartpole.costWeights[3] = 0.1
terminalCartpole.costWeights[4] = 0.01
terminalCartpole.costWeights[5] = 0.0001
problem = ShootingProblem(x0, [cartpoleIAM] * T, terminalCartpoleIAM)

# Solving it using DDP
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPVerbose()]
xs, us, done = ddp.solve(maxiter=300)

HTML(animateCartpole(xs).to_html5_video())
