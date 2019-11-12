import numpy as np
import pinocchio
import crocoddyl


class CostModelDoublePendulum(crocoddyl.CostModelAbstract):
    def __init__(self, state, nu, activation):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u):
        c1, c2 = np.cos(x[0, 0]), np.cos(x[1, 0])
        s1, s2 = np.sin(x[0, 0]), np.sin(x[1, 0])
        data.r = np.matrix([s1, s2, 1 - c1, 1 - c2, x[2, 0], x[3, 0]]).T
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        c1, c2 = np.cos(x[0, 0]), np.cos(x[1, 0])
        s1, s2 = np.sin(x[0, 0]), np.sin(x[1, 0])

        self.activation.calcDiff(data.activation, data.r, recalc)
        Ax, Axx = data.activation.Ar, data.activation.Arr

        J = pinocchio.utils.zero((6, 4))
        J[:2, :2] = np.diag([c1, c2])
        J[2:4, :2] = np.diag([s1, s2])
        J[4:6, 2:4] = np.diag([1, 1])
        data.Lx = J.T * Ax

        H = pinocchio.utils.zero((6, 4))
        H[:2, :2] = np.diag([c1**2 - s1**2, c2**2 - s2**2])
        H[2:4, :2] = np.diag([s1**2 + (1 - c1) * c1, s2**2 + (1 - c2) * c2])
        H[4:6, 2:4] = np.diag([1, 1])
        Lxx = H.T * Axx
        data.Lxx = np.diag([Lxx[0, 0], Lxx[1, 0], Lxx[2, 0], Lxx[3, 0]])


class ActuationModelDoublePendulum(crocoddyl.ActuationModelAbstract):
    def __init__(self, state, actLink):
        crocoddyl.ActuationModelAbstract.__init__(self, state, 1)
        self.nv = state.nv
        self.actLink = actLink

    def calc(self, data, x, u):
        S = pinocchio.utils.zero((self.nv, self.nu))
        if self.actLink == 1:
            S[0] = 1
        else:
            S[1] = 1
        
        data.tau = S * u

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)

        S = pinocchio.utils.zero((2, 1))
        if self.actLink == 1:
            S[0] = 1
        else:
            S[1] = 1
        
        data.dtau_du = S
