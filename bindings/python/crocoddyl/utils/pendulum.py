import crocoddyl
import pinocchio
import numpy as np


class CostModelDoublePendulum(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation, nu):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu=nu)

    def calc(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])
        data.r = np.array([s1, s2, 1 - c1, 1 - c2, x[2], x[3]])
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])

        self.activation.calcDiff(data.activation, data.r)

        J = pinocchio.utils.zero((6, 4))
        J[:2, :2] = np.diag([c1, c2])
        J[2:4, :2] = np.diag([s1, s2])
        J[4:6, 2:4] = np.diag([1, 1])
        data.Lx = np.dot(J.T, data.activation.Ar)

        H = pinocchio.utils.zero((6, 4))
        H[:2, :2] = np.diag([c1**2 - s1**2, c2**2 - s2**2])
        H[2:4, :2] = np.diag([s1**2 + (1 - c1) * c1, s2**2 + (1 - c2) * c2])
        H[4:6, 2:4] = np.diag([1, 1])
        Lxx = np.dot(H.T, np.diag(data.activation.Arr))
        data.Lxx = np.diag(Lxx)


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

    def calcDiff(self, data, x, u):
        S = np.zeros((self.nv, self.nu))
        if self.actLink == 1:
            S[0] = 1
        else:
            S[1] = 1

        data.dtau_du = S
