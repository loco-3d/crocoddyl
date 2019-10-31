import numpy as np
import pinocchio
import crocoddyl


class CostModelDoublePendulum(crocoddyl.costModelAbstract):
    def __init__(self, state, nu, activation):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u):
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])
        data.r = np.array([s1, s2, 1 - c1, 1 - c2, x[2], x[3]])
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        c1, c2 = np.cos(x[0]), np.cos(x[1])
        s1, s2 = np.sin(x[0]), np.sin(x[1])

        self.activation.calcDiff(data, data.r, recalc)
        Ax, Axx = self.activation.Ar, self.activation.Arr

        J = np.zeros([6, 4])
        J[:2, :2] = np.diag([c1, c2])
        J[2:4, :2] = np.diag([s1, s2])
        J[4:6, 2:4] = np.diag([1, 1])
        data.Lx = np.dot(J.T, Ax)

        H = np.zeros([6, 4])
        H[:2, :2] = np.diag([c1**2 - s1**2, c2**2 - s2**2])
        H[2:4, :2] = np.diag([s1**2 + (1 - c1) * c1, s2**2 + (1 - c2) * c2])
        H[4:6, 2:4] = np.diag([1, 1])
        Lxx = np.dot(H.T, Axx)
        data.Lxx = np.diag([Lxx[0, 0], Lxx[1, 0], Lxx[2, 0], Lxx[3, 0]])

class ActuationModelDoublePendulum:
    def __init__(self, pinocchioModel, actLink):
        self.pinocchio = pinocchioModel
        self.nq = pinocchioModel.nq
        self.nv = pinocchioModel.nv
        self.nx = self.nq + self.nv
        self.ndx = self.nv * 2
        self.nu = 1
        self.actLink = actLink

    def calc(self, data, x, u):
        S = np.zeros([self.nv, self.nu])
        if self.actLink == 1:
            S[0] = 1
        else:
            S[1] = 1
        data.a[:] = np.dot(S, u)
        return data.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        return data.a

    def createData(self, pinocchioData):
        return ActuationDataDoublePendulum(self, pinocchioData)
