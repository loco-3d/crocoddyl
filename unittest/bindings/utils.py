import crocoddyl
import numpy as np


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


class StateVectorDerived(crocoddyl.StateAbstract):
    def __init__(self, nx):
        crocoddyl.StateAbstract.__init__(self, nx, nx)

    def zero(self):
        return np.matrix(np.zeros(self.nx)).T

    def rand(self):
        return np.matrix(np.random.rand(self.nx)).T

    def diff(self, x0, x1):
        dx = x1 - x0
        return x1 - x0

    def integrate(self, x, dx):
        return x + dx

    def Jdiff(self, x1, x2, firstsecond='both'):
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jdiff(x1, x2, 'first'), self.Jdiff(x1, x2, 'second')]

        J = np.zeros([self.ndx, self.ndx])
        if firstsecond == 'first':
            J[:, :] = -np.eye(self.ndx)
        elif firstsecond == 'second':
            J[:, :] = np.eye(self.ndx)
        return J

    def Jintegrate(self, x, dx, firstsecond='both'):
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jintegrate(x, dx, 'first'), self.Jintegrate(x, dx, 'second')]
        return np.eye(self.ndx)


class UnicycleDerived(crocoddyl.ActionModelAbstract):
    def __init__(self):
        self.state = crocoddyl.StateVector(3)
        crocoddyl.ActionModelAbstract.__init__(self, self.state, 2, 5)
        self.dt = .1
        self.costWeights = [10., 1.]

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        v, w = m2a(u)
        px, py, theta = m2a(x)
        c, s = np.cos(theta), np.sin(theta)
        # Rollout the dynamics
        data.xnext = a2m([px + c * v * self.dt, py + s * v * self.dt, theta + w * self.dt])
        # Compute the cost value
        data.costResiduals = np.vstack([self.costWeights[0] * x, self.costWeights[1] * u])
        data.cost = .5 * sum(m2a(data.costResiduals)**2)
        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        xnext, cost = self.calc(data, x, u)
        v, w = m2a(u)
        px, py, theta = m2a(x)
        # Cost derivatives
        data.Lx = a2m(m2a(x) * ([self.costWeights[0]**2] * self.nx))
        data.Lu = a2m(m2a(u) * ([self.costWeights[1]**2] * self.nu))
        data.Lxx = np.diag([self.costWeights[0]**2] * self.nx)
        data.Luu = np.diag([self.costWeights[1]**2] * self.nu)
        # Dynamic derivatives
        c, s, dt = np.cos(theta), np.sin(theta), self.dt
        v, w = m2a(u)
        data.Fx = np.matrix([[1, 0, -s * v * dt], [0, 1, c * v * dt], [0, 0, 1]])
        data.Fu = np.matrix([[c * self.dt, 0], [s * self.dt, 0], [0, self.dt]])
        return xnext, cost
