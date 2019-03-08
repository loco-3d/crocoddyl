import numpy as np

from state import StateVector
from utils import EPS


class ActionModelLQR:
    def __init__(self, nx, nu):
        """ Define an action model for a LQR problem.

        A Linear-Quadratic Regulator problem has a transition model of the form
        xnext(x,u) = Fx*x + Fu*x. Its cost function is quadratic of the form:
        cost(x,u) = 1/2 [x,u].T [Lxx Lxu ; Lxu.T Luu ] [x,u] + [Lx,Lu].T [x,u]
        """
        self.State = StateVector(nx)
        self.nx = self.State.nx
        self.ndx = self.State.ndx
        self.nu = nu

        self.Lx = None
        self.Lu = None
        self.Lxx = None
        self.Lxu = None
        self.Luu = None
        self.Fx = None
        self.Fu = None

        self.unone = np.zeros(self.nu)

    def setUpRandom(self):
        self.Lx = np.random.rand(self.ndx)
        self.Lu = np.random.rand(self.nu)
        self.Ls = L = np.random.rand(self.ndx + self.nu, self.ndx + self.nu) * 2 - 1
        self.L = L = .5 * np.dot(L.T, L)
        self.Lxx = L[:self.ndx, :self.ndx]
        self.Lxu = L[:self.ndx, self.ndx:]
        self.Luu = L[self.ndx:, self.ndx:]
        self.Fx = np.random.rand(self.ndx, self.ndx) * 2 - 1
        self.Fu = np.random.rand(self.ndx, self.nu) * 2 - 1
        self.F = np.random.rand(self.nx)  # Affine (nonautom) part of the dynamics

    def createData(self):
        """ Create the LQR action data
        """
        return ActionDataLQR(self)

    def calc(self, data, x, u=None):
        """ Update the next state and the cost function.

        :params data: action data
        :params x: state
        :params u: control
        :returns xnext,cost for current state,control pair data.x,data.u.
        """
        if u is None:
            u = self.unone

        def quad(a, Q, b):
            return .5 * np.dot(np.dot(Q, b).T, a)

        data.xnext = np.dot(self.Fx, x) + np.dot(self.Fu, u) + self.F
        data.cost = quad(x, self.Lxx, x) + 2 * quad(x, self.Lxu, u) + quad(u, self.Luu, u) + np.dot(
            self.Lx, x) + np.dot(self.Lu, u)
        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None):
        """ Update the derivatives of the dynamics and cost.

        :params data: action data
        :params x: state
        :params u: control
        :returns xnext,cost for current state,control pair data.x,data.u.
        """
        if u is None:
            u = self.unone
        xnext, cost = self.calc(data, x, u)
        data.Lx = self.Lx + np.dot(self.Lxx, x) + np.dot(self.Lxu, u)
        data.Lu = self.Lu + np.dot(self.Lxu.T, x) + np.dot(self.Luu, u)
        data.Lxx = self.Lxx
        data.Lxu = self.Lxu
        data.Luu = self.Luu
        data.Fx = self.Fx
        data.Fu = self.Fu
        return xnext, cost


class ActionDataLQR:
    def __init__(self, actionModel):
        assert (isinstance(actionModel, ActionModelLQR))
        self.model = actionModel

        self.xnext = np.zeros([self.model.nx])
        self.cost = np.nan
        self.Lx = np.zeros([self.model.ndx])
        self.Lu = np.zeros([self.model.nu])
        self.Lxx = self.model.Lxx
        self.Lxu = self.model.Lxu
        self.Luu = self.model.Luu
        self.Fx = self.model.Fx
        self.Fu = self.model.Fu


class ActionModelNumDiff:
    """ Abstract action model that uses NumDiff for derivative computation.
    """

    def __init__(self, model, withGaussApprox=False):
        self.model0 = model
        self.nx = model.nx
        self.ndx = model.ndx
        self.nu = model.nu
        self.State = model.State
        self.disturbance = np.sqrt(2 * EPS)
        try:
            self.ncost = model.ncost
        except:
            self.ncost = 1
        self.withGaussApprox = withGaussApprox
        assert (not self.withGaussApprox or self.ncost > 1)

    def createData(self):
        return ActionDataNumDiff(self)

    def calc(self, data, x, u):
        return self.model0.calc(data.data0, x, u)

    def calcDiff(self, data, x, u):
        xn0, c0 = self.calc(data, x, u)
        h = self.disturbance
        dist = lambda i, n, h: np.array([h if ii == i else 0 for ii in range(n)])
        Xint = lambda x, dx: self.State.integrate(x, dx)
        Xdiff = lambda x1, x2: self.State.diff(x1, x2)
        for ix in range(self.ndx):
            xn, c = self.model0.calc(data.datax[ix], Xint(x, dist(ix, self.ndx, h)), u)
            data.Fx[:, ix] = Xdiff(xn0, xn) / h
            data.Lx[ix] = (c - c0) / h
            if self.ncost > 1:
                data.Rx[:, ix] = (data.datax[ix].costResiduals - data.data0.costResiduals) / h
        for iu in range(self.nu):
            xn, c = self.model0.calc(data.datau[iu], x, u + dist(iu, self.nu, h))
            data.Fu[:, iu] = Xdiff(xn0, xn) / h
            data.Lu[iu] = (c - c0) / h
            if self.ncost > 1:
                data.Ru[:, iu] = (data.datau[iu].costResiduals - data.data0.costResiduals) / h
        if self.withGaussApprox:
            data.Lxx[:, :] = np.dot(data.Rx.T, data.Rx)
            data.Lxu[:, :] = np.dot(data.Rx.T, data.Ru)
            data.Lux[:, :] = data.Lxu.T
            data.Luu[:, :] = np.dot(data.Ru.T, data.Ru)


class ActionDataNumDiff:
    def __init__(self, model):
        ndx, nu = model.ndx, model.nu
        self.data0 = model.model0.createData()
        self.datax = [model.model0.createData() for i in range(model.ndx)]
        self.datau = [model.model0.createData() for i in range(model.nu)]
        self.Lx = np.zeros([model.ndx])
        self.Lu = np.zeros([model.nu])
        self.Fx = np.zeros([model.ndx, model.ndx])
        self.Fu = np.zeros([model.ndx, model.nu])
        if model.ncost > 1:
            self.Rx = np.zeros([model.ncost, model.ndx])
            self.Ru = np.zeros([model.ncost, model.nu])
        if model.withGaussApprox:
            self.L = np.zeros([ndx + nu, ndx + nu])
            self.Lxx = self.L[:ndx, :ndx]
            self.Lxu = self.L[:ndx, ndx:]
            self.Lux = self.L[ndx:, :ndx]
            self.Luu = self.L[ndx:, ndx:]
