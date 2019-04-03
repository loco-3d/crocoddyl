import numpy as np
from cost import CostModelState, CostModelSum
from floating_contact import DifferentialActionModelFloatingInContact
from state import StateVector
from utils import EPS


class ActionModelAbstract:
    """ Abstract class for action models.

    In crocoddyl, an action model combines dynamics and cost data. Each node,
    in our optimal control problem, is described through an action model. Every
    time that we want describe a problem, we need to provide ways of computing
    the dynamics, cost functions and their derivatives. These computations are
    mainly carry on inside calc() and calcDiff(), respectively.
    """

    def __init__(self, State, nu):
        """ Construct common variables for action models.

        :param state: state description
        :param nu: dimension of control vector
        """
        self.State = State
        self.nx = State.nx
        self.ndx = State.ndx
        self.nu = nu
        self.unone = np.zeros(self.nu)

    def createData(self):
        """ Create the action data.

        Each action model (AM) has its own data that needs to be allocated.
        This function returns the allocated data for a predefined AM. Note that
        you need to defined the ActionDataType inside your AM.
        :return AM data.
        """
        return self.ActionDataType(self)

    def calc(model, data, x, u=None):
        """ Compute the next state and cost value.

        First, it describes the time-discrete evolution of our dynamical system
        in which we obtain the next discrete state. Additionally it computes
        the cost value associated to this discrete state and control pair.
        :param model: action model
        :param data: action data
        :param x: time-discrete state vector
        :param u: time-discrete control input
        :returns the next state and cost value
        """
        raise NotImplementedError("Not implemented yet.")

    def calcDiff(model, data, x, u=None, recalc=True):
        """ Compute the derivatives of the dynamics and cost functions.

        It computes the partial derivatives of the dynamical system and the
        cost function. If recalc == True, it first updates the state evolution
        and cost value. This function builds a quadratic approximation of the
        action model (i.e. dynamical system and cost function).
        :param model: action model
        :param data: action data
        :param x: time-discrete state vector
        :param u: time-discrete control input
        :param recalc: If true, it updates the state evolution and the cost
        value.
        :returns the next state and cost value
        """
        raise NotImplementedError("Not implemented yet.")


class ActionDataAbstract:
    def __init__(self, model, costData=None):
        """ Create common data shared between AMs.

        In crocoddyl, an action data might use an externally defined cost data.
        If so, you need to pass your own cost data using costData. Otherwise
        it will be allocated here.
        :param model: action model
        :param costData: external cost data (optional)
        """
        nx, ndx, nu = model.nx, model.ndx, model.nu
        # State evolution and cost data
        self.cost = np.nan
        self.xnext = np.zeros(nx)

        # Dynamics data
        self.Fx = np.zeros([ndx, ndx])
        self.Fu = np.zeros([ndx, nu])

        # Cost data
        if costData is None:
            self.g = np.zeros([ndx + nu])
            self.L = np.zeros([ndx + nu, ndx + nu])
            self.Lx = self.g[:ndx]
            self.Lu = self.g[ndx:]
            self.Lxx = self.L[:ndx, :ndx]
            self.Lxu = self.L[:ndx, ndx:]
            self.Luu = self.L[ndx:, ndx:]
            if hasattr(model, 'ncost') and model.ncost > 1:
                ncost = model.ncost
                self.costResiduals = np.zeros(ncost)
                self.R = np.zeros([ncost, ndx + nu])
                self.Rx = self.R[:, ndx:]
                self.Ru = self.R[:, ndx:]
        else:
            self.costs = costData
            self.Lx = self.costs.Lx
            self.Lu = self.costs.Lu
            self.Lxx = self.costs.Lxx
            self.Lxu = self.costs.Lxu
            self.Luu = self.costs.Luu
            if model.ncost > 1:
                self.costResiduals = self.costs.residuals
                self.Rx = self.costs.Rx
                self.Ru = self.costs.Ru


class ActionModelLQR(ActionModelAbstract):
    def __init__(self, nx, nu, driftFree=True):
        """ Define an action model for a LQR problem.

        A Linear-Quadratic Regulator problem has a transition model of the form
        xnext(x,u) = Fx*x + Fu*u + f0. Its cost function is quadratic of the
        form: 1/2 [x,u].T [Lxx Lxu; Lxu.T Luu] [x,u] + [lx,lu].T [x,u]
        """
        ActionModelAbstract.__init__(self, StateVector(nx), nu)
        self.ActionDataType = ActionDataLQR

        from utils import randomOrthonormalMatrix
        self.Fx = randomOrthonormalMatrix(self.ndx)
        self.Fu = randomOrthonormalMatrix(self.ndx)[:, :self.nu]
        self.f0 = np.zeros(self.ndx) if driftFree else np.random.rand(self.ndx)
        A = np.random.rand(self.ndx + self.nu, self.ndx + self.nu)
        L = np.dot(A.T, A)
        self.Lxx = L[:self.ndx, :self.ndx]
        self.Lxu = L[:self.ndx, self.ndx:]
        self.Luu = L[self.ndx:, self.ndx:]
        self.lx = np.random.rand(self.ndx)
        self.lu = np.random.rand(self.nu)

    def calc(model, data, x, u=None):
        """ Update the next state and the cost function.

        :param model: action model
        :params data: action data
        :params x: time-discrete state vector
        :params u: time-discrete control vector
        :returns the next state and cost value
        """
        if u is None:
            u = model.unone
        data.xnext[:] = np.dot(model.Fx, x) + np.dot(model.Fu, u) + model.f0
        data.cost = 0.5 * np.dot(x, np.dot(model.Lxx, x)) + 0.5 * np.dot(u, np.dot(model.Luu, u))
        data.cost += np.dot(x, np.dot(model.Lxu, u)) + np.dot(model.lx, x) + np.dot(model.lu, u)
        return data.xnext, data.cost

    def calcDiff(model, data, x, u=None, recalc=True):
        """ Update the derivatives of the dynamics and cost.

        :param model: action model
        :params data: action data
        :params x: time-discrete state vector
        :params u: time-discrete control vector
        :returns the next state and cost value
        """
        if u is None:
            u = model.unone
        if recalc:
            xnext, cost = model.calc(data, x, u)
        data.Lx[:] = model.lx + np.dot(model.Lxx, x) + np.dot(model.Lxu, u)
        data.Lu[:] = model.lu + np.dot(model.Lxu.T, x) + np.dot(model.Luu, u)
        data.Fx[:, :] = model.Fx
        data.Fu[:, :] = model.Fu
        data.Lxx[:, :] = model.Lxx
        data.Luu[:, :] = model.Luu
        data.Lxu[:, :] = model.Lxu
        return xnext, cost


class ActionDataLQR(ActionDataAbstract):
    def __init__(self, model):
        ActionDataAbstract.__init__(self, model)

        # Setting the linear model and quadratic cost here because they are
        # constant
        self.Fx[:, :] = model.Fx
        self.Fu[:, :] = model.Fu
        self.Lxx[:, :] = model.Lxx
        self.Luu[:, :] = model.Luu
        self.Lxu[:, :] = model.Lxu


class ActionModelNumDiff(ActionModelAbstract):
    """ Abstract action model that uses NumDiff for derivative computation.
    """

    def __init__(self, model, withGaussApprox=False):
        ActionModelAbstract.__init__(self, model.State, model.nu)
        self.ActionDataType = ActionDataNumDiff
        self.model0 = model
        self.disturbance = np.sqrt(2 * EPS)
        self.ncost = model.ncost if hasattr(model, 'ncost') else 1
        self.withGaussApprox = withGaussApprox
        assert (not self.withGaussApprox or self.ncost > 1)

    def calc(model, data, x, u):
        return model.model0.calc(data.data0, x, u)

    def calcDiff(model, data, x, u):
        xn0, c0 = model.calc(data, x, u)
        h = model.disturbance

        def dist(i, n, h):
            return np.array([h if ii == i else 0 for ii in range(n)])

        def Xint(x, dx):
            return model.State.integrate(x, dx)

        def Xdiff(x1, x2):
            return model.State.diff(x1, x2)

        model._assertStableStateFD(x)
        for ix in range(model.ndx):
            xn, c = model.model0.calc(data.datax[ix], Xint(x, dist(ix, model.ndx, h)), u)
            data.Fx[:, ix] = Xdiff(xn0, xn) / h
            data.Lx[ix] = (c - c0) / h
            if model.ncost > 1:
                data.Rx[:, ix] = (data.datax[ix].costResiduals - data.data0.costResiduals) / h
        for iu in range(model.nu):
            xn, c = model.model0.calc(data.datau[iu], x, u + dist(iu, model.nu, h))
            data.Fu[:, iu] = Xdiff(xn0, xn) / h
            data.Lu[iu] = (c - c0) / h
            if model.ncost > 1:
                data.Ru[:, iu] = (data.datau[iu].costResiduals - data.data0.costResiduals) / h
        if model.withGaussApprox:
            data.Lxx[:, :] = np.dot(data.Rx.T, data.Rx)
            data.Lxu[:, :] = np.dot(data.Rx.T, data.Ru)
            data.Lux[:, :] = data.Lxu.T
            data.Luu[:, :] = np.dot(data.Ru.T, data.Ru)

    def _assertStableStateFD(model, x):
        """ Make sure that when we finite difference the Action Model, the user does
        not face unknown behaviour because of the finite differencing of a quaternion around pi.
        This behaviour might occur if CostModelState and FloatingInContact differential model are used
        together.

        For full discussions see issue https://gepgitlab.laas.fr/loco-3d/crocoddyl/issues/139
        """
        if hasattr(model.model0, "differential"):
            md = model.model0.differential
            if isinstance(md, DifferentialActionModelFloatingInContact):
                if hasattr(md, "costs"):
                    mc = md.costs
                    if isinstance(mc, CostModelState):
                        assert (~np.isclose(model.State.diff(mc.ref, x)[3:6], np.ones(3) * np.pi, atol=1e-6).any())
                        assert (~np.isclose(model.State.diff(mc.ref, x)[3:6], -np.ones(3) * np.pi, atol=1e-6).any())
                    elif isinstance(mc, CostModelSum):
                        for (key, cost) in mc.costs.iteritems():
                            if isinstance(cost.cost, CostModelState):
                                assert (~np.isclose(
                                    model.State.diff(cost.cost.ref, x)[3:6], np.ones(3) * np.pi, atol=1e-6).any())
                                assert (~np.isclose(
                                    model.State.diff(cost.cost.ref, x)[3:6], -np.ones(3) * np.pi, atol=1e-6).any())


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
