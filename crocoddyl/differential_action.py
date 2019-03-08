import numpy as np
from numpy.random import rand

import pinocchio
from state import StatePinocchio, StateVector
from utils import EPS, a2m, randomOrthonormalMatrix


class DifferentialActionModelAbstract:
    """ Abstract class for the differential action model.

    In crocoddyl, an action model combines dynamics and cost data. Each node, in
    our optimal control problem, is described through an action model. Every
    time that we want describe a problem, we need to provide ways of computing
    the dynamics, cost functions and their derivatives. These computations are
    mainly carry on inside calc() and calcDiff(), respectively.
    """

    def __init__(self, nq, nv, nu):
        self.nq = nq
        self.nv = nv
        self.nu = nu
        self.nx = nq + nv
        self.ndx = 2 * nv
        self.nout = nv
        self.unone = np.zeros(self.nu)

    def createData(self):
        """ Create the differential action data.

        Each differential action model has its own data that needs to be
        allocated. This function returns the allocated data for a predefined
        DAM. Note that you need to defined the DifferentialActionDataType inside
        your DAM.
        :return DAM data.
        """
        return self.DifferentialActionDataType(self)

    def calc(self, data, x, u=None):
        """ Compute the state evolution and cost value.

        First, it describes the time-continuous evolution of our dynamical system
        in which along predefined integrated action self we might obtain the
        next discrete state. Indeed it computes the time derivatives of the
        state from a predefined dynamical system. Additionally it computes the
        cost value associated to this state and control pair.
        :param self: differential action model
        :param data: differential action data
        :param x: state vector
        :param u: control input
        """
        raise NotImplementedError("Not implemented yet.")

    def calcDiff(self, data, x, u=None, recalc=True):
        """ Compute the derivatives of the dynamics and cost functions.

        It computes the partial derivatives of the dynamical system and the cost
        function. If recalc == True, it first updates the state evolution and
        cost value. This function builds a quadratic approximation of the
        time-continuous action model (i.e. dynamical system and cost function).
        :param model: differential action model
        :param data: differential action data
        :param x: state vector
        :param u: control input
        :param recalc: If true, it updates the state evolution and the cost
        value.
        """
        raise NotImplementedError("Not implemented yet.")


class DifferentialActionDataAbstract:
    def __init__(self, model, costData=None):
        """ Create common data shared between DAMs.

        In crocoddyl, a DAD might use an externally defined cost data. If so,
        you need to pass your own cost data using costData. Otherwise it will
        be allocated here.
        :param model: differential action model
        :param costData: external cost data (optional)
        """
        nu, ndx, nout = model.nu, model.ndx, model.nout
        # State evolution and cost data
        self.cost = np.nan
        self.xout = np.zeros(nout)

        # Dynamics data
        self.Fx = np.zeros([nout, ndx])
        self.Fu = np.zeros([nout, nu])

        # Cost data
        if costData is None:
            self.Lx = np.zeros(ndx)
            self.Lu = np.zeros(nu)
            self.Lxx = np.zeros([ndx, ndx])
            self.Lxu = np.zeros([ndx, nu])
            self.Luu = np.zeros([nu, nu])
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


class DifferentialActionModelFullyActuated(DifferentialActionModelAbstract):
    def __init__(self, pinocchioModel, costModel):
        DifferentialActionModelAbstract.__init__(self, pinocchioModel.nq, pinocchioModel.nv, pinocchioModel.nv)
        self.DifferentialActionDataType = DifferentialActionDataFullyActuated
        self.pinocchio = pinocchioModel
        self.State = StatePinocchio(self.pinocchio)
        self.costs = costModel
        # Use this to force the computation with ABA
        # Side effect is that armature is not used.
        self.forceAba = False

    @property
    def ncost(self):
        return self.costs.ncost

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        nq, nv = self.nq, self.nv
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(u)
        # --- Dynamics
        if self.forceAba:
            data.xout[:] = pinocchio.aba(self.pinocchio, data.pinocchio, q, v, tauq).flat
        else:
            pinocchio.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
            data.M = data.pinocchio.M
            if hasattr(self.pinocchio, 'armature'):
                data.M[range(nv), range(nv)] += self.pinocchio.armature.flat
            data.Minv = np.linalg.inv(data.M)
            data.xout[:] = data.Minv * (tauq - data.pinocchio.nle).flat
        # --- Cost
        pinocchio.forwardKinematics(self.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
        data.cost = self.costs.calc(data.costs, x, u)
        return data.xout, data.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            xout, cost = self.calc(data, x, u)
        nq, nv = self.nq, self.nv
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(u)
        a = a2m(data.xout)
        # --- Dynamics
        if self.forceAba:
            pinocchio.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, tauq)
            data.Fx[:, :nv] = data.pinocchio.ddq_dq
            data.Fx[:, nv:] = data.pinocchio.ddq_dv
            data.Fu[:, :] = data.Minv
        else:
            pinocchio.computeRNEADerivatives(self.pinocchio, data.pinocchio, q, v, a)
            data.Fx[:, :nv] = -np.dot(data.Minv, data.pinocchio.dtau_dq)
            data.Fx[:, nv:] = -np.dot(data.Minv, data.pinocchio.dtau_dv)
            data.Fu[:, :] = data.Minv
        # --- Cost
        pinocchio.computeJointJacobians(self.pinocchio, data.pinocchio, q)
        pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
        self.costs.calcDiff(data.costs, x, u, recalc=False)
        return data.xout, data.cost


class DifferentialActionDataFullyActuated(DifferentialActionDataAbstract):
    def __init__(self, model):
        self.pinocchio = model.pinocchio.createData()
        costData = model.costs.createData(self.pinocchio)
        DifferentialActionDataAbstract.__init__(self, model, costData)


class DifferentialActionModelLQR(DifferentialActionModelAbstract):
    """ Differential action model for linear dynamics and quadracti cost.

  This class implements a linear dynamics, and quadratic costs (i.e. LQR action).
  Since the DAM is a second order system, and the integratedactionmodels are
  implemented as being second order integrators, This class implements a second
  order linear system given by
    x = [q, v]
    dv = A v + B q + C u + d
  where  A, B, C and d are constant terms.

  On the other hand the cost function is given by
    l(x,u) = x^T*Q*x + u^T*U*u
  """

    def __init__(self, nq, nu):
        DifferentialActionModelAbstract.__init__(self, nq, nq, nu)
        self.DifferentialActionDataType = DifferentialActionDataLQR
        self.State = StateVector(self.nx)

        v1 = randomOrthonormalMatrix(self.nq)
        v2 = randomOrthonormalMatrix(self.nq)
        v3 = randomOrthonormalMatrix(self.nq)

        # linear dynamics and quadratic cost terms
        self.A = v2
        self.B = v1
        self.C = v3
        self.d = rand(self.nv)
        self.Q = randomOrthonormalMatrix(self.nx)
        self.U = randomOrthonormalMatrix(self.nu)

    def calc(self, data, x, u=None):
        q = x[:self.nq]
        v = x[self.nq:]
        data.xout[:] = (np.dot(self.A, v) + np.dot(self.B, q) + np.dot(self.C, u)).flat + self.d
        data.cost = np.dot(x, np.dot(self.Q, x)) + np.dot(u, np.dot(self.U, u))
        return data.xout, data.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            xout, cost = self.calc(data, x, u)
        data.Lx[:] = np.dot(x.T, data.Lxx)
        data.Lu[:] = np.dot(u.T, data.Luu)
        return data.xout, data.cost


class DifferentialActionDataLQR(DifferentialActionDataAbstract):
    def __init__(self, model):
        DifferentialActionDataAbstract.__init__(self, model)

        # Setting the linear model and quadratic cost here because they are constant
        self.Fx[:, :model.nv] = model.B
        self.Fx[:, model.nv:] = model.A
        self.Fu[:, :] = model.C
        np.copyto(self.Lxx, model.Q + model.Q.T)
        np.copyto(self.Lxu, np.zeros((model.nx, model.nu)))
        np.copyto(self.Luu, model.U + model.U.T)


class DifferentialActionModelNumDiff(DifferentialActionModelAbstract):
    def __init__(self, model, withGaussApprox=False):
        self.DifferentialActionDataType = DifferentialActionDataNumDiff
        self.model0 = model
        self.nx = model.nx
        self.ndx = model.ndx
        self.nout = model.nout
        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv
        self.State = model.State
        self.disturbance = np.sqrt(2 * EPS)
        try:
            self.ncost = model.ncost
        except:
            self.ncost = 1
        self.withGaussApprox = withGaussApprox
        assert (not self.withGaussApprox or self.ncost > 1)

    def calc(self, data, x, u):
        return self.model0.calc(data.data0, x, u)

    def calcDiff(self, data, x, u, recalc=True):
        xn0, c0 = self.calc(data, x, u)
        h = self.disturbance
        dist = lambda i, n, h: np.array([h if ii == i else 0 for ii in range(n)])
        Xint = lambda x, dx: self.State.integrate(x, dx)
        for ix in range(self.ndx):
            xn, c = self.model0.calc(data.datax[ix], Xint(x, dist(ix, self.ndx, h)), u)
            data.Fx[:, ix] = (xn - xn0) / h
            data.Lx[ix] = (c - c0) / h
            if self.ncost > 1:
                data.Rx[:, ix] = (data.datax[ix].costResiduals - data.data0.costResiduals) / h
        if u is not None:
            for iu in range(self.nu):
                xn, c = self.model0.calc(data.datau[iu], x, u + dist(iu, self.nu, h))
                data.Fu[:, iu] = (xn - xn0) / h
                data.Lu[iu] = (c - c0) / h
                if self.ncost > 1:
                    data.Ru[:, iu] = (data.datau[iu].costResiduals - data.data0.costResiduals) / h
        if self.withGaussApprox:
            data.Lxx[:, :] = np.dot(data.Rx.T, data.Rx)
            data.Lxu[:, :] = np.dot(data.Rx.T, data.Ru)
            data.Luu[:, :] = np.dot(data.Ru.T, data.Ru)


class DifferentialActionDataNumDiff:
    def __init__(self, model):
        ndx, nu, nout = model.ndx, model.nu, model.nout
        self.data0 = model.model0.createData()
        self.datax = [model.model0.createData() for i in range(model.ndx)]
        self.datau = [model.model0.createData() for i in range(model.nu)]

        # Dynamics data
        self.F = np.zeros([nout, ndx + nu])
        self.Fx = self.F[:, :ndx]
        self.Fu = self.F[:, ndx:]

        # Cost data
        self.g = np.zeros(ndx + nu)
        self.L = np.zeros([ndx + nu, ndx + nu])
        self.Lx = self.g[:ndx]
        self.Lu = self.g[ndx:]
        if model.ncost > 1:
            self.costResiduals = self.data0.costResiduals
            self.R = np.zeros([model.ncost, ndx + nu])
            self.Rx = self.R[:, :ndx]
            self.Ru = self.R[:, ndx:]
        if model.withGaussApprox:
            self.Lxx = self.L[:ndx, :ndx]
            self.Lxu = self.L[:ndx, ndx:]
            self.Luu = self.L[ndx:, ndx:]
