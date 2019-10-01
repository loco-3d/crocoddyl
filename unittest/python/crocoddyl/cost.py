from collections import OrderedDict

import numpy as np

import pinocchio

from .activation import ActivationModelInequality, ActivationModelQuad
from .utils import EPS, m2a


class CostModelPinocchio:
    """ Abstract for Pinocchio-based cost models.

    It defines a template of cost model whose residual and derivatives
    can be retrieved from Pinocchio data, through the calc and calcDiff
    functions, respectively.
    """
    def __init__(self, pinocchioModel, ncost, withResiduals=True, nu=None):
        self.ncost = ncost
        self.nq = pinocchioModel.nq
        self.nv = pinocchioModel.nv
        self.nx = self.nq + self.nv
        self.ndx = self.nv + self.nv
        self.nu = nu if nu is not None else pinocchioModel.nv
        self.pinocchio = pinocchioModel
        self.withResiduals = withResiduals

    def createData(self, pinocchioData):
        return self.CostDataType(self, pinocchioData)

    def calc(self, data, x, u):
        assert (False and "This should be defined in the derivative class.")

    def calcDiff(self, data, x, u, recalc=True):
        assert (False and "This should be defined in the derivative class.")


class CostDataPinocchio:
    """ Abstract for Pinocchio-based cost datas.

    It stores the data corresponting to the CostModelPinocchio class.
    """
    def __init__(self, model, pinocchioData):
        ncost, nv, ndx, nu = model.ncost, model.nv, model.ndx, model.nu
        self.pinocchio = pinocchioData
        self.cost = np.nan
        self.g = np.zeros(ndx + nu)
        self.L = np.zeros([ndx + nu, ndx + nu])

        self.Lx = self.g[:ndx]
        self.Lu = self.g[ndx:]
        self.Lxx = self.L[:ndx, :ndx]
        self.Lxu = self.L[:ndx, ndx:]
        self.Luu = self.L[ndx:, ndx:]

        self.Lq = self.Lx[:nv]
        self.Lqq = self.Lxx[:nv, :nv]
        self.Lv = self.Lx[nv:]
        self.Lvv = self.Lxx[nv:, nv:]

        if model.withResiduals:
            self.residuals = np.zeros(ncost)
            self.R = np.zeros([ncost, ndx + nu])
            self.Rx = self.R[:, :ndx]
            self.Ru = self.R[:, ndx:]
            self.Rq = self.Rx[:, :nv]
            self.Rv = self.Rx[:, nv:]


class CostModelNumDiff(CostModelPinocchio):
    """ Abstract cost model that uses NumDiff for derivative computation.
    """
    def __init__(self, costModel, State, withGaussApprox=False, reevals=[]):
        '''
        reevals is a list of lambdas of (pinocchiomodel,pinocchiodata,x,u) to be
        reevaluated at each num diff.
        '''
        self.CostDataType = CostDataNumDiff
        CostModelPinocchio.__init__(self, costModel.pinocchio, ncost=costModel.ncost, nu=costModel.nu)
        self.State = State
        self.model0 = costModel
        self.disturbance = np.sqrt(2 * EPS)
        self.withGaussApprox = withGaussApprox
        if withGaussApprox:
            assert (costModel.withResiduals)
        self.reevals = reevals

    def calc(self, data, x, u):
        data.cost = self.model0.calc(data.data0, x, u)
        if self.withGaussApprox:
            data.residuals = data.data0.residuals

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        ndx, nu = self.ndx, self.nu
        h = self.disturbance

        def dist(i, n, h):
            return np.array([h if ii == i else 0 for ii in range(n)])

        def Xint(x, dx):
            return self.State.integrate(x, dx)

        for ix in range(ndx):
            xi = Xint(x, dist(ix, ndx, h))
            [r(self.model0.pinocchio, data.datax[ix].pinocchio, xi, u) for r in self.reevals]
            c = self.model0.calc(data.datax[ix], xi, u)
            data.Lx[ix] = (c - data.data0.cost) / h
            if self.withGaussApprox:
                data.Rx[:, ix] = (data.datax[ix].residuals - data.data0.residuals) / h
        for iu in range(nu):
            ui = u + dist(iu, nu, h)
            [r(self.model0.pinocchio, data.datau[iu].pinocchio, x, ui) for r in self.reevals]
            c = self.model0.calc(data.datau[iu], x, ui)
            data.Lu[iu] = (c - data.data0.cost) / h
            if self.withGaussApprox:
                data.Ru[:, iu] = (data.datau[iu].residuals - data.data0.residuals) / h
        if self.withGaussApprox:
            data.L[:, :] = np.dot(data.R.T, data.R)


class CostDataNumDiff(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        nx, nu = model.nx, model.nu
        self.pinocchio = pinocchioData
        self.data0 = model.model0.createData(pinocchioData)
        self.datax = [model.model0.createData(model.model0.pinocchio.createData()) for i in range(nx)]
        self.datau = [model.model0.createData(model.model0.pinocchio.createData()) for i in range(nu)]


class CostModelSum(CostModelPinocchio):
    # This could be done with a namedtuple but I don't like the read-only labels.
    class CostItem:
        def __init__(self, name, cost, weight):
            self.name = name
            self.cost = cost
            self.weight = weight

        def __str__(self):
            return "CostItem(name=%s, cost=%s, weight=%s)" % (str(self.name), str(self.cost.__class__), str(
                self.weight))

        __repr__ = __str__

    def __init__(self, pinocchioModel, nu=None, withResiduals=True):
        self.CostDataType = CostDataSum
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=0, nu=nu)
        # Preserve task order in evaluation, which is a nicer behavior when debuging.
        self.costs = OrderedDict()

    def addCost(self, name, cost, weight):
        assert (cost.withResiduals and '''
                The cost-of-sums class has not been designed nor tested for non sum of squares
                cost functions. It should not be a big deal to modify it, but this is not done
                yet. ''')
        self.costs.update([[name, self.CostItem(cost=cost, name=name, weight=weight)]])
        self.ncost += cost.ncost

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.costs[key]
        elif isinstance(key, CostModelPinocchio):
            filter = [v for k, v in self.costs.items() if v.cost == key]
            assert (len(filter) == 1 and "The given key is not or not unique in the costs dict. ")
            return filter[0]
        else:
            raise (KeyError("The key should be string or costmodel."))

    def calc(self, data, x, u):
        data.cost = 0
        nr = 0
        for m, d in zip(self.costs.values(), data.costs.values()):
            data.cost += m.weight * m.cost.calc(d, x, u)
            if self.withResiduals:
                data.residuals[nr:nr + m.cost.ncost] = np.sqrt(m.weight) * d.residuals
                nr += m.cost.ncost
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.g.fill(0)
        data.L.fill(0)
        nr = 0
        for m, d in zip(self.costs.values(), data.costs.values()):
            m.cost.calcDiff(d, x, u, recalc=False)
            data.Lx[:] += m.weight * d.Lx
            data.Lu[:] += m.weight * d.Lu
            data.Lxx[:] += m.weight * d.Lxx
            data.Lxu[:] += m.weight * d.Lxu
            data.Luu[:] += m.weight * d.Luu
            if self.withResiduals:
                data.Rx[nr:nr + m.cost.ncost] = np.sqrt(m.weight) * d.Rx
                data.Ru[nr:nr + m.cost.ncost] = np.sqrt(m.weight) * d.Ru
                nr += m.cost.ncost
        return data.cost


class CostDataSum(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.model = model
        self.costs = OrderedDict([[i.name, i.cost.createData(pinocchioData)] for i in model.costs.values()])

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.costs[key]
        elif isinstance(key, CostModelPinocchio):
            filter = [k for k, v in self.model.costs.items() if v.cost == key]
            assert (len(filter) == 1 and "The given key is not or not unique in the costs dict. ")
            return self.costs[filter[0]]
        else:
            raise (KeyError("The key should be string or costmodel."))


class CostModelFrameTranslation(CostModelPinocchio):
    """ Cost model for frame 3d positioning.

    The class proposes a model of a cost function positioning (3d)
    a frame of the robot. Parametrize it with the frame index frameIdx and
    the effector desired position ref.
    """
    def __init__(self, pinocchioModel, frame, ref, nu=None, activation=None):
        self.CostDataType = CostDataFrameTranslation
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=3, nu=nu)
        self.ref = ref
        self.frame = frame
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.residuals = m2a(data.pinocchio.oMf[self.frame].translation) - self.ref
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        nq = self.nq
        pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
        R = data.pinocchio.oMf[self.frame].rotation
        J = R * pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame,
                                           pinocchio.ReferenceFrame.LOCAL)[:3, :]
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Rq[:, :nq] = J
        data.Lq[:] = np.dot(J.T, Ax)
        data.Lqq[:, :] = np.dot(data.Rq.T, Axx * data.Rq)  # J is a matrix, use Rq instead.
        return data.cost


class CostDataFrameTranslation(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.activation = model.activation.createData()
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0


class CostModelFrameVelocity(CostModelPinocchio):
    """ Cost model for frame velocity.

    The class proposes a model of a cost function that penalize the velocity of a given
    end-effector. It assumes that updateFramePlacement and computeForwardKinematicsDerivatives
    have been runned.
    """
    def __init__(self, pinocchioModel, frame, ref=None, nu=None, activation=None):
        self.CostDataType = CostDataFrameVelocity
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=6)
        self.ref = ref if ref is not None else np.zeros(6)
        self.frame = frame
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.residuals[:] = m2a(pinocchio.getFrameVelocity(self.pinocchio, data.pinocchio,
                                                           self.frame).vector) - self.ref
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        dv_dq, dv_dvq = pinocchio.getJointVelocityDerivatives(self.pinocchio, data.pinocchio, data.joint,
                                                              pinocchio.ReferenceFrame.LOCAL)

        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Rq[:, :] = data.fXj * dv_dq
        data.Rv[:, :] = data.fXj * dv_dvq
        data.Lx[:] = np.dot(data.Rx.T, Ax)
        data.Lxx[:, :] = np.dot(data.Rx.T, Axx * data.Rx)
        return data.cost


class CostDataFrameVelocity(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.activation = model.activation.createData()
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0


class CostModelFrameVelocityLinear(CostModelPinocchio):
    """ Cost model for linear frame velocities.

    The class proposes a model of a cost function that penalize the linear velocity of a given
    end-effector. It assumes that updateFramePlacement and computeForwardKinematicsDerivatives
    have been runned.
    """
    def __init__(self, pinocchioModel, frame, ref=None, nu=None, activation=None):
        self.CostDataType = CostDataFrameVelocityLinear
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=3)
        self.ref = ref if ref is not None else np.zeros(3)
        self.frame = frame
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.residuals[:] = m2a(pinocchio.getFrameVelocity(self.pinocchio, data.pinocchio,
                                                           self.frame).linear) - self.ref
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        dv_dq, dv_dvq = pinocchio.getJointVelocityDerivatives(self.pinocchio, data.pinocchio, data.joint,
                                                              pinocchio.ReferenceFrame.LOCAL)

        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Rq[:, :] = (data.fXj * dv_dq)[:3, :]
        data.Rv[:, :] = (data.fXj * dv_dvq)[:3, :]
        data.Lx[:] = np.dot(data.Rx.T, Ax)
        data.Lxx[:, :] = np.dot(data.Rx.T, Axx * data.Rx)
        return data.cost


class CostDataFrameVelocityLinear(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.activation = model.activation.createData()
        frame = model.pinocchio.frames[model.frame]
        self.joint = frame.parent
        self.jMf = frame.placement
        self.fXj = self.jMf.inverse().action
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0


class CostModelFramePlacement(CostModelPinocchio):
    """ Cost model for SE(3) frame positioning.

    The class proposes a model of a cost function position and orientation (6d)
    for a frame of the robot. Parametrize it with the frame index frameIdx and
    the effector desired pinocchio::SE3 ref.
    """
    def __init__(self, pinocchioModel, frame, ref, nu=None, activation=None):
        self.CostDataType = CostDataFramePlacement
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=6, nu=nu)
        self.ref = ref
        self.frame = frame
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.rMf = self.ref.inverse() * data.pinocchio.oMf[self.frame]
        data.residuals[:] = m2a(pinocchio.log(data.rMf).vector)
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        nq = self.nq
        pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
        J = np.dot(
            pinocchio.Jlog6(data.rMf),
            pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame, pinocchio.ReferenceFrame.LOCAL))
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Rq[:, :nq] = J
        data.Lq[:] = np.dot(J.T, Ax)
        data.Lqq[:, :] = np.dot(data.Rq.T, Axx * data.Rq)  # J is a matrix, use Rq instead.
        return data.cost


class CostDataFramePlacement(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.activation = model.activation.createData()
        self.rMf = None
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0


class CostModelFrameRotation(CostModelPinocchio):
    """ Cost model for frame rotation.

    The class proposes a model of a cost function orientation (3d) for a frame of the robot.
    Parametrize it with the frame index frameIdx and the effector desired rotation matrix.
    """
    def __init__(self, pinocchioModel, frame, ref, nu=None, activation=None):
        self.CostDataType = CostDataFrameRotation
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=3, nu=nu)
        self.ref = ref
        self.frame = frame
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.rRf = self.ref.transpose() * data.pinocchio.oMf[self.frame].rotation
        data.residuals[:] = m2a(pinocchio.log3(data.rRf))
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        nq = self.nq
        pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
        J = np.dot(
            pinocchio.Jlog3(data.rRf),
            pinocchio.getFrameJacobian(self.pinocchio, data.pinocchio, self.frame,
                                       pinocchio.ReferenceFrame.LOCAL)[3:, :])
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Rq[:, :nq] = J
        data.Lq[:] = np.dot(J.T, Ax)
        data.Lqq[:, :] = np.dot(data.Rq.T, Axx * data.Rq)  # J is a matrix, use Rq instead.
        return data.cost


class CostDataFrameRotation(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.activation = model.activation.createData()
        self.rRf = None
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0


class CostModelCoM(CostModelPinocchio):
    """ Cost model for CoM positioning.

    The class proposes a model of a cost function CoM. It is parametrized with the
    desired CoM position.
    """
    def __init__(self, pinocchioModel, ref, nu=None, activation=None):
        self.CostDataType = CostDataCoM
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=3, nu=nu)
        self.ref = ref
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.residuals = m2a(data.pinocchio.com[0]) - self.ref
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        nq = self.nq
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        J = data.pinocchio.Jcom
        data.Rq[:, :nq] = J
        data.Lq[:] = np.dot(J.T, Ax)
        data.Lqq[:, :] = np.dot(data.Rq.T, Axx * data.Rq)  # J is a matrix, use Rq instead.
        return data.cost


class CostDataCoM(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.activation = model.activation.createData()
        self.Lu = 0
        self.Lv = 0
        self.Lxu = 0
        self.Luu = 0
        self.Lvv = 0
        self.Ru = 0
        self.Rv = 0


class CostModelState(CostModelPinocchio):
    """ Cost model for state.

    It tracks a reference state vector. Generally speaking, the state error lie in the
    tangent-space of the state manifold (or more precisely the configuration manifold).
    """
    def __init__(self, pinocchioModel, State, ref=None, nu=None, activation=None):
        self.CostDataType = CostDataState
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=State.ndx, nu=nu)
        self.State = State
        self.ref = ref if ref is not None else State.zero()
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.residuals[:] = self.State.diff(self.ref, x)
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.Rx[:, :] = (self.State.Jdiff(self.ref, x, 'second').T).T
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Lx[:] = np.dot(data.Rx.T, Ax)
        data.Lxx[:, :] = np.dot(data.Rx.T, Axx * data.Rx)


class CostDataState(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.activation = model.activation.createData()
        self.Lu = 0
        self.Lxu = 0
        self.Luu = 0
        self.Ru = 0


class CostModelControl(CostModelPinocchio):
    """ Cost model for control.

    It tracks a reference control vector.
    """
    def __init__(self, pinocchioModel, nu=None, ref=None, activation=None):
        self.CostDataType = CostDataControl
        nu = nu if nu is not None else pinocchioModel.nv
        if ref is not None:
            assert (ref.shape == (nu, ))
        CostModelPinocchio.__init__(self, pinocchioModel, nu=nu, ncost=nu)
        self.ref = ref
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        data.residuals[:] = u if self.ref is None else u - self.ref
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        # data.Ru[:,:] = np.eye(nu)
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Lu[:] = Ax
        data.Luu[:, :] = np.diag(m2a(Axx))


class CostDataControl(CostDataPinocchio):
    def __init__(self, model, pinocchioData):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        nu = model.nu
        self.activation = model.activation.createData()
        self.Lx = 0
        self.Lxx = 0
        self.Lxu = 0
        self.Rx = 0
        self.Luu[:, :] = np.eye(nu)
        self.Ru[:, :] = self.Luu


class CostModelForce(CostModelPinocchio):
    """ Cost model for 6D forces (wrench).

    The class proposes a model of a cost function for tracking a reference
    value of a 6D force, being given the contact model and its derivatives.
    """
    def __init__(self, pinocchioModel, contactModel, ncost=6, ref=None, nu=None, activation=None):
        self.CostDataType = CostDataForce
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=ncost, nu=nu)
        self.ref = ref if ref is not None else np.zeros(ncost)
        self.contact = contactModel
        self.activation = activation if activation is not None else ActivationModelQuad()

    def calc(self, data, x, u):
        if data.contact is None:
            raise RuntimeError('''The CostForce data should be specifically initialized from the
            contact data ... no automatic way of doing that yet ...''')
        data.f = data.contact.f
        data.residuals = data.f - self.ref
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        assert (self.nu == len(u) and self.contact.nu == self.nu)
        df_dx, df_du = data.contact.df_dx, data.contact.df_du
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals, recalc=recalc)
        data.Rx[:, :] = df_dx  # This is useless.
        data.Ru[:, :] = df_du  # This is useless

        data.Lx[:] = np.dot(df_dx.T, Ax)
        data.Lu[:] = np.dot(df_du.T, Ax)

        data.Lxx[:, :] = np.dot(df_dx.T, Axx * df_dx)
        data.Lxu[:, :] = np.dot(df_dx.T, Axx * df_du)
        data.Luu[:, :] = np.dot(df_du.T, Axx * df_du)

        return data.cost


class CostDataForce(CostDataPinocchio):
    def __init__(self, model, pinocchioData, contactData=None):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.contact = contactData
        self.activation = model.activation.createData()


class CostModelForceLinearCone(CostModelPinocchio):
    """
    The class proposes a model which implements Af-ref<=0 for the linear conic cost. (The inequality
    is implemented by the activation function. By default ref is zero (Af<=0).
    """
    def __init__(self, pinocchioModel, contactModel, A, ref=None, nu=None, activation=None):
        self.CostDataType = CostDataForce
        CostModelPinocchio.__init__(self, pinocchioModel, ncost=A.shape[0], nu=nu)
        self.A = A
        self.nfaces = A.shape[0]
        self.ref = ref if ref is not None else np.zeros(self.nfaces)
        self.contact = contactModel
        assert (activation is None and "defining your own activation model is not possible here.")
        self.activation = ActivationModelInequality(np.array([-np.inf] * self.nfaces), np.zeros(self.nfaces))

    def calc(self, data, x, u):
        if data.contact is None:
            raise RuntimeError('''The CostForce data should be specifically initialized from the
            contact data ... no automatic way of doing that yet ...''')
        data.f = data.contact.f
        data.residuals = np.dot(self.A, data.f) - self.ref
        data.cost = sum(self.activation.calc(data.activation, data.residuals))
        return data.cost

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        assert (self.nu == len(u) and self.contact.nu == self.nu)
        df_dx, df_du = data.contact.df_dx, data.contact.df_du
        Ax, Axx = self.activation.calcDiff(data.activation, data.residuals)
        sel = Axx.astype(bool)[:, 0]
        A = self.A[sel, :]
        A2 = np.dot(A.T, A)

        data.Rx[:, :] = np.dot(self.A, df_dx)
        data.Ru[:, :] = np.dot(self.A, df_du)

        data.Lx[:] = np.dot(data.Rx[sel, :].T, Ax[sel])
        data.Lu[:] = np.dot(data.Ru[sel, :].T, Ax[sel])

        data.Lxx[:, :] = np.dot(df_dx.T, np.dot(A2, df_dx))
        data.Lxu[:, :] = np.dot(df_dx.T, np.dot(A2, df_du))
        data.Luu[:, :] = np.dot(df_du.T, np.dot(A2, df_du))

        return data.cost


class CostDataForceCone(CostDataPinocchio):
    def __init__(self, model, pinocchioData, contactData=None):
        CostDataPinocchio.__init__(self, model, pinocchioData)
        self.contact = contactData
        self.activation = model.activation.createData()
