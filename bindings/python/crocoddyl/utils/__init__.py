import crocoddyl
import pinocchio
import numpy as np
import scipy.linalg as scl

crocoddyl.switchToNumpyMatrix()


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def absmax(A):
    return np.max(abs(A))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class StateVectorDerived(crocoddyl.StateAbstract):
    def __init__(self, nx):
        crocoddyl.StateAbstract.__init__(self, nx, nx)

    def zero(self):
        return np.matrix(np.zeros(self.nx)).T

    def rand(self):
        return np.matrix(np.random.rand(self.nx)).T

    def diff(self, x0, x1):
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


class StateMultibodyDerived(crocoddyl.StateAbstract):
    def __init__(self, pinocchioModel):
        crocoddyl.StateAbstract.__init__(self, pinocchioModel.nq + pinocchioModel.nv, 2 * pinocchioModel.nv)
        self.model = pinocchioModel

    def zero(self):
        q = pinocchio.neutral(self.model)
        v = pinocchio.utils.zero(self.nv)
        return np.concatenate([q, v])

    def rand(self):
        q = pinocchio.randomConfiguration(self.model)
        v = pinocchio.utils.rand(self.nv)
        return np.concatenate([q, v])

    def diff(self, x0, x1):
        q0 = x0[:self.nq]
        q1 = x1[:self.nq]
        v0 = x0[-self.nv:]
        v1 = x1[-self.nv:]
        dq = pinocchio.difference(self.model, q0, q1)
        return np.concatenate([dq, v1 - v0])

    def integrate(self, x, dx):
        q = x[:self.nq]
        v = x[-self.nv:]
        dq = dx[:self.nv]
        dv = dx[-self.nv:]
        qn = pinocchio.integrate(self.model, q, dq)
        return np.concatenate([qn, v + dv])

    def Jdiff(self, x1, x2, firstsecond='both'):
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jdiff(x1, x2, 'first'), self.Jdiff(x1, x2, 'second')]

        if firstsecond == 'first':
            dx = self.diff(x2, x1)
            q = x2[:self.model.nq]
            dq = dx[:self.model.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))
        elif firstsecond == 'second':
            dx = self.diff(x1, x2)
            q = x1[:self.nq]
            dq = dx[:self.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))

    def Jintegrate(self, x, dx, firstsecond='both'):
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jintegrate(x, dx, 'first'), self.Jintegrate(x, dx, 'second')]

        q = x[:self.nq]
        dq = dx[:self.nv]
        Jq, Jdq = pinocchio.dIntegrate(self.model, q, dq)
        if firstsecond == 'first':
            return np.matrix(scl.block_diag(np.linalg.inv(Jq), np.eye(self.nv)))
        elif firstsecond == 'second':
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))


class FreeFloatingActuationDerived(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        assert (state.pinocchio.joints[1].shortname() == 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv - 6)

    def calc(self, data, x, u):
        data.tau = np.vstack([pinocchio.utils.zero(6), u])

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        dtau_du = np.vstack([pinocchio.utils.zero((6, self.nu)), pinocchio.utils.eye(self.nu)])
        data.dtau_du = dtau_du


class FullActuationDerived(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        assert (state.pinocchio.joints[1].shortname() != 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv)

    def calc(self, data, x, u):
        data.tau = u

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.dtau_du = pinocchio.utils.eye(self.nu)


class UnicycleDerived(crocoddyl.ActionModelAbstract):
    def __init__(self):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
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
        data.r = np.vstack([self.costWeights[0] * x, self.costWeights[1] * u])
        data.cost = .5 * sum(m2a(data.r)**2)

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
        v, w = m2a(u)
        px, py, theta = m2a(x)
        # Cost derivatives
        data.Lx = a2m(m2a(x) * ([self.costWeights[0]**2] * self.state.nx))
        data.Lu = a2m(m2a(u) * ([self.costWeights[1]**2] * self.nu))
        data.Lxx = np.diag([self.costWeights[0]**2] * self.state.nx)
        data.Luu = np.diag([self.costWeights[1]**2] * self.nu)
        # Dynamic derivatives
        c, s, dt = np.cos(theta), np.sin(theta), self.dt
        v, w = m2a(u)
        data.Fx = np.matrix([[1, 0, -s * v * dt], [0, 1, c * v * dt], [0, 0, 1]])
        data.Fu = np.matrix([[c * self.dt, 0], [s * self.dt, 0], [0, self.dt]])


class LQRDerived(crocoddyl.ActionModelAbstract):
    def __init__(self, nx, nu, driftFree=True):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(nx), nu)

        self.Fx = np.matrix(np.eye(self.state.nx))
        self.Fu = np.matrix(np.eye(self.state.nx))[:, :self.nu]
        self.f0 = np.matrix(np.zeros(self.state.nx)).T
        self.Lxx = np.matrix(np.eye(self.state.nx))
        self.Lxu = np.matrix(np.eye(self.state.nx))[:, :self.nu]
        self.Luu = np.matrix(np.eye(self.nu))
        self.lx = np.matrix(np.ones(self.state.nx)).T
        self.lu = np.matrix(np.ones(self.nu)).T

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        data.xnext = self.Fx * x + self.Fu * u + self.f0
        data.cost = 0.5 * np.asscalar(x.T * self.Lxx * x) + 0.5 * np.asscalar(u.T * self.Luu * u)
        data.cost += np.asscalar(x.T * self.Lxu * u) + np.asscalar(self.lx.T * x) + np.asscalar(self.lu.T * u)

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
        data.Lx = self.lx + np.dot(self.Lxx, x) + np.dot(self.Lxu, u)
        data.Lu = self.lu + np.dot(self.Lxu.T, x) + np.dot(self.Luu, u)
        data.Fx = self.Fx
        data.Fu = self.Fu
        data.Lxx = self.Lxx
        data.Luu = self.Luu
        data.Lxu = self.Lxu


class DifferentialLQRDerived(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, nq, nu, driftFree=True):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(2 * nq), nu)

        self.Fq = np.matrix(np.eye(self.state.nq))
        self.Fv = np.matrix(np.eye(self.state.nv))
        self.Fu = np.matrix(np.eye(self.state.nq))[:, :self.nu]
        self.f0 = np.matrix(np.zeros(self.state.nv)).T
        self.Lxx = np.matrix(np.eye(self.state.nx))
        self.Lxu = np.matrix(np.eye(self.state.nx))[:, :self.nu]
        self.Luu = np.matrix(np.eye(self.nu))
        self.lx = np.matrix(np.ones(self.state.nx)).T
        self.lu = np.matrix(np.ones(self.nu)).T

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[:self.state.nq], x[self.state.nq:]
        data.xout = self.Fq * q + self.Fv * v + self.Fu * u + self.f0
        data.cost = 0.5 * np.asscalar(x.T * self.Lxx * x) + 0.5 * np.asscalar(u.T * self.Luu * u)
        data.cost += np.asscalar(x.T * self.Lxu * u) + np.asscalar(self.lx.T * x) + np.asscalar(self.lu.T * u)

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
        data.Lx = self.lx + np.dot(self.Lxx, x) + np.dot(self.Lxu, u)
        data.Lu = self.lu + np.dot(self.Lxu.T, x) + np.dot(self.Luu, u)
        data.Fx = np.hstack([self.Fq, self.Fv])
        data.Fu = self.Fu
        data.Lxx = self.Lxx
        data.Luu = self.Luu
        data.Lxu = self.Lxu


class DifferentialFreeFwdDynamicsDerived(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu, costModel.nr)
        self.actuation = actuationModel
        self.costs = costModel
        self.enable_force = True
        self.armature = np.matrix(np.zeros(0))

        # We cannot abstract data in Python bindings, let's create this internal data inside model
        self.pinocchioData = pinocchio.Data(self.state.pinocchio)
        self.multibodyData = crocoddyl.DataCollectorMultibody(self.pinocchioData)
        self.actuationData = self.actuation.createData()
        self.costsData = self.costs.createData(self.multibodyData)
        self.Minv = None

    def calc(self, data, x, u=None):
        self.costsData.shareMemory(data)
        if u is None:
            u = self.unone
        q, v = x[:self.state.nq], x[-self.state.nv:]
        self.actuation.calc(self.actuationData, x, u)
        tau = self.actuationData.tau

        # Computing the dynamics using ABA or manually for armature case
        if self.enable_force:
            data.xout = pinocchio.aba(self.state.pinocchio, self.pinocchioData, q, v, tau)
        else:
            pinocchio.computeAllTerms(self.state.pinocchio, self.pinocchioData, q, v)
            data.M = self.pinocchioData.M
            if self.armature.size == self.state.nv:
                data.M[range(self.state.nv), range(self.state.nv)] += self.armature
            self.Minv = np.linalg.inv(data.M)
            data.xout = self.Minv * (tau - self.pinocchioData.nle)

        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.state.pinocchio, self.pinocchioData, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, self.pinocchioData)
        self.costs.calc(self.costsData, x, u)
        data.cost = self.costsData.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        self.costsData.shareMemory(data)
        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[-nv:]
        self.actuation.calcDiff(self.actuationData, x, u)
        tau = self.actuationData.tau

        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
            pinocchio.computeJointJacobians(self.state.pinocchio, self.pinocchioData, q)
        # Computing the dynamics derivatives
        if self.enable_force:
            pinocchio.computeABADerivatives(self.state.pinocchio, self.pinocchioData, q, v, tau)
            ddq_dq = self.pinocchioData.ddq_dq
            ddq_dv = self.pinocchioData.ddq_dv
            data.Fx = np.hstack([ddq_dq, ddq_dv]) + self.pinocchioData.Minv * self.actuationData.dtau_dx
            data.Fu = self.pinocchioData.Minv * self.actuationData.dtau_du
        else:
            pinocchio.computeRNEADerivatives(self.state.pinocchio, self.pinocchioData, q, v, data.xout)
            ddq_dq = self.Minv * (self.actuationData.dtau_dx[:, :nv] - self.pinocchioData.dtau_dq)
            ddq_dv = self.Minv * (self.actuationData.dtau_dx[:, nv:] - self.pinocchioData.dtau_dv)
            data.Fx = np.hstack([ddq_dq, ddq_dv])
            data.Fu = self.Minv * self.actuationData.dtau_du
        # Computing the cost derivatives
        self.costs.calcDiff(self.costsData, x, u, False)

    def set_armature(self, armature):
        if armature.size is not self.state.nv:
            print('The armature dimension is wrong, we cannot set it.')
        else:
            self.enable_force = False
            self.armature = armature.T


class IntegratedActionModelEuler(crocoddyl.ActionModelAbstract):
    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True):
        crocoddyl.ActionModelAbstract.__init__(self, diffModel.state, diffModel.nv, diffModel.nr)
        self.differential = diffModel
        self.withCostResiduals = withCostResiduals
        self.timeStep = timeStep

    def calc(self, data, x, u=None):
        nq, dt = self.state.nq, self.timeStep
        acc, cost = self.differential.calc(data.differential, x, u)
        if self.withCostResiduals:
            data.r = data.differential.r
        data.cost = cost
        # data.xnext[nq:] = x[nq:] + acc*dt
        # data.xnext[:nq] = pinocchio.integrate(self.differential.pinocchio,
        #                                       a2m(x[:nq]),a2m(data.xnext[nq:]*dt)).flat
        data.dx = np.concatenate([x[nq:] * dt + acc * dt**2, acc * dt])
        data.xnext[:] = self.differential.state.integrate(x, data.dx)

        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        nv, dt = self.state.nv, self.timeStep
        if recalc:
            self.calc(data, x, u)
        self.differential.calcDiff(data.differential, x, u, recalc=False)
        dxnext_dx, dxnext_ddx = self.state.Jintegrate(x, data.dx)
        da_dx, da_du = data.differential.Fx, data.differential.Fu
        ddx_dx = np.vstack([da_dx * dt, da_dx])
        ddx_dx[range(nv), range(nv, 2 * nv)] += 1
        data.Fx[:, :] = dxnext_dx + dt * np.dot(dxnext_ddx, ddx_dx)
        ddx_du = np.vstack([da_du * dt, da_du])
        data.Fu[:, :] = dt * np.dot(dxnext_ddx, ddx_du)
        data.Lx[:] = data.differential.Lx
        data.Lu[:] = data.differential.Lu
        data.Lxx[:] = data.differential.Lxx
        data.Lxu[:] = data.differential.Lxu
        data.Luu[:] = data.differential.Luu


class StateCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, xref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        self.xref = xref if xref is not None else state.zero()
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u):
        data.r = self.state.diff(self.xref, x)
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.Rx = self.state.Jdiff(self.xref, x, 'second')[0]
        self.activation.calcDiff(data.activation, data.r, recalc)
        data.Lx = data.Rx.T * data.activation.Ar
        data.Lxx = data.Rx.T * data.activation.Arr * data.Rx


class ControlCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, uref=None, nu=None):
        nu = nu if nu is not None else state.nv
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(nu)
        self.uref = uref if uref is not None else pinocchio.utils.zero(nu)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u):
        data.r = u - self.uref
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        self.activation.calcDiff(data.activation, data.r, recalc)
        data.Lu = data.activation.Ar
        data.Luu = data.activation.Arr


class CoMPositionCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, cref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.cref = cref

    def calc(self, data, x, u):
        data.r = data.shared.pinocchio.com[0] - self.cref
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        self.activation.calcDiff(data.activation, data.r, recalc)
        data.Rx = np.hstack([data.shared.pinocchio.Jcom, pinocchio.utils.zero((self.activation.nr, self.state.nv))])
        data.Lx = np.vstack(
            [data.shared.pinocchio.Jcom.T * data.activation.Ar,
             pinocchio.utils.zero((self.state.nv, 1))])
        data.Lxx = np.vstack([
            np.hstack([
                data.shared.pinocchio.Jcom.T * data.activation.Arr * data.shared.pinocchio.Jcom,
                pinocchio.utils.zero((self.state.nv, self.state.nv))
            ]),
            pinocchio.utils.zero((self.state.nv, self.state.ndx))
        ])


class FramePlacementCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, Mref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.Mref = Mref

    def calc(self, data, x, u):
        data.rMf = self.Mref.oMf.inverse() * data.shared.pinocchio.oMf[self.Mref.frame]
        data.r = pinocchio.log(data.rMf).vector
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf = pinocchio.Jlog6(data.rMf)
        data.fJf = pinocchio.getFrameJacobian(self.state.pinocchio, data.shared.pinocchio, self.Mref.frame,
                                              pinocchio.ReferenceFrame.LOCAL)
        data.J = data.rJf * data.fJf
        self.activation.calcDiff(data.activation, data.r, recalc)
        data.Rx = np.hstack([data.J, pinocchio.utils.zero((self.activation.nr, self.state.nv))])
        data.Lx = np.vstack([data.J.T * data.activation.Ar, pinocchio.utils.zero((self.state.nv, 1))])
        data.Lxx = np.vstack([
            np.hstack([data.J.T * data.activation.Arr * data.J,
                       pinocchio.utils.zero((self.state.nv, self.state.nv))]),
            pinocchio.utils.zero((self.state.nv, self.state.ndx))
        ])


class FrameTranslationCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, xref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.xref = xref

    def calc(self, data, x, u):
        data.r = data.shared.pinocchio.oMf[self.xref.frame].translation - self.xref.oxf
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.R = data.shared.pinocchio.oMf[self.xref.frame].rotation
        data.J = data.R * pinocchio.getFrameJacobian(self.state.pinocchio, data.shared.pinocchio, self.xref.frame,
                                                     pinocchio.ReferenceFrame.LOCAL)[:3, :]
        self.activation.calcDiff(data.activation, data.r, recalc)
        data.Rx = np.hstack([data.J, pinocchio.utils.zero((self.activation.nr, self.state.nv))])
        data.Lx = np.vstack([data.J.T * data.activation.Ar, pinocchio.utils.zero((self.state.nv, 1))])
        data.Lxx = np.vstack([
            np.hstack([data.J.T * data.activation.Arr * data.J,
                       pinocchio.utils.zero((self.state.nv, self.state.nv))]),
            pinocchio.utils.zero((self.state.nv, self.state.ndx))
        ])


class FrameRotationCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, Rref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.Rref = Rref

    def calc(self, data, x, u):
        data.rRf = self.Rref.oRf.transpose() * data.shared.pinocchio.oMf[self.Rref.frame].rotation
        data.r = pinocchio.log3(data.rRf)
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf = pinocchio.Jlog3(data.rRf)
        data.fJf = pinocchio.getFrameJacobian(self.state.pinocchio, data.shared.pinocchio, self.Rref.frame,
                                              pinocchio.ReferenceFrame.LOCAL)[:3, :]
        data.J = data.rJf * data.fJf
        self.activation.calcDiff(data.activation, data.r, recalc)
        data.Rx = np.hstack([data.J, pinocchio.utils.zero((self.activation.nr, self.state.nv))])
        data.Lx = np.vstack([data.J.T * data.activation.Ar, pinocchio.utils.zero((self.state.nv, 1))])
        data.Lxx = np.vstack([
            np.hstack([data.J.T * data.activation.Arr * data.J,
                       pinocchio.utils.zero((self.state.nv, self.state.nv))]),
            pinocchio.utils.zero((self.state.nv, self.state.ndx))
        ])


class FrameVelocityCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, vref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.vref = vref
        self.joint = state.pinocchio.frames[vref.frame].parent
        self.fXj = state.pinocchio.frames[vref.frame].placement.inverse().action

    def calc(self, data, x, u):
        data.r = (pinocchio.getFrameVelocity(self.state.pinocchio, data.shared.pinocchio, self.vref.frame) -
                  self.vref.oMf).vector
        self.activation.calc(data.activation, data.r)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(self.state.pinocchio, data.shared.pinocchio,
                                                                           self.joint, pinocchio.ReferenceFrame.LOCAL)

        self.activation.calcDiff(data.activation, data.r, recalc)
        data.Rx = np.hstack([self.fXj * v_partial_dq, self.fXj * v_partial_dv])
        data.Lx = data.Rx.T * data.activation.Ar
        data.Lxx = data.Rx.T * data.activation.Arr * data.Rx


class Contact3DDerived(crocoddyl.ContactModelAbstract):
    def __init__(self, state, xref, gains=[0., 0.]):
        crocoddyl.ContactModelAbstract.__init__(self, state, 3)
        self.xref = xref
        self.gains = gains
        self.joint = state.pinocchio.frames[xref.frame].parent
        self.fXj = state.pinocchio.frames[xref.frame].placement.inverse().action
        v = pinocchio.Motion().Zero()
        self.vw = v.angular
        self.vv = v.linear
        self.Jw = pinocchio.utils.zero((3, state.pinocchio.nv))

    def calc(self, data, x):
        assert (self.xref.oxf is not None or self.gains[0] == 0.)
        v = pinocchio.getFrameVelocity(self.state.pinocchio, data.pinocchio, self.xref.frame)
        self.vw = v.angular
        self.vv = v.linear

        fJf = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.xref.frame,
                                         pinocchio.ReferenceFrame.LOCAL)
        data.Jc = fJf[:3, :]
        self.Jw = fJf[3:, :]

        data.a0 = pinocchio.getFrameAcceleration(self.state.pinocchio, data.pinocchio,
                                                 self.xref.frame).linear + pinocchio.utils.cross(self.vw, self.vv)
        if self.gains[0] != 0.:
            data.a0 += np.asscalar(self.gains[0]) * (data.pinocchio.oMf[self.xref.frame].translation - self.xref.oxf)
        if self.gains[1] != 0.:
            data.a0 += np.asscalar(self.gains[1]) * self.vv

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, self.joint, pinocchio.ReferenceFrame.LOCAL)

        vv_skew = pinocchio.utils.skew(self.vv)
        vw_skew = pinocchio.utils.skew(self.vw)
        fXjdv_dq = self.fXj * v_partial_dq
        da0_dq = (self.fXj * a_partial_dq)[:3, :] + vw_skew * fXjdv_dq[:3, :] - vv_skew * fXjdv_dq[3:, :]
        da0_dv = (self.fXj * a_partial_dv)[:3, :] + vw_skew * data.Jc - vv_skew * self.Jw

        if np.asscalar(self.gains[0]) != 0.:
            R = data.pinocchio.oMf[self.xref.frame].rotation
            da0_dq += np.asscalar(self.gains[0]) * R * pinocchio.getFrameJacobian(
                self.state.pinocchio, data.pinocchio, self.xref.frame, pinocchio.ReferenceFrame.LOCAL)[:3, :]
        if np.asscalar(self.gains[1]) != 0.:
            da0_dq += np.asscalar(self.gains[1]) * (self.fXj[:3, :] * v_partial_dq)
            da0_dv += np.asscalar(self.gains[1]) * (self.fXj[:3, :] * a_partial_da)
        data.da0_dx = np.hstack([da0_dq, da0_dv])


class Contact6DDerived(crocoddyl.ContactModelAbstract):
    def __init__(self, state, Mref, gains=[0., 0.]):
        crocoddyl.ContactModelAbstract.__init__(self, state, 6)
        self.Mref = Mref
        self.gains = gains
        self.joint = state.pinocchio.frames[Mref.frame].parent
        self.fXj = state.pinocchio.frames[Mref.frame].placement.inverse().action

    def calc(self, data, x):
        assert (self.Mref.oMf is not None or self.gains[0] == 0.)
        data.Jc = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.Mref.frame,
                                             pinocchio.ReferenceFrame.LOCAL)
        data.a0 = pinocchio.getFrameAcceleration(self.state.pinocchio, data.pinocchio, self.Mref.frame).vector
        if self.gains[0] != 0.:
            self.rMf = self.Mref.oMf.inverse() * data.pinocchio.oMf[self.Mref.frame]
            data.a0 += np.asscalar(self.gains[0]) * pinocchio.log6(self.rMf).vector
        if self.gains[1] != 0.:
            v = pinocchio.getFrameVelocity(self.state.pinocchio, data.pinocchio, self.Mref.frame).vector
            data.a0 += np.asscalar(self.gains[1]) * v

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, self.joint, pinocchio.ReferenceFrame.LOCAL)

        da0_dq = (self.fXj * a_partial_dq)
        da0_dv = (self.fXj * a_partial_dv)

        if np.asscalar(self.gains[0]) != 0.:
            da0_dq += np.asscalar(self.gains[0]) * pinocchio.Jlog6(self.rMf) * data.Jc
        if np.asscalar(self.gains[1]) != 0.:
            da0_dq += np.asscalar(self.gains[1]) * (self.fXj * v_partial_dq)
            da0_dv += np.asscalar(self.gains[1]) * (self.fXj * a_partial_da)
        data.da0_dx = np.hstack([da0_dq, da0_dv])


class Impulse3DDerived(crocoddyl.ImpulseModelAbstract):
    def __init__(self, state, frame):
        crocoddyl.ImpulseModelAbstract.__init__(self, state, 3)
        self.frame = frame
        self.joint = state.pinocchio.frames[frame].parent
        self.fXj = state.pinocchio.frames[frame].placement.inverse().action

    def calc(self, data, x):
        data.Jc = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.frame,
                                             pinocchio.ReferenceFrame.LOCAL)[:3, :]

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(self.state.pinocchio, data.pinocchio,
                                                                           self.joint, pinocchio.ReferenceFrame.LOCAL)
        data.dv0_dq = self.fXj[:3, :] * v_partial_dq


class Impulse6DDerived(crocoddyl.ImpulseModelAbstract):
    def __init__(self, state, frame):
        crocoddyl.ImpulseModelAbstract.__init__(self, state, 6)
        self.frame = frame
        self.joint = state.pinocchio.frames[frame].parent
        self.fXj = state.pinocchio.frames[frame].placement.inverse().action

    def calc(self, data, x):
        data.Jc = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.frame,
                                             pinocchio.ReferenceFrame.LOCAL)

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(self.state.pinocchio, data.pinocchio,
                                                                           self.joint, pinocchio.ReferenceFrame.LOCAL)
        data.dv0_dq = self.fXj * v_partial_dq


class DDPDerived(crocoddyl.SolverAbstract):
    def __init__(self, shootingProblem):
        crocoddyl.SolverAbstract.__init__(self, shootingProblem)
        self.allocateData()  # TODO remove it?

        self.isFeasible = False
        self.alphas = [2**(-n) for n in range(10)]
        self.th_grad = 1e-12

        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5

    def calc(self):
        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.gaps[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.x0)
            for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
                self.gaps[i + 1] = m.state.diff(x, d.xnext)

        return self.cost

    def computeDirection(self, recalc=True):
        if recalc:
            self.calc()
        self.backwardPass()
        return [np.nan] * (self.problem.T + 1), self.k, self.Vx

    def stoppingCriteria(self):
        return sum([np.asscalar(q.T * q) for q in self.Qu])

    def expectedImprovement(self):
        d1 = sum([np.asscalar(q.T * k) for q, k in zip(self.Qu, self.k)])
        d2 = sum([-np.asscalar(k.T * q * k) for q, k in zip(self.Quu, self.k)])
        return np.matrix([d1, d2]).T

    def tryStep(self, stepLength=1):
        self.forwardPass(stepLength)
        return self.cost - self.cost_try

    def solve(self, init_xs=[], init_us=[], maxiter=100, isFeasible=False, regInit=None):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        self.wasFeasible = False
        for i in range(maxiter):
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    recalc = False
                    self.increaseRegularization()
                    if self.x_reg == self.regMax:
                        return self.xs, self.us, False
                    else:
                        continue
                break
            d = self.expectedImprovement()
            d1, d2 = np.asscalar(d[0]), np.asscalar(d[1])

            for a in self.alphas:
                try:
                    self.dV = self.tryStep(a)
                except ArithmeticError:
                    continue
                self.dV_exp = a * (d1 + .5 * d2 * a)
                if self.dV_exp >= 0:
                    if d1 < self.th_grad or not self.isFeasible or self.dV > self.th_acceptStep * self.dV_exp:
                        # Accept step
                        self.wasFeasible = self.isFeasible
                        self.setCandidate(self.xs_try, self.us_try, True)
                        self.cost = self.cost_try
                        break
            if a > self.th_step:
                self.decreaseRegularization()
            if a == self.alphas[-1]:
                self.increaseRegularization()
                if self.x_reg == self.regMax:
                    return self.xs, self.us, False
            self.stepLength = a
            self.iter = i
            self.stop = self.stoppingCriteria()
            # TODO @Carlos bind the callbacks
            # if self.callback is not None:
            #     [c(self) for c in self.callback]

            if self.wasFeasible and self.stop < self.th_stop:
                return self.xs, self.us, True
        return self.xs, self.us, False

    def increaseRegularization(self):
        self.x_reg *= self.regFactor
        if self.x_reg > self.regMax:
            self.x_reg = self.regMax
        self.u_reg = self.x_reg

    def decreaseRegularization(self):
        self.x_reg /= self.regFactor
        if self.x_reg < self.regMin:
            self.x_reg = self.regMin
        self.u_reg = self.x_reg

    def allocateData(self):
        self.Vxx = [a2m(np.zeros([m.state.ndx, m.state.ndx])) for m in self.models()]
        self.Vx = [a2m(np.zeros([m.state.ndx])) for m in self.models()]

        self.Q = [a2m(np.zeros([m.state.ndx + m.nu, m.state.ndx + m.nu])) for m in self.problem.runningModels]
        self.q = [a2m(np.zeros([m.state.ndx + m.nu])) for m in self.problem.runningModels]
        self.Qxx = [Q[:m.state.ndx, :m.state.ndx] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qxu = [Q[:m.state.ndx, m.state.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qux = [Qxu.T for m, Qxu in zip(self.problem.runningModels, self.Qxu)]
        self.Quu = [Q[m.state.ndx:, m.state.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qx = [q[:m.state.ndx] for m, q in zip(self.problem.runningModels, self.q)]
        self.Qu = [q[m.state.ndx:] for m, q in zip(self.problem.runningModels, self.q)]

        self.K = [np.matrix(np.zeros([m.nu, m.state.ndx])) for m in self.problem.runningModels]
        self.k = [a2m(np.zeros([m.nu])) for m in self.problem.runningModels]

        self.xs_try = [self.problem.x0] + [np.nan * self.problem.x0] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        self.gaps = [a2m(np.zeros(self.problem.runningModels[0].state.ndx))
                     ] + [a2m(np.zeros(m.state.ndx)) for m in self.problem.runningModels]

    def backwardPass(self):
        self.Vx[-1][:] = self.problem.terminalData.Lx
        self.Vxx[-1][:, :] = self.problem.terminalData.Lxx

        if self.x_reg != 0:
            ndx = self.problem.terminalModel.state.ndx
            self.Vxx[-1][range(ndx), range(ndx)] += self.x_reg

        # Compute and store the Vx gradient at end of the interval (rollout state)
        if not self.isFeasible:
            self.Vx[-1] += np.dot(self.Vxx[-1], self.gaps[-1])

        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.Qxx[t][:, :] = data.Lxx + data.Fx.T * self.Vxx[t + 1] * data.Fx
            self.Qxu[t][:, :] = data.Lxu + data.Fx.T * self.Vxx[t + 1] * data.Fu
            self.Quu[t][:, :] = data.Luu + data.Fu.T * self.Vxx[t + 1] * data.Fu
            self.Qx[t][:] = data.Lx + data.Fx.T * self.Vx[t + 1]
            self.Qu[t][:] = data.Lu + data.Fu.T * self.Vx[t + 1]

            if self.u_reg != 0:
                self.Quu[t][range(model.nu), range(model.nu)] += self.u_reg

            self.computeGains(t)

            if self.u_reg == 0:
                self.Vx[t][:] = self.Qx[t] - self.K[t].T * self.Qu[t]
            else:
                self.Vx[t][:] = self.Qx[t] - 2 * self.K[t].T * self.Qu[t] + self.K[t].T * self.Quu[t] * self.k[t]
            self.Vxx[t][:, :] = self.Qxx[t] - self.Qxu[t] * self.K[t]
            self.Vxx[t][:, :] = 0.5 * (self.Vxx[t][:, :] + self.Vxx[t][:, :].T)  # ensure symmetric

            if self.x_reg != 0:
                self.Vxx[t][range(model.state.ndx), range(model.state.ndx)] += self.x_reg

            # Compute and store the Vx gradient at end of the interval (rollout state)
            if not self.isFeasible:
                self.Vx[t] += np.dot(self.Vxx[t], self.gaps[t])

            raiseIfNan(self.Vxx[t], ArithmeticError('backward error'))
            raiseIfNan(self.Vx[t], ArithmeticError('backward error'))

    def computeGains(self, t):
        try:
            if self.Quu[t].shape[0] > 0:
                Lb = scl.cho_factor(self.Quu[t])
                self.K[t][:, :] = scl.cho_solve(Lb, self.Qux[t])
                self.k[t][:] = scl.cho_solve(Lb, self.Qu[t])
            else:
                pass
        except scl.LinAlgError:
            raise ArithmeticError('backward error')

    def forwardPass(self, stepLength, warning='ignore'):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            utry[t] = us[t] - self.k[t] * stepLength - np.dot(self.K[t], m.state.diff(xs[t], xtry[t]))
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, xtry[t], utry[t])
                xnext, cost = d.xnext, d.cost
            xtry[t + 1] = xnext.copy()  # not sure copy helpful here.
            ctry += cost
            raiseIfNan([ctry, cost], ArithmeticError('forward error'))
            raiseIfNan(xtry[t + 1], ArithmeticError('forward error'))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, ArithmeticError('forward error'))
        self.cost_try = ctry
        return xtry, utry, ctry


class FDDPDerived(DDPDerived):
    def __init__(self, shootingProblem):
        DDPDerived.__init__(self, shootingProblem)

        self.th_acceptNegStep = 2.
        self.dg = 0.
        self.dq = 0.
        self.dv = 0.

    def solve(self, init_xs=[], init_us=[], maxiter=100, isFeasible=False, regInit=None):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        self.wasFeasible = False
        for i in range(maxiter):
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    recalc = False
                    self.increaseRegularization()
                    if self.x_reg == self.regMax:
                        return self.xs, self.us, False
                    else:
                        continue
                break
            self.updateExpectedImprovement()

            for a in self.alphas:
                try:
                    self.dV = self.tryStep(a)
                except ArithmeticError:
                    continue
                d1, d2 = self.expectedImprovement()

                self.dV_exp = a * (d1 + .5 * d2 * a)
                if self.dV_exp >= 0.:  # descend direction
                    if d1 < self.th_grad or self.dV > self.th_acceptStep * self.dV_exp:
                        self.wasFeasible = self.isFeasible
                        self.setCandidate(self.xs_try, self.us_try, (self.wasFeasible or a == 1))
                        self.cost = self.cost_try
                        break
                else:  # reducing the gaps by allowing a small increment in the cost value
                    if self.dV > self.th_acceptNegStep * self.dV_exp:
                        self.wasFeasible = self.isFeasible
                        self.setCandidate(self.xs_try, self.us_try, (self.wasFeasible or a == 1))
                        self.cost = self.cost_try
                        break
            if a > self.th_step:
                self.decreaseRegularization()
            if a == self.alphas[-1]:
                self.increaseRegularization()
                if self.x_reg == self.regMax:
                    return self.xs, self.us, False
            self.stepLength = a
            self.iter = i
            self.stop = self.stoppingCriteria()
            # TODO @Carlos bind the callbacks
            # if self.callback is not None:
            #     [c(self) for c in self.callback]

            if self.wasFeasible and self.stop < self.th_stop:
                return self.xs, self.us, True
        return self.xs, self.us, False

    def computeDirection(self, recalc=True):
        if recalc:
            self.calc()
        self.backwardPass()
        return [np.nan] * (self.problem.T + 1), self.k, self.Vx

    def tryStep(self, stepLength=1):
        self.forwardPass(stepLength)
        return self.cost - self.cost_try

    def updateExpectedImprovement(self):
        self.dg = 0.
        self.dq = 0.
        if not self.isFeasible:
            self.dg -= np.asscalar(self.Vx[-1].T * self.gaps[-1])
            self.dq += np.asscalar(self.gaps[-1].T * self.Vxx[-1] * self.gaps[-1])
        for t in range(self.problem.T):
            self.dg += np.asscalar(self.Qu[t].T * self.k[t])
            self.dq -= np.asscalar(self.k[t].T * self.Quu[t] * self.k[t])
            if not self.isFeasible:
                self.dg -= np.asscalar(self.Vx[t].T * self.gaps[t])
                self.dq += np.asscalar(self.gaps[t].T * self.Vxx[t] * self.gaps[t])

    def expectedImprovement(self):
        self.dv = 0.
        if not self.isFeasible:
            dx = self.problem.runningModels[-1].state.diff(self.xs_try[-1], self.xs[-1])
            self.dv -= np.asscalar(self.gaps[-1].T * self.Vxx[-1] * dx)
            for t in range(self.problem.T):
                dx = self.problem.runningModels[t].state.diff(self.xs_try[t], self.xs[t])
                self.dv -= np.asscalar(self.gaps[t].T * self.Vxx[t] * dx)
        d1 = self.dg + self.dv
        d2 = self.dq - 2 * self.dv
        return np.matrix([d1, d2]).T

    def calc(self):
        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.gaps[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.x0)
            for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
                self.gaps[i + 1] = m.state.diff(x, d.xnext)
        elif not self.wasFeasible:
            self.gaps[:] = [np.zeros_like(f) for f in self.gaps]
        return self.cost

    def forwardPass(self, stepLength, warning='ignore'):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        xnext = self.problem.x0
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            if self.isFeasible or stepLength == 1:
                xtry[t] = xnext.copy()
            else:
                xtry[t] = m.state.integrate(xnext, self.gaps[t] * (stepLength - 1))
            utry[t] = us[t] - self.k[t] * stepLength - self.K[t] * m.state.diff(xs[t], xtry[t])
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, xtry[t], utry[t])
                xnext, cost = d.xnext, d.cost
            ctry += cost
            raiseIfNan([ctry, cost], ArithmeticError('forward error'))
            raiseIfNan(xnext, ArithmeticError('forward error'))
        if self.isFeasible or stepLength == 1:
            xtry[-1] = xnext.copy()
        else:
            xtry[-1] = self.problem.terminalModel.state.integrate(xnext, self.gaps[-1] * (stepLength - 1))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, ArithmeticError('forward error'))
        self.cost_try = ctry
        return xtry, utry, ctry
