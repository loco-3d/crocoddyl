import crocoddyl
import pinocchio
import numpy as np
import scipy.linalg as scl


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


def rev_enumerate(l):
    return reversed(list(enumerate(l)))


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
        v = np.matrix(np.zeros(self.nv)).T
        return np.concatenate([q, v])

    def rand(self):
        q = pinocchio.randomConfiguration(self.model)
        v = np.matrix(np.random.rand(self.nv)).T
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
        data.costResiduals = np.vstack([self.costWeights[0] * x, self.costWeights[1] * u])
        data.cost = .5 * sum(m2a(data.costResiduals)**2)

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
        v, w = m2a(u)
        px, py, theta = m2a(x)
        # Cost derivatives
        data.Lx = a2m(m2a(x) * ([self.costWeights[0]**2] * self.State.nx))
        data.Lu = a2m(m2a(u) * ([self.costWeights[1]**2] * self.nu))
        data.Lxx = np.diag([self.costWeights[0]**2] * self.State.nx)
        data.Luu = np.diag([self.costWeights[1]**2] * self.nu)
        # Dynamic derivatives
        c, s, dt = np.cos(theta), np.sin(theta), self.dt
        v, w = m2a(u)
        data.Fx = np.matrix([[1, 0, -s * v * dt], [0, 1, c * v * dt], [0, 0, 1]])
        data.Fu = np.matrix([[c * self.dt, 0], [s * self.dt, 0], [0, self.dt]])


class LQRDerived(crocoddyl.ActionModelAbstract):
    def __init__(self, nx, nu, driftFree=True):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(nx), nu)

        self.Fx = np.matrix(np.eye(self.State.nx))
        self.Fu = np.matrix(np.eye(self.State.nx))[:, :self.nu]
        self.f0 = np.matrix(np.zeros(self.State.nx)).T
        self.Lxx = np.matrix(np.eye(self.State.nx))
        self.Lxu = np.matrix(np.eye(self.State.nx))[:, :self.nu]
        self.Luu = np.matrix(np.eye(self.nu))
        self.lx = np.matrix(np.ones(self.State.nx)).T
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

        self.Fq = np.matrix(np.eye(self.State.nq))
        self.Fv = np.matrix(np.eye(self.State.nv))
        self.Fu = np.matrix(np.eye(self.State.nq))[:, :self.nu]
        self.f0 = np.matrix(np.zeros(self.State.nv)).T
        self.Lxx = np.matrix(np.eye(self.State.nx))
        self.Lxu = np.matrix(np.eye(self.State.nx))[:, :self.nu]
        self.Luu = np.matrix(np.eye(self.nu))
        self.lx = np.matrix(np.ones(self.State.nx)).T
        self.lu = np.matrix(np.ones(self.nu)).T

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[:self.State.nq], x[self.State.nq:]
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
    def __init__(self, state, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, state.nv, costModel.nr)
        self.costs = costModel
        self.forceAba = True
        self.armature = np.matrix(np.zeros(0))

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[:self.State.nq], x[-self.State.nv:]
        # Computing the dynamics using ABA or manually for armature case
        if self.forceAba:
            data.xout = pinocchio.aba(self.State.pinocchio, data.pinocchio, q, v, u)
        else:
            pinocchio.computeAllTerms(self.State.pinocchio, data.pinocchio, q, v)
            data.M = data.pinocchio.M
            if self.armature.size == self.State.nv:
                data.M[range(self.State.nv), range(self.State.nv)] += self.armature
            data.Minv = np.linalg.inv(data.M)
            data.xout = data.Minv * (u - data.pinocchio.nle)
        # Computing the cost value and residuals
        pinocchio.forwardKinematics(self.State.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.State.pinocchio, data.pinocchio)
        self.costs.calc(data.costs, x, u)
        data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        q, v = x[:self.State.nq], x[-self.State.nv:]
        if u is None:
            u = self.unone
        if recalc:
            self.calc(data, x, u)
            pinocchio.computeJointJacobians(self.State.pinocchio, data.pinocchio, q)
        # Computing the dynamics derivatives
        if self.forceAba:
            pinocchio.computeABADerivatives(self.State.pinocchio, data.pinocchio, q, v, u)
            data.Fx = np.hstack([data.pinocchio.ddq_dq, data.pinocchio.ddq_dv])
            data.Fu = data.pinocchio.Minv
        else:
            pinocchio.computeRNEADerivatives(self.State.pinocchio, data.pinocchio, q, v, data.xout)
            data.Fx = -np.hstack([data.Minv * data.pinocchio.dtau_dq, data.Minv * data.pinocchio.dtau_dv])
            data.Fu = data.Minv
        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u, False)

    def set_armature(self, armature):
        if armature.size is not self.State.nv:
            print('The armature dimension is wrong, we cannot set it.')
        else:
            self.forceAba = False
            self.armature = armature.T

    def createData(self):
        data = crocoddyl.DifferentialActionModelAbstract.createData(self)
        data.pinocchio = pinocchio.Data(self.State.pinocchio)
        data.costs = self.costs.createData(data.pinocchio)
        data.shareCostMemory(data.costs)
        return data


class IntegratedActionModelEuler(crocoddyl.ActionModelAbstract):
    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True):
        crocoddyl.ActionModelAbstract.__init__(self, diffModel.State, diffModel.nv, diffModel.nr)
        self.differential = diffModel
        self.withCostResiduals = withCostResiduals
        self.timeStep = timeStep

    def calc(self, data, x, u=None):
        nq, dt = self.State.nq, self.timeStep
        acc, cost = self.differential.calc(data.differential, x, u)
        if self.withCostResiduals:
            data.costResiduals = data.differential.costResiduals
        data.cost = cost
        # data.xnext[nq:] = x[nq:] + acc*dt
        # data.xnext[:nq] = pinocchio.integrate(self.differential.pinocchio,
        #                                       a2m(x[:nq]),a2m(data.xnext[nq:]*dt)).flat
        data.dx = np.concatenate([x[nq:] * dt + acc * dt**2, acc * dt])
        data.xnext[:] = self.differential.State.integrate(x, data.dx)

        return data.xnext, data.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        nv, dt = self.State.nv, self.timeStep
        if recalc:
            self.calc(data, x, u)
        self.differential.calcDiff(data.differential, x, u, recalc=False)
        dxnext_dx, dxnext_ddx = self.State.Jintegrate(x, data.dx)
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
        data.costResiduals = self.State.diff(self.xref, x)
        self.activation.calc(data.activation, data.costResiduals)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        data.Rx = self.State.Jdiff(self.xref, x, 'second')[0]
        self.activation.calcDiff(data.activation, data.costResiduals, recalc)
        data.Lx = data.Rx.T * data.activation.Ar
        data.Lxx = data.Rx.T * data.activation.Arr * data.Rx


class ControlCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, uref=None, nu=None):
        nu = nu if nu is not None else state.nv
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(nu)
        self.uref = uref if uref is not None else np.matrix(np.zeros(nu)).T
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u):
        data.costResiduals = u - self.uref
        self.activation.calc(data.activation, data.costResiduals)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        self.activation.calcDiff(data.activation, data.costResiduals, recalc)
        data.Lu = data.activation.Ar
        data.Luu = data.activation.Arr


class FramePlacementCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, Mref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.Mref = Mref

    def calc(self, data, x, u):
        data.rMf = self.Mref.oMf.inverse() * data.pinocchio.oMf[self.Mref.frame]
        data.costResiduals = pinocchio.log(data.rMf).vector
        self.activation.calc(data.activation, data.costResiduals)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        pinocchio.updateFramePlacements(self.State.pinocchio, data.pinocchio)
        data.rJf = pinocchio.Jlog6(data.rMf)
        data.fJf = pinocchio.getFrameJacobian(self.State.pinocchio, data.pinocchio, self.Mref.frame,
                                              pinocchio.ReferenceFrame.LOCAL)
        data.J = data.rJf * data.fJf
        self.activation.calcDiff(data.activation, data.costResiduals, recalc)
        data.Rx = np.hstack([data.J, np.zeros((self.activation.nr, self.State.nv))])
        data.Lx = np.vstack([data.J.T * data.activation.Ar, np.zeros((self.State.nv, 1))])
        data.Lxx = np.vstack([
            np.hstack([data.J.T * data.activation.Arr * data.J,
                       np.zeros((self.State.nv, self.State.nv))]),
            np.zeros((self.State.nv, self.State.ndx))
        ])


class FrameTranslationCostDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, xref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.xref = xref

    def calc(self, data, x, u):
        data.costResiduals = data.pinocchio.oMf[self.xref.frame].translation - self.xref.oxf
        self.activation.calc(data.activation, data.costResiduals)
        data.cost = data.activation.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        pinocchio.updateFramePlacements(self.State.pinocchio, data.pinocchio)
        data.R = data.pinocchio.oMf[self.xref.frame].rotation
        data.J = data.R * pinocchio.getFrameJacobian(self.State.pinocchio, data.pinocchio, self.xref.frame,
                                                     pinocchio.ReferenceFrame.LOCAL)[:3, :]
        self.activation.calcDiff(data.activation, data.costResiduals, recalc)
        data.Rx = np.hstack([data.J, np.zeros((self.activation.nr, self.State.nv))])
        data.Lx = np.vstack([data.J.T * data.activation.Ar, np.zeros((self.State.nv, 1))])
        data.Lxx = np.vstack([
            np.hstack([data.J.T * data.activation.Arr * data.J,
                       np.zeros((self.State.nv, self.State.nv))]),
            np.zeros((self.State.nv, self.State.ndx))
        ])


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
        self.Jw = np.matrix(np.zeros((3, state.pinocchio.nv)))

    def calc(self, data, x):
        assert (self.xref.oxf is not None or self.gains[0] == 0.)
        v = pinocchio.getFrameVelocity(self.State.pinocchio, data.pinocchio, self.xref.frame)
        self.vw = v.angular
        self.vv = v.linear

        fJf = pinocchio.getFrameJacobian(self.State.pinocchio, data.pinocchio, self.xref.frame,
                                         pinocchio.ReferenceFrame.LOCAL)
        data.Jc = fJf[:3, :]
        self.Jw = fJf[3:, :]

        data.a0 = pinocchio.getFrameAcceleration(self.State.pinocchio, data.pinocchio,
                                                 self.xref.frame).linear + pinocchio.utils.cross(self.vw, self.vv)
        if self.gains[0] != 0.:
            data.a0 += np.asscalar(self.gains[0]) * (data.pinocchio.oMf[self.xref.frame].translation - self.xref.oxf)
        if self.gains[1] != 0.:
            data.a0 += np.asscalar(self.gains[1]) * self.vv

    def calcDiff(self, data, x, recalc=True):
        if recalc:
            self.calc(data, x)
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pinocchio.getJointAccelerationDerivatives(
            self.State.pinocchio, data.pinocchio, self.joint, pinocchio.ReferenceFrame.LOCAL)

        vv_skew = pinocchio.utils.skew(self.vv)
        vw_skew = pinocchio.utils.skew(self.vw)
        fXjdv_dq = self.fXj * v_partial_dq
        Aq = (self.fXj * a_partial_dq)[:3, :] + vw_skew * fXjdv_dq[:3, :] - vv_skew * fXjdv_dq[3:, :]
        Av = (self.fXj * a_partial_dv)[:3, :] + vw_skew * data.Jc - vv_skew * self.Jw

        if np.asscalar(self.gains[0]) != 0.:
            R = data.pinocchio.oMf[self.xref.frame].rotation
            Aq += np.asscalar(self.gains[0]) * R * pinocchio.getFrameJacobian(
                self.State.pinocchio, data.pinocchio, self.xref.frame, pinocchio.ReferenceFrame.LOCAL)[:3, :]
        if np.asscalar(self.gains[1]) != 0.:
            Aq += np.asscalar(self.gains[1]) * (self.fXj[:3, :] * v_partial_dq)
            Av += np.asscalar(self.gains[1]) * (self.fXj[:3, :] * a_partial_da)
        data.Ax = np.hstack([Aq, Av])


class DDPDerived(crocoddyl.SolverAbstract):
    def __init__(self, shootingProblem):
        crocoddyl.SolverAbstract.__init__(self, shootingProblem)
        self.allocateData()  # TODO remove it?

        self.isFeasible = False  # Change it to true if you know that datas[t].xnext = xs[t+1]
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
            # Gap store the state defect from the guess to feasible (rollout) trajectory, i.e.
            #   gap = x_rollout [-] x_guess = DIFF(x_guess, x_rollout)
            self.gaps[0] = self.problem.runningModels[0].State.diff(self.xs[0], self.problem.initialState)
            for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
                self.gaps[i + 1] = m.State.diff(x, d.xnext)

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

        # Warning: no convergence in max iterations
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

    # DDP Specific
    def allocateData(self):
        self.Vxx = [a2m(np.zeros([m.State.ndx, m.State.ndx])) for m in self.models()]
        self.Vx = [a2m(np.zeros([m.State.ndx])) for m in self.models()]

        self.Q = [a2m(np.zeros([m.State.ndx + m.nu, m.State.ndx + m.nu])) for m in self.problem.runningModels]
        self.q = [a2m(np.zeros([m.State.ndx + m.nu])) for m in self.problem.runningModels]
        self.Qxx = [Q[:m.State.ndx, :m.State.ndx] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qxu = [Q[:m.State.ndx, m.State.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qux = [Qxu.T for m, Qxu in zip(self.problem.runningModels, self.Qxu)]
        self.Quu = [Q[m.State.ndx:, m.State.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qx = [q[:m.State.ndx] for m, q in zip(self.problem.runningModels, self.q)]
        self.Qu = [q[m.State.ndx:] for m, q in zip(self.problem.runningModels, self.q)]

        self.K = [np.matrix(np.zeros([m.nu, m.State.ndx])) for m in self.problem.runningModels]
        self.k = [a2m(np.zeros([m.nu])) for m in self.problem.runningModels]

        self.xs_try = [self.problem.initialState] + [np.nan * self.problem.initialState] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        self.gaps = [a2m(np.zeros(self.problem.runningModels[0].State.ndx))
                     ] + [a2m(np.zeros(m.State.ndx)) for m in self.problem.runningModels]

    def backwardPass(self):
        self.Vx[-1][:] = self.problem.terminalData.Lx
        self.Vxx[-1][:, :] = self.problem.terminalData.Lxx

        if self.x_reg != 0:
            ndx = self.problem.terminalModel.State.ndx
            self.Vxx[-1][range(ndx), range(ndx)] += self.x_reg

        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.Qxx[t][:, :] = data.Lxx + data.Fx.T * self.Vxx[t + 1] * data.Fx
            self.Qxu[t][:, :] = data.Lxu + data.Fx.T * self.Vxx[t + 1] * data.Fu
            self.Quu[t][:, :] = data.Luu + data.Fu.T * self.Vxx[t + 1] * data.Fu
            self.Qx[t][:] = data.Lx + data.Fx.T * self.Vx[t + 1]
            self.Qu[t][:] = data.Lu + data.Fu.T * self.Vx[t + 1]
            if not self.isFeasible:
                # In case the xt+1 are not f(xt,ut) i.e warm start not obtained from roll-out.
                relinearization = self.Vxx[t + 1] * self.gaps[t + 1]
                self.Qx[t][:] += data.Fx.T * relinearization
                self.Qu[t][:] += data.Fu.T * relinearization

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
                self.Vxx[t][range(model.State.ndx), range(model.State.ndx)] += self.x_reg
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
        # Argument warning is also introduce for debug: by default, it masks the numpy warnings
        #    that can be reactivated during debug.
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            utry[t] = us[t] - self.k[t] * stepLength - np.dot(self.K[t], m.State.diff(xs[t], xtry[t]))
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
