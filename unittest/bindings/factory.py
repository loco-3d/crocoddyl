import warnings

import numpy as np
import pinocchio
import scipy.linalg as scl

import crocoddyl


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


def rev_enumerate(lname):
    return reversed(list(enumerate(lname)))


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
        rng = np.random.default_rng()
        return np.matrix(rng.random(self.nx)).T

    def diff(self, x0, x1):
        return x1 - x0

    def integrate(self, x, dx):
        return x + dx

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.first),
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.second),
            ]

        J = np.zeros([self.ndx, self.ndx])
        if firstsecond == crocoddyl.Jcomponent.first:
            J[:, :] = -np.eye(self.ndx)
        elif firstsecond == crocoddyl.Jcomponent.second:
            J[:, :] = np.eye(self.ndx)
        return J

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second),
            ]
        return np.eye(self.ndx)


class StateMultibodyDerived(crocoddyl.StateAbstract):
    def __init__(self, pinocchioModel):
        crocoddyl.StateAbstract.__init__(
            self, pinocchioModel.nq + pinocchioModel.nv, 2 * pinocchioModel.nv
        )
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
        q0 = x0[: self.nq]
        q1 = x1[: self.nq]
        v0 = x0[-self.nv :]
        v1 = x1[-self.nv :]
        dq = pinocchio.difference(self.model, q0, q1)
        return np.concatenate([dq, v1 - v0])

    def integrate(self, x, dx):
        q = x[: self.nq]
        v = x[-self.nv :]
        dq = dx[: self.nv]
        dv = dx[-self.nv :]
        qn = pinocchio.integrate(self.model, q, dq)
        return np.concatenate([qn, v + dv])

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.first),
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.second),
            ]

        if firstsecond == crocoddyl.Jcomponent.first:
            dx = self.diff(x2, x1)
            q = x2[: self.model.nq]
            dq = dx[: self.model.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            dx = self.diff(x1, x2)
            q = x1[: self.nq]
            dq = dx[: self.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second),
            ]

        q = x[: self.nq]
        dq = dx[: self.nv]
        Jq, Jdq = pinocchio.dIntegrate(self.model, q, dq)
        if firstsecond == crocoddyl.Jcomponent.first:
            return np.matrix(scl.block_diag(np.linalg.inv(Jq), np.eye(self.nv)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))


class SquashingSmoothSatDerived(crocoddyl.SquashingModelAbstract):
    def __init__(self, u_lb, u_ub, ns):
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.smooth = 0.1
        crocoddyl.SquashingModelAbstract.__init__(self, ns)

    def calc(self, data, s):
        a = np.power(self.smooth * (self.u_ub - self.u_lb), 2)
        data.u = 0.5 * (
            self.u_lb + np.power(a + np.power(s - self.u_lb, 2), 0.5)
        ) + 0.5 * (self.u_ub - np.power(a + np.power(s - self.u_ub, 2), 0.5))

    def calcDiff(self, data, s):
        a = np.power(self.smooth * (self.u_ub - self.u_lb), 2)
        du_ds = 0.5 * (
            np.multiply(
                np.power(a + np.power((s - self.u_lb), 2), -0.5), (s - self.u_lb)
            )
            - np.multiply(
                np.power(a + np.power((s - self.u_ub), 2), -0.5), (s - self.u_ub)
            )
        )
        np.fill_diagonal(data.du_ds, du_ds)


class FreeFloatingActuationDerived(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        assert state.pinocchio.joints[1].shortname() == "JointModelFreeFlyer"
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv - 6)

    def calc(self, data, x, u):
        data.tau[:] = np.hstack([np.zeros(6), u])

    def calcDiff(self, data, x, u):
        data.dtau_du[:, :] = np.vstack([np.zeros((6, self.nu)), np.eye(self.nu)])


class FullActuationDerived(crocoddyl.ActuationModelAbstract):
    def __init__(self, state):
        assert state.pinocchio.joints[1].shortname() != "JointModelFreeFlyer"
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv)

    def calc(self, data, x, u):
        data.tau[:] = u

    def calcDiff(self, data, x, u):
        data.dtau_du[:, :] = pinocchio.utils.eye(self.nu)


class UnicycleModelDerived(crocoddyl.ActionModelAbstract):
    def __init__(self):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.dt = 0.1
        self.costWeights = [10.0, 1.0]

    def calc(self, data, x, u=None):
        if u is None:
            data.xnext[:] = x
            data.r[:3] = self.costWeights[0] * x
            data.cost = 0.5 * sum(data.r**2)
        else:
            v, w = u
            px, py, theta = x
            c, s, dt = np.cos(theta), np.sin(theta), self.dt
            data.xnext[0] = px + c * v * dt
            data.xnext[1] = py + s * v * dt
            data.xnext[2] = theta + w * dt
            data.r[:3] = self.costWeights[0] * x
            data.r[3:] = self.costWeights[1] * u
            data.cost = 0.5 * sum(data.r**2)

    def calcDiff(self, data, x, u=None):
        if u is None:
            data.Lx[:] = x * ([self.costWeights[0] ** 2] * self.state.nx)
        else:
            v = u[0]
            theta = x[2]
            data.Lx[:] = x * ([self.costWeights[0] ** 2] * self.state.nx)
            c, s, dt = np.cos(theta), np.sin(theta), self.dt
            data.Fx[0, 2] = -s * v * dt
            data.Fx[1, 2] = c * v * dt
            data.Fu[0, 0] = c * dt
            data.Fu[1, 0] = s * dt
            data.Fu[2, 1] = dt
            data.Lu[:] = u * ([self.costWeights[1] ** 2] * self.nu)

    def createData(self):
        data = UnicycleDataDerived(self)
        return data


class UnicycleDataDerived(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        nx, nu = model.state.nx, model.nu
        self.Lxx[range(nx), range(nx)] = [model.costWeights[0] ** 2] * nx
        self.Luu[range(nu), range(nu)] = [model.costWeights[1] ** 2] * nu
        self.Fx[0, 0] = 1
        self.Fx[1, 1] = 1
        self.Fx[2, 2] = 1


class LQRModelDerived(crocoddyl.ActionModelAbstract):
    def __init__(self, nx, nu, driftFree=True):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(nx), nu)
        self.A = np.eye(self.state.nx)
        self.B = np.eye(self.state.nx)[:, : self.nu]
        self.Q = np.eye(self.state.nx)
        self.R = np.eye(self.nu)
        self.N = np.zeros((self.state.nx, self.nu))
        self.f = [np.zeros(self.state.nx) if driftFree else np.ones(self.state.nx)]
        self.q = np.ones(self.state.nx)
        self.r = np.ones(self.nu)

    @classmethod
    def fromLQR(cls, A, B, Q, R, N, f, q, r):
        model = cls(A.shape[1], B.shape[1], False)
        model.A = A
        model.B = B
        model.Q = Q
        model.R = R
        model.N = N
        model.f = f
        model.q = q
        model.r = r
        return model

    def calc(self, data, x, u=None):
        if u is None:
            data.xnext[:] = x
            data.cost = 0.5 * np.dot(x.T, np.dot(self.Q, x))
            data.cost += np.dot(self.q.T, x)
        else:
            data.xnext[:] = np.dot(self.A, x) + np.dot(self.B, u) + self.f
            data.cost = 0.5 * np.dot(x.T, np.dot(self.Q, x))
            data.cost += 0.5 * np.dot(u.T, np.dot(self.R, u))
            data.cost += np.dot(x.T, np.dot(self.N, u))
            data.cost += np.dot(self.q.T, x) + np.dot(self.r.T, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            data.Lx[:] = self.q + np.dot(self.Q, x)
        else:
            data.Lx[:] = self.q + np.dot(self.Q, x) + np.dot(self.N, u)
            data.Lu[:] = self.r + np.dot(self.R, u) + np.dot(self.N.T, x)

    def createData(self):
        data = LQRDataDerived(self)
        return data


class LQRDataDerived(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        self.Fx[:, :] = model.A
        self.Fu[:, :] = model.B
        self.Lxx[:, :] = model.Q
        self.Luu[:, :] = model.R
        self.Lxu[:, :] = model.N


class DifferentialLQRModelDerived(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, nq, nu, driftFree=True):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, crocoddyl.StateVector(2 * nq), nu
        )
        self.Aq = np.eye(self.state.nq)
        self.Av = np.eye(self.state.nv)
        self.B = np.eye(self.state.nq)[:, : self.nu]
        self.f = [np.zeros(nq) if driftFree else np.ones(nq)]
        self.Q = np.eye(self.state.nx)
        self.R = np.eye(self.nu)
        self.N = np.zeros((self.state.nx, self.nu))
        self.q = np.ones(self.state.nx)
        self.r = np.ones(self.nu)

    @classmethod
    def fromLQR(cls, Aq, Av, B, Q, R, N, f, q, r):
        model = cls(Aq.shape[1], B.shape[1], False)
        model.Aq = Aq
        model.Av = Av
        model.B = B
        model.Q = Q
        model.R = R
        model.N = N
        model.f = f
        model.q = q
        model.r = r
        return model

    def calc(self, data, x, u=None):
        if u is None:
            data.cost = 0.5 * np.dot(x.T, np.dot(self.Q, x))
            data.cost += np.dot(self.q.T, x)
        else:
            q, v = x[: self.state.nq], x[self.state.nq :]
            data.xout[:] = (
                np.dot(self.Aq, q) + np.dot(self.Av, v) + np.dot(self.B, u) + self.f
            )
            data.cost = 0.5 * np.dot(x.T, np.dot(self.Q, x))
            data.cost += 0.5 * np.dot(u.T, np.dot(self.R, u))
            data.cost += np.dot(x.T, np.dot(self.N, u))
            data.cost += np.dot(self.q.T, x) + np.dot(self.r.T, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            data.Lx[:] = self.q + np.dot(self.Q, x)
        else:
            data.Lx[:] = self.q + np.dot(self.Q, x) + np.dot(self.N, u)
            data.Lu[:] = self.r + np.dot(self.R, u) + np.dot(self.N.T, x)

    def createData(self):
        data = DifferentialLQRDataDerived(self)
        return data


class DifferentialLQRDataDerived(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.Fx[:, :] = np.hstack([model.Aq, model.Av])
        self.Fu[:, :] = model.B
        self.Lxx[:, :] = model.Q
        self.Luu[:, :] = model.R
        self.Lxu[:, :] = model.N


class DifferentialFreeFwdDynamicsModelDerived(
    crocoddyl.DifferentialActionModelAbstract
):
    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(
            self, state, actuationModel.nu, costModel.nr
        )
        self.actuation = actuationModel
        self.costs = costModel
        self.enable_force = True
        self.armature = np.matrix(np.zeros(0))

    def calc(self, data, x, u=None):
        if u is None:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
            self.costs.calc(data.costs, x)
            data.cost = data.costs.cost
        else:
            q, v = x[: self.state.nq], x[-self.state.nv :]
            self.actuation.calc(data.actuation, x, u)
            tau = data.actuation.tau
            # Computing the dynamics using ABA or manually for armature case
            if self.enable_force:
                data.xout[:] = pinocchio.aba(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
            else:
                pinocchio.computeAllTerms(self.state.pinocchio, data.pinocchio, q, v)
                data.M = data.pinocchio.M
                if self.armature.size == self.state.nv:
                    data.M[range(self.state.nv), range(self.state.nv)] += self.armature
                data.Minv = np.linalg.inv(data.M)
                data.xout[:] = np.dot(data.Minv, (tau - data.pinocchio.nle))
            # Computing the cost value and residuals
            pinocchio.forwardKinematics(self.state.pinocchio, data.pinocchio, q, v)
            pinocchio.updateFramePlacements(self.state.pinocchio, data.pinocchio)
            self.costs.calc(data.costs, x, u)
            data.cost = data.costs.cost

    def calcDiff(self, data, x, u=None):
        if u is None:
            self.costs.calcDiff(data.costs, x)
        else:
            nq, nv = self.state.nq, self.state.nv
            q, v = x[:nq], x[-nv:]
            # Computing the actuation derivatives
            self.actuation.calcDiff(data.actuation, x, u)
            tau = data.actuation.tau
            # Computing the dynamics derivatives
            if self.enable_force:
                pinocchio.computeABADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, tau
                )
                ddq_dq = data.pinocchio.ddq_dq
                ddq_dv = data.pinocchio.ddq_dv
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + np.dot(
                    data.pinocchio.Minv, data.actuation.dtau_dx
                )
                data.Fu[:, :] = np.dot(data.pinocchio.Minv, data.actuation.dtau_du)
            else:
                pinocchio.computeRNEADerivatives(
                    self.state.pinocchio, data.pinocchio, q, v, data.xout
                )
                ddq_dq = np.dot(
                    data.Minv, (data.actuation.dtau_dx[:, :nv] - data.pinocchio.dtau_dq)
                )
                ddq_dv = np.dot(
                    data.Minv, (data.actuation.dtau_dx[:, nv:] - data.pinocchio.dtau_dv)
                )
                data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv])
                data.Fu[:, :] = np.dot(data.Minv, data.actuation.dtau_du)
            # Computing the cost derivatives
            self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DifferentialFreeFwdDynamicsDataDerived(self)
        return data

    def set_armature(self, armature):
        if armature.size is not self.state.nv:
            print("The armature dimension is wrong, we cannot set it.")
        else:
            self.enable_force = False
            self.armature = armature.T


class DifferentialFreeFwdDynamicsDataDerived(crocoddyl.DifferentialActionDataAbstract):
    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.pinocchio = pinocchio.Model.createData(model.state.pinocchio)
        self.multibody = crocoddyl.DataCollectorMultibody(self.pinocchio)
        self.actuation = model.actuation.createData()
        self.costs = model.costs.createData(self.multibody)
        self.costs.shareMemory(self)
        self.Minv = None


class IntegratedActionModelEulerDerived(crocoddyl.ActionModelAbstract):
    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True):
        crocoddyl.ActionModelAbstract.__init__(
            self, diffModel.state, diffModel.nu, diffModel.nr
        )
        self.differential = diffModel
        self.withCostResiduals = withCostResiduals
        self.timeStep = timeStep

    def calc(self, data, x, u=None):
        if u is None:
            self.differential.calc(data.differential, x)
            data.dx[:] *= 0.0
            data.xnext[:] = x
            data.cost = data.differential.cost
        else:
            nq, dt = self.state.nq, self.timeStep
            self.differential.calc(data.differential, x, u)
            acc = data.differential.xout
            if self.withCostResiduals:
                data.r = data.differential.r
            data.cost = dt * data.differential.cost
            data.dx[:] = np.concatenate([x[nq:] * dt + acc * dt**2, acc * dt])
            data.xnext[:] = self.differential.state.integrate(x, data.dx)

    def calcDiff(self, data, x, u=None):
        if u is None:
            self.differential.calcDiff(data.differential, x)
            dxnext_dx, _ = self.state.Jintegrate(x, data.dx)
            data.Fx[:, :] = dxnext_dx
            data.Lx[:] = data.differential.Lx
            data.Lxx[:] = data.differential.Lxx
        else:
            nv, dt = self.state.nv, self.timeStep
            self.differential.calcDiff(data.differential, x, u)
            dxnext_dx, dxnext_ddx = self.state.Jintegrate(x, data.dx)
            da_dx, da_du = data.differential.Fx, data.differential.Fu
            ddx_dx = np.vstack([da_dx * dt, da_dx])
            ddx_dx[range(nv), range(nv, 2 * nv)] += 1
            data.Fx[:, :] = dxnext_dx + dt * np.dot(dxnext_ddx, ddx_dx)
            ddx_du = np.vstack([da_du * dt, da_du])
            if self.nu == 1:
                data.Fu[:] = (dt * np.dot(dxnext_ddx, ddx_du)).reshape(self.state.nx)
            else:
                data.Fu[:, :] = dt * np.dot(dxnext_ddx, ddx_du)
            data.Lx[:] = data.differential.Lx * dt
            data.Lu[:] = data.differential.Lu * dt
            data.Lxx[:, :] = data.differential.Lxx * dt
            data.Lxu[:, :] = data.differential.Lxu * dt
            data.Luu[:, :] = data.differential.Luu * dt

    def createData(self):
        data = IntegratedActionDataEulerDerived(self)
        return data


class IntegratedActionDataEulerDerived(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        self.differential = model.differential.createData()
        self.dx = np.zeros(model.state.ndx)


class IntegratedActionModelRK4Derived(crocoddyl.ActionModelAbstract):
    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True):
        crocoddyl.ActionModelAbstract.__init__(
            self, diffModel.state, diffModel.nu, diffModel.nr
        )
        self.differential = diffModel
        self.timeStep = timeStep
        self.rk4_inc = [0.5, 0.5, 1.0]
        self.nx = self.differential.state.nx
        self.ndx = self.differential.state.ndx
        self.nq = self.differential.state.nq
        self.nv = self.differential.state.nv

    def createData(self):
        return IntegratedActionDataRK4Derived(self)

    def calc(self, data, x, u=None):
        if u is None:
            k0_data = data.differential[0]
            self.differential.calc(k0_data, x)
            data.dx[:] *= 0.0
            data.xnext[:] = x
            data.cost = k0_data.cost
        else:
            nq, dt = self.nq, self.timeStep
            data.y[0] = x
            for i in range(3):
                self.differential.calc(data.differential[i], data.y[i], u)
                data.acc[i] = data.differential[i].xout
                data.int[i] = data.differential[i].cost
                data.ki[i] = np.concatenate([data.y[i][nq:], data.acc[i]])
                data.y[i + 1] = self.differential.state.integrate(
                    x, data.ki[i] * self.rk4_inc[i] * dt
                )
            self.differential.calc(data.differential[3], data.y[3], u)
            data.acc[3] = data.differential[3].xout
            data.int[3] = data.differential[3].cost
            data.ki[3] = np.concatenate([data.y[3][nq:], data.acc[3]])
            data.dx = (
                (data.ki[0] + 2.0 * data.ki[1] + 2.0 * data.ki[2] + data.ki[3]) * dt / 6
            )
            data.xnext = self.differential.state.integrate(x, data.dx)
            data.cost = (
                (data.int[0] + 2 * data.int[1] + 2 * data.int[2] + data.int[3]) * dt / 6
            )

    def calcDiff(self, data, x, u=None):
        if u is None:
            k0_data = data.differential[0]
            self.differential.calcDiff(k0_data, x)
            data.Lx[:] = k0_data.Lx
            data.Lxx[:] = k0_data.Lxx
        else:
            ndx, nu, nv, dt = self.ndx, self.nu, self.nv, self.timeStep
            for i in range(4):
                self.differential.calcDiff(data.differential[i], data.y[i], u)
                data.dki_dy[i] = np.bmat(
                    [[np.zeros([nv, nv]), np.identity(nv)], [data.differential[i].Fx]]
                )
            data.dki_du[0] = np.vstack([np.zeros([nv, nu]), data.differential[0].Fu])
            data.Lx[:] = data.differential[0].Lx
            data.Lu[:] = data.differential[0].Lu
            data.dy_du[0] = np.zeros((ndx, nu))
            data.dki_dx[0] = data.dki_dy[0]
            data.dli_dx[0] = data.differential[0].Lx
            data.dli_du[0] = data.differential[0].Lu
            data.ddli_ddx[0] = data.differential[0].Lxx
            data.ddli_ddu[0] = data.differential[0].Luu
            data.ddli_dxdu[0] = data.differential[0].Lxu
            for i in range(1, 4):
                c = self.rk4_inc[i - 1] * dt
                dyi_dx, dyi_ddx = self.state.Jintegrate(x, c * data.ki[i - 1])
                data.dy_du[i] = c * np.dot(dyi_ddx, data.dki_du[i - 1])
                data.dki_du[i] = np.vstack(
                    [
                        c * data.dki_du[i - 1][nv:, :],
                        data.differential[i].Fu
                        + np.dot(data.differential[i].Fx, data.dy_du[i]),
                    ]
                )
                data.dli_du[i] = data.differential[i].Lu + np.dot(
                    data.differential[i].Lx, data.dy_du[i]
                )
                data.Luu_partialx[i] = np.dot(data.differential[i].Lxu.T, data.dy_du[i])
                data.ddli_ddu[i] = (
                    data.differential[i].Luu
                    + data.Luu_partialx[i].T
                    + data.Luu_partialx[i]
                    + np.dot(
                        data.dy_du[i].T, np.dot(data.differential[i].Lxx, data.dy_du[i])
                    )
                )
                data.dy_dx[i] = dyi_dx + c * np.dot(dyi_ddx, data.dki_dx[i - 1])
                data.dki_dx[i] = np.dot(data.dki_dy[i], data.dy_dx[i])

                data.dli_dx[i] = np.dot(data.differential[i].Lx, data.dy_dx[i])
                data.ddli_ddx[i] = np.dot(
                    data.dy_dx[i].T, np.dot(data.differential[i].Lxx, data.dy_dx[i])
                )
                data.ddli_dxdu[i] = np.dot(
                    data.dy_dx[i].T, data.differential[i].Lxu
                ) + np.dot(
                    data.dy_dx[i].T, np.dot(data.differential[i].Lxx, data.dy_du[i])
                )
            dxnext_dx, dxnext_ddx = self.state.Jintegrate(x, data.dx)
            ddx_dx = (
                (
                    data.dki_dx[0]
                    + 2.0 * data.dki_dx[1]
                    + 2.0 * data.dki_dx[2]
                    + data.dki_dx[3]
                )
                * dt
                / 6
            )
            data.ddx_du = (
                (
                    data.dki_du[0]
                    + 2.0 * data.dki_du[1]
                    + 2.0 * data.dki_du[2]
                    + data.dki_du[3]
                )
                * dt
                / 6
            )
            data.Fx[:] = dxnext_dx + np.dot(dxnext_ddx, ddx_dx)
            data.Fu[:] = np.dot(dxnext_ddx, data.ddx_du)
            data.Lx[:] = (
                (
                    data.dli_dx[0]
                    + 2.0 * data.dli_dx[1]
                    + 2.0 * data.dli_dx[2]
                    + data.dli_dx[3]
                )
                * dt
                / 6
            )
            data.Lu[:] = (
                (
                    data.dli_du[0]
                    + 2.0 * data.dli_du[1]
                    + 2.0 * data.dli_du[2]
                    + data.dli_du[3]
                )
                * dt
                / 6
            )
            data.Lxx[:] = (
                (
                    data.ddli_ddx[0]
                    + 2.0 * data.ddli_ddx[1]
                    + 2.0 * data.ddli_ddx[2]
                    + data.ddli_ddx[3]
                )
                * dt
                / 6
            )
            data.Luu[:] = (
                (
                    data.ddli_ddu[0]
                    + 2.0 * data.ddli_ddu[1]
                    + 2.0 * data.ddli_ddu[2]
                    + data.ddli_ddu[3]
                )
                * dt
                / 6
            )
            data.Lxu[:] = (
                (
                    data.ddli_dxdu[0]
                    + 2.0 * data.ddli_dxdu[1]
                    + 2.0 * data.ddli_dxdu[2]
                    + data.ddli_dxdu[3]
                )
                * dt
                / 6
            )
            data.Lux = data.Lxu.T


class IntegratedActionDataRK4Derived(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)

        nx, ndx, nv, nu = model.state.nx, model.state.ndx, model.state.nv, model.nu
        self.differential = [None] * 4

        for i in range(4):
            self.differential[i] = model.differential.createData()
        self.int = [np.nan] * 4
        self.ki = [np.zeros([ndx])] * 4

        self.F = np.zeros([ndx, ndx + nu])
        self.xnext = np.zeros([nx])
        self.cost = np.nan

        self.Lx = np.zeros(ndx)
        self.Lu = np.zeros(nu)
        self.Lxx = np.zeros([ndx, ndx])
        self.Lxu = np.zeros([ndx, nu])
        self.Luu = np.zeros([nu, nu])
        self.Fx = self.F[:, :ndx]
        self.Fu = self.F[:, ndx:]

        # Quantities for derivatives
        self.dx = np.zeros([ndx])
        self.y = [np.zeros([nx])] * 4
        self.acc = [np.zeros([nu])] * 4

        self.dki_dy = [np.zeros([ndx, ndx])] * 4
        self.dki_dx = [np.zeros([ndx, ndx])] * 4
        self.dy_dx = [np.zeros([ndx, ndx])] * 4
        self.dki_du = [np.zeros([ndx, nu])] * 4
        self.dy_du = [np.zeros([ndx, nu])] * 4

        self.ddx_du = np.zeros([ndx, nu])
        self.dli_dx = [np.zeros([ndx])] * 4
        self.dli_du = [np.zeros([nu])] * 4

        self.ddli_ddx = [np.zeros([ndx, ndx])] * 4
        self.ddli_ddu = [np.zeros([nu, nu])] * 4
        self.ddli_dxdu = [np.zeros([ndx, nu])] * 4
        self.Luu_partialx = [np.zeros([nu, nu])] * 4

        self.dy_dx[0][:, :] = np.identity(nv * 2)


class StateCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, xref=None, nu=None):
        activation = (
            activation
            if activation is not None
            else crocoddyl.ActivationModelQuad(state.ndx)
        )
        self.xref = xref if xref is not None else state.zero()
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u=None):
        data.residual.r[:] = self.state.diff(self.xref, x)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        # The old code was looking like this.
        # But, the std::vector<Eigen::MatrixXd> returned by Jdiff is destroyed
        # before the assignment.
        # To avoid this issue, we store the std::vector in a variable.
        # data.residual.Rx[:] = self.state.Jdiff(
        #     self.xref, x, crocoddyl.Jcomponent.second
        # )[0]

        diff = self.state.Jdiff(self.xref, x, crocoddyl.Jcomponent.second)
        data.residual.Rx[:] = diff[0]
        self.activation.calcDiff(data.activation, data.residual.r)
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)
        data.Lxx[:, :] = np.dot(
            data.residual.Rx.T, np.dot(data.activation.Arr, data.residual.Rx)
        )


class ControlCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, uref=None, nu=None):
        nu = nu if nu is not None else state.nv
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(nu)
        )
        self.uref = uref if uref is not None else pinocchio.utils.zero(nu)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u=None):
        if u is None:
            data.cost = 0.0
        else:
            data.residual.r[:] = u - self.uref
            self.activation.calc(data.activation, data.residual.r)
            data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        if u is not None:
            self.activation.calcDiff(data.activation, data.residual.r)
            data.Lu[:] = data.activation.Ar
            data.Luu[:, :] = data.activation.Arr


class CoMPositionCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, cref=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.cref = cref

    def calc(self, data, x, u=None):
        data.residual.r[:] = data.shared.pinocchio.com[0] - self.cref
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [
                data.shared.pinocchio.Jcom,
                pinocchio.utils.zero((self.activation.nr, self.state.nv)),
            ]
        )
        data.Lx[:] = np.hstack(
            [
                np.dot(data.shared.pinocchio.Jcom.T, data.activation.Ar),
                np.zeros(self.state.nv),
            ]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(
                            data.shared.pinocchio.Jcom.T,
                            np.dot(data.activation.Arr, data.shared.pinocchio.Jcom),
                        ),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )


class FramePlacementCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, frame_id=None, placement=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._placement = placement

    def calc(self, data, x, u=None):
        data.rMf = self._placement.inverse() * data.shared.pinocchio.oMf[self._frame_id]
        data.residual.r[:] = pinocchio.log(data.rMf).vector
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf[:, :] = pinocchio.Jlog6(data.rMf)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.shared.pinocchio,
            self._frame_id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        data.J[:, :] = np.dot(data.rJf, data.fJf)
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [data.J, np.zeros((self.activation.nr, self.state.nv))]
        )
        data.Lx[:] = np.hstack(
            [np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )

    def createData(self, collector):
        data = FramePlacementCostDataDerived(self, collector)
        return data


class FramePlacementCostDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.rMf = pinocchio.SE3.Identity()
        self.rJf = pinocchio.Jlog6(self.rMf)
        self.fJf = np.zeros((6, model.state.nv))
        self.rJf = np.zeros((6, 6))
        self.J = np.zeros((6, model.state.nv))


class FrameTranslationCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(
        self, state, activation=None, frame_id=None, translation=None, nu=None
    ):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._translation = translation

    def calc(self, data, x, u=None):
        data.residual.r[:] = (
            data.shared.pinocchio.oMf[self._frame_id].translation - self._translation
        )
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.R[:, :] = data.shared.pinocchio.oMf[self._frame_id].rotation
        data.J[:, :] = np.dot(
            data.R,
            pinocchio.getFrameJacobian(
                self.state.pinocchio,
                data.shared.pinocchio,
                self._frame_id,
                pinocchio.ReferenceFrame.LOCAL,
            )[:3, :],
        )
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [data.J, np.zeros((self.activation.nr, self.state.nv))]
        )
        data.Lx[:] = np.hstack(
            [np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )

    def createData(self, collector):
        data = FrameTranslationDataDerived(self, collector)
        return data


class FrameTranslationDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.R = np.eye(3)
        self.J = np.zeros((3, model.state.nv))


class FrameRotationCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, frame_id=None, rotation=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._rotation = rotation

    def calc(self, data, x, u=None):
        data.rRf[:, :] = np.dot(
            self._rotation.T, data.shared.pinocchio.oMf[self._frame_id].rotation
        )
        data.residual.r[:] = pinocchio.log3(data.rRf)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf[:, :] = pinocchio.Jlog3(data.rRf)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.shared.pinocchio,
            self._frame_id,
            pinocchio.ReferenceFrame.LOCAL,
        )[3:, :]
        data.J[:, :] = np.dot(data.rJf, data.fJf)
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [data.J, np.zeros((self.activation.nr, self.state.nv))]
        )
        data.Lx[:] = np.hstack(
            [np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )

    def createData(self, collector):
        data = FrameRotationCostDataDerived(self, collector)
        return data


class FrameRotationCostDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.rRf = np.eye(3)
        self.rJf = np.zeros((3, 3))
        self.fJf = np.zeros((3, model.state.nv))
        self.J = np.zeros((3, model.state.nv))


class FrameVelocityCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, frame_id=None, velocity=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._velocity = velocity

    def calc(self, data, x, u=None):
        data.residual.r[:] = (
            pinocchio.getFrameVelocity(
                self.state.pinocchio,
                data.shared.pinocchio,
                self._frame_id,
                pinocchio.LOCAL,
            )
            - self._velocity
        ).vector
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(
            self.state.pinocchio,
            data.shared.pinocchio,
            data.joint,
            pinocchio.ReferenceFrame.LOCAL,
        )
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [np.dot(data.fXj, v_partial_dq), np.dot(data.fXj, v_partial_dv)]
        )
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)
        data.Lxx[:, :] = np.dot(
            data.residual.Rx.T, np.dot(data.activation.Arr, data.residual.Rx)
        )

    def createData(self, collector):
        data = FrameVelocityCostDataDerived(self, collector)
        return data


class FrameVelocityCostDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.fXj = (
            model.state.pinocchio.frames[model._frame_id].placement.inverse().action
        )
        if tuple(int(i) for i in pinocchio.__version__.split(".")) >= (3, 0, 0):
            self.joint = model.state.pinocchio.frames[model._frame_id].parentJoint
        else:
            self.joint = model.state.pinocchio.frames[model._frame_id].parent


class Contact1DModelDerived(crocoddyl.ContactModelAbstract):
    def __init__(
        self, state, id, xref, type=pinocchio.ReferenceFrame.LOCAL, gains=[0.0, 0.0]
    ):
        crocoddyl.ContactModelAbstract.__init__(self, state, type, 1)
        self.id = id
        self.xref = xref
        self.gains = gains
        self.Raxis = np.eye(3)

    def calc(self, data, x):
        assert self.xref is not None or self.gains[0] == 0.0
        pinocchio.updateFramePlacement(self.state.pinocchio, data.pinocchio, self.id)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.pinocchio,
            self.id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        data.v = pinocchio.getFrameVelocity(
            self.state.pinocchio, data.pinocchio, self.id
        )
        data.a0_local[:] = pinocchio.getFrameClassicalAcceleration(
            self.state.pinocchio, data.pinocchio, self.id, pinocchio.LOCAL
        ).linear

        oRf = data.pinocchio.oMf[self.id].rotation
        data.vw[:] = data.v.angular
        data.vv[:] = data.v.linear
        data.dp[:] = data.pinocchio.oMf[self.id].translation - self.xref * np.dot(
            self.Raxis, np.array([0, 0, 1])
        )
        data.dp_local[:] = np.dot(oRf.T, data.dp)

        if self.gains[0] != 0.0:
            data.a0_local[:] += self.gains[0] * data.dp_local
        if self.gains[1] != 0.0:
            data.a0_local[:] += self.gains[1] * data.vv

        if self.type == pinocchio.LOCAL:
            data.Jc = np.dot(self.Raxis, data.fJf[:3, :])[2]
            data.a0[0] = np.dot(self.Raxis, data.a0_local)[2]
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.Jc = np.dot(np.dot(self.Raxis, oRf), data.fJf[:3, :])[2]
            data.a0[0] = np.dot(np.dot(self.Raxis, oRf), data.a0_local)[2]

    def calcDiff(self, data, x):
        if tuple(int(i) for i in pinocchio.__version__.split(".")) >= (3, 0, 0):
            joint = self.state.pinocchio.frames[self.id].parentJoint
        else:
            joint = self.state.pinocchio.frames[self.id].parent
        (
            v_partial_dq,
            a_partial_dq,
            a_partial_dv,
            _,
        ) = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, joint, pinocchio.ReferenceFrame.LOCAL
        )
        nv = self.state.nv
        data.vv_skew[:, :] = pinocchio.skew(data.vv)
        data.vw_skew[:, :] = pinocchio.skew(data.vw)
        data.dp_skew[:, :] = pinocchio.skew(data.dp_local)
        data.fXjdv_dq[:] = np.dot(data.fXj, v_partial_dq)
        data.fXjda_dq[:] = np.dot(data.fXj, a_partial_dq)
        data.fXjda_dv[:] = np.dot(data.fXj, a_partial_dv)
        data.da0_local_dx[:, :nv] = data.fXjda_dq[:3, :]
        data.da0_local_dx[:, :nv] += np.dot(data.vw_skew, data.fXjdv_dq[:3, :])
        data.da0_local_dx[:, :nv] -= np.dot(data.vv_skew, data.fXjdv_dq[3:, :])
        data.da0_local_dx[:, nv:] = data.fXjda_dv[:3, :]
        data.da0_local_dx[:, nv:] += np.dot(data.vw_skew, data.fJf[:3, :])
        data.da0_local_dx[:, nv:] -= np.dot(data.vv_skew, data.fJf[3:, :])
        oRf = data.pinocchio.oMf[self.id].rotation

        if self.gains[0] != 0.0:
            data.da0_local_dx[:, :nv] += self.gains[0] * np.dot(
                data.dp_skew, data.fJf[3:, :]
            )
            data.da0_local_dx[:, :nv] += self.gains[0] * data.fJf[:3, :]
        if self.gains[1] != 0.0:
            data.da0_local_dx[:, :nv] += self.gains[1] * data.fXjdv_dq[:3, :]
            data.da0_local_dx[:, nv:] += self.gains[1] * data.fJf[:3, :]

        if self.type == pinocchio.LOCAL:
            data.da0_dx[:] = (self.Raxis @ data.da0_local_dx)[2, :]
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.a0_local[:] = pinocchio.getFrameClassicalAcceleration(
                self.state.pinocchio, data.pinocchio, self.id, pinocchio.LOCAL
            ).linear
            if self.gains[0] != 0.0:
                data.a0_local[:] += self.gains[0] * data.dp_local
            if self.gains[1] != 0.0:
                data.a0_local[:] += self.gains[1] * data.vv
            data.a0[0] = np.dot(np.dot(self.Raxis, oRf), data.a0_local)[2]

            data.a0_skew[:, :] = pinocchio.skew(
                np.dot(np.dot(self.Raxis, oRf), data.a0_local)
            )
            data.a0_world_skew = np.dot(data.a0_skew, np.dot(self.Raxis, oRf))
            data.da0_dx[:] = np.dot(np.dot(self.Raxis, oRf), data.da0_local_dx)[2]
            data.da0_dx[:nv] -= np.dot(data.a0_world_skew, data.fJf[3:, :])[2]

    def createData(self, data):
        data = Contact1DDataDerived(self, data)
        return data

    def updateForce(self, data, force):
        assert force.shape[0] == 1
        nv = self.state.nv
        data.f.linear[2] = force[0]
        data.f.linear[:2] = np.zeros(2)
        data.f.angular = np.zeros(3)
        if self.type == pinocchio.LOCAL:
            data.fext.linear = np.dot(data.jMf.rotation, self.Raxis.T)[:, 2] * force[0]
            data.fext.angular = np.cross(data.jMf.translation, data.fext.linear)
            data.dtau_dq[:, :] = np.zeros((nv, nv))
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            oRf = data.pinocchio.oMf[self.id].rotation
            data.f_local.linear = np.dot(oRf.T, self.Raxis.T)[:, 2] * force[0]
            data.f_local.angular = np.zeros(3)
            data.fext = data.jMf.act(data.f_local)
            data.f_skew[:, :] = pinocchio.skew(data.f_local.linear)
            data.fJf_df[:, :] = np.dot(data.f_skew, data.fJf[3:, :])
            data.dtau_dq[:, :] = -np.dot(data.fJf[:3, :].T, data.fJf_df)


class Contact3DModelDerived(crocoddyl.ContactModelAbstract):
    def __init__(
        self, state, id, xref, type=pinocchio.ReferenceFrame.LOCAL, gains=[0.0, 0.0]
    ):
        crocoddyl.ContactModelAbstract.__init__(self, state, type, 3)
        self.id = id
        self.xref = xref
        self.gains = gains

    def calc(self, data, x):
        assert self.xref is not None or self.gains[0] == 0.0
        pinocchio.updateFramePlacement(self.state.pinocchio, data.pinocchio, self.id)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.pinocchio,
            self.id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        data.v = pinocchio.getFrameVelocity(
            self.state.pinocchio, data.pinocchio, self.id
        )
        data.a0_local[:] = pinocchio.getFrameClassicalAcceleration(
            self.state.pinocchio, data.pinocchio, self.id, pinocchio.LOCAL
        ).linear

        oRf = data.pinocchio.oMf[self.id].rotation
        data.vw[:] = data.v.angular
        data.vv[:] = data.v.linear
        data.dp[:] = data.pinocchio.oMf[self.id].translation - self.xref
        data.dp_local[:] = np.dot(oRf.T, data.dp)

        if self.gains[0] != 0.0:
            data.a0_local[:] += self.gains[0] * data.dp_local
        if self.gains[1] != 0.0:
            data.a0_local[:] += self.gains[1] * data.vv

        if self.type == pinocchio.LOCAL:
            data.Jc[:, :] = data.fJf[:3, :]
            data.a0[:] = data.a0_local
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.Jc[:, :] = np.dot(oRf, data.fJf[:3, :])
            data.a0[:] = np.dot(oRf, data.a0_local)

    def calcDiff(self, data, x):
        if tuple(int(i) for i in pinocchio.__version__.split(".")) >= (3, 0, 0):
            joint = self.state.pinocchio.frames[self.id].parentJoint
        else:
            joint = self.state.pinocchio.frames[self.id].parent
        (
            v_partial_dq,
            a_partial_dq,
            a_partial_dv,
            _,
        ) = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, joint, pinocchio.ReferenceFrame.LOCAL
        )
        nv = self.state.nv
        data.vv_skew[:, :] = pinocchio.skew(data.vv)
        data.vw_skew[:, :] = pinocchio.skew(data.vw)
        data.dp_skew[:, :] = pinocchio.skew(data.dp_local)
        data.fXjdv_dq[:] = np.dot(data.fXj, v_partial_dq)
        data.fXjda_dq[:] = np.dot(data.fXj, a_partial_dq)
        data.fXjda_dv[:] = np.dot(data.fXj, a_partial_dv)
        data.da0_local_dx[:, :nv] = data.fXjda_dq[:3, :]
        data.da0_local_dx[:, :nv] += np.dot(data.vw_skew, data.fXjdv_dq[:3, :])
        data.da0_local_dx[:, :nv] -= np.dot(data.vv_skew, data.fXjdv_dq[3:, :])
        data.da0_local_dx[:, nv:] = data.fXjda_dv[:3, :]
        data.da0_local_dx[:, nv:] += np.dot(data.vw_skew, data.fJf[:3, :])
        data.da0_local_dx[:, nv:] -= np.dot(data.vv_skew, data.fJf[3:, :])
        oRf = data.pinocchio.oMf[self.id].rotation

        if self.gains[0] != 0.0:
            data.da0_local_dx[:, :nv] += self.gains[0] * np.dot(
                data.dp_skew, data.fJf[3:, :]
            )
            data.da0_local_dx[:, :nv] += self.gains[0] * data.fJf[:3, :]
        if self.gains[1] != 0.0:
            data.da0_local_dx[:, :nv] += self.gains[1] * data.fXjdv_dq[:3, :]
            data.da0_local_dx[:, nv:] += self.gains[1] * data.fJf[:3, :]
        if self.type == pinocchio.LOCAL:
            data.da0_dx[:, :] = data.da0_local_dx
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.a0_skew[:, :] = pinocchio.skew(data.a0)
            data.da0_dx = np.dot(oRf, data.da0_local_dx)
            data.da0_dx[:, :nv] -= np.dot(np.dot(data.a0_skew, oRf), data.fJf[3:, :])

    def createData(self, data):
        data = Contact3DDataDerived(self, data)
        return data

    def updateForce(self, data, force):
        assert force.shape[0] == 3
        nv = self.state.nv
        data.f.linear = force
        data.f.angular = np.zeros(3)
        if self.type == pinocchio.LOCAL:
            data.fext = data.jMf.act(data.f)
            data.dtau_dq[:, :] = np.zeros((nv, nv))
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            oRf = data.pinocchio.oMf[self.id].rotation
            data.f_local.linear = np.dot(oRf.T, force)
            data.f_local.angular = np.zeros(3)
            data.fext = data.jMf.act(data.f_local)
            data.f_skew[:, :] = pinocchio.skew(data.f_local.linear)
            data.fJf_df[:, :] = np.dot(data.f_skew, data.fJf[3:, :])
            data.dtau_dq[:, :] = -np.dot(data.fJf[:3, :].T, data.fJf_df)


class Contact1DDataDerived(crocoddyl.ContactDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ContactDataAbstract.__init__(self, model, data)
        self.jMf = model.state.pinocchio.frames[model.id].placement
        self.fXj = self.jMf.inverse().action
        self.v = pinocchio.Motion.Zero()
        self.a0_local = np.zeros(3)
        self.vw = np.zeros(3)
        self.vv = np.zeros(3)
        self.dp = np.zeros(3)
        self.dp_local = np.zeros(3)
        self.f_local = pinocchio.Force.Zero()
        self.da0_local_dx = np.zeros((3, model.state.ndx))
        self.fJf = np.zeros((6, model.state.nv))
        self.vv_skew = np.zeros((3, 3))
        self.vw_skew = np.zeros((3, 3))
        self.a0_skew = np.zeros((3, 3))
        self.a0_world_skew = np.zeros((3, 3))
        self.dp_skew = np.zeros((3, 3))
        self.f_skew = np.zeros((3, 3))
        self.vw_skew = np.zeros((3, 3))
        self.fXjdv_dq = np.zeros((6, model.state.nv))
        self.fXjda_dq = np.zeros((6, model.state.nv))
        self.fXjda_dv = np.zeros((6, model.state.nv))
        self.fJf_df = np.zeros((3, model.state.nv))


class Contact3DDataDerived(crocoddyl.ContactDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ContactDataAbstract.__init__(self, model, data)
        self.jMf = model.state.pinocchio.frames[model.id].placement
        self.fXj = self.jMf.inverse().action
        self.v = pinocchio.Motion.Zero()
        self.a0_local = np.zeros(3)
        self.vw = np.zeros(3)
        self.vv = np.zeros(3)
        self.dp = np.zeros(3)
        self.dp_local = np.zeros(3)
        self.f_local = pinocchio.Force.Zero()
        self.da0_local_dx = np.zeros((3, model.state.ndx))
        self.fJf = np.zeros((6, model.state.nv))
        self.vv_skew = np.zeros((3, 3))
        self.vw_skew = np.zeros((3, 3))
        self.a0_skew = np.zeros((3, 3))
        self.a0_world_skew = np.zeros((3, 3))
        self.dp_skew = np.zeros((3, 3))
        self.f_skew = np.zeros((3, 3))
        self.vw_skew = np.zeros((3, 3))
        self.fXjdv_dq = np.zeros((6, model.state.nv))
        self.fXjda_dq = np.zeros((6, model.state.nv))
        self.fXjda_dv = np.zeros((6, model.state.nv))
        self.fJf_df = np.zeros((3, model.state.nv))


class Contact6DModelDerived(crocoddyl.ContactModelAbstract):
    def __init__(
        self, state, id, Mref, type=pinocchio.ReferenceFrame.LOCAL, gains=[0.0, 0.0]
    ):
        crocoddyl.ContactModelAbstract.__init__(self, state, type, 6)
        self.id = id
        self.Mref = Mref
        self.gains = gains

    def calc(self, data, x):
        assert self.Mref is not None or self.gains[0] == 0.0
        pinocchio.updateFramePlacement(self.state.pinocchio, data.pinocchio, self.id)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.pinocchio,
            self.id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        data.a0_local = pinocchio.getFrameAcceleration(
            self.state.pinocchio, data.pinocchio, self.id
        )
        if self.gains[0] != 0.0:
            data.rMf = self.Mref.actInv(data.pinocchio.oMf[self.id])
            data.a0_local += self.gains[0].item() * pinocchio.log6(data.rMf)
        if self.gains[1] != 0.0:
            data.v = pinocchio.getFrameVelocity(
                self.state.pinocchio, data.pinocchio, self.id
            )
            data.a0_local += self.gains[1].item() * data.v

        if self.type == pinocchio.LOCAL:
            data.Jc[:, :] = data.fJf
            data.a0[:] = data.a0_local.vector
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.lwaMl.rotation = data.pinocchio.oMf[self.id].rotation
            data.Jc[:, :] = np.dot(data.lwaMl.toActionMatrix(), data.fJf)
            data.a0[:] = data.lwaMl.act(data.a0_local).vector

    def calcDiff(self, data, x):
        if tuple(int(i) for i in pinocchio.__version__.split(".")) >= (3, 0, 0):
            joint = self.state.pinocchio.frames[self.id].parentJoint
        else:
            joint = self.state.pinocchio.frames[self.id].parent
        (
            v_partial_dq,
            a_partial_dq,
            a_partial_dv,
            a_partial_da,
        ) = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, joint, pinocchio.ReferenceFrame.LOCAL
        )

        nv = self.state.nv
        data.da0_local_dx[:, :nv] = np.dot(data.fXj, a_partial_dq)
        data.da0_local_dx[:, nv:] = np.dot(data.fXj, a_partial_dv)

        if self.gains[0] != 0.0:
            data.da0_local_dx[:, :nv] += self.gains[0].item() * np.dot(
                pinocchio.Jlog6(data.rMf), data.fJf
            )
        if self.gains[1] != 0.0:
            data.da0_local_dx[:, :nv] += self.gains[1].item() * np.dot(
                data.fXj, v_partial_dq
            )
            data.da0_local_dx[:, nv:] += self.gains[1].item() * data.fJf

        if self.type == pinocchio.LOCAL:
            data.da0_dx[:, :] = data.da0_local_dx
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            # Recalculate the constrained accelerations after imposing contact
            # constraints.
            # This is necessary for the forward-dynamics case.
            data.a0_local = pinocchio.getFrameAcceleration(
                self.state.pinocchio, data.pinocchio, self.id
            )
            if self.gains[0] != 0.0:
                data.a0_local += self.gains[0].item() * pinocchio.log6(data.rMf)
            if self.gains[1] != 0.0:
                data.a0_local += self.gains[1].item() * data.v
            data.a0[:] = data.lwaMl.act(data.a0_local).vector

            oRf = data.pinocchio.oMf[self.id].rotation
            data.av_skew[:, :] = pinocchio.skew(data.a0[:3])
            data.aw_skew[:, :] = pinocchio.skew(data.a0[3:])
            data.av_world_skew[:, :] = np.dot(data.av_skew, oRf)
            data.aw_world_skew[:, :] = np.dot(data.aw_skew, oRf)
            data.da0_dx[:, :] = np.dot(data.lwaMl.toActionMatrix(), data.da0_local_dx)
            data.da0_dx[:3, :nv] -= np.dot(data.av_world_skew, data.fJf[3:, :])
            data.da0_dx[3:, :nv] -= np.dot(data.aw_world_skew, data.fJf[3:, :])

    def createData(self, data):
        data = Contact6DDataDerived(self, data)
        return data

    def updateForce(self, data, force):
        assert force.shape[0] == 6
        nv = self.state.nv
        data.f = pinocchio.Force(force)
        if self.type == pinocchio.LOCAL:
            data.fext = data.jMf.act(data.f)
            data.dtau_dq[:, :] = np.zeros((nv, nv))
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.f_local = data.lwaMl.actInv(data.f)
            data.fext = data.jMf.act(data.f_local)
            data.fv_skew[:, :] = pinocchio.skew(data.f_local.linear)
            data.fw_skew[:, :] = pinocchio.skew(data.f_local.angular)
            data.fJf_df[:3, :] = np.dot(data.fv_skew, data.fJf[3:, :])
            data.fJf_df[3:, :] = np.dot(data.fw_skew, data.fJf[3:, :])
            data.dtau_dq[:, :] = -np.dot(data.fJf.T, data.fJf_df)


class Contact6DDataDerived(crocoddyl.ContactDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ContactDataAbstract.__init__(self, model, data)
        self.jMf = model.state.pinocchio.frames[model.id].placement
        self.fXj = self.jMf.inverse().action
        self.fMf = pinocchio.SE3.Identity()
        self.lwaMl = pinocchio.SE3.Identity()
        self.v = pinocchio.Motion.Zero()
        self.a0_local = pinocchio.Motion.Zero()
        self.f_local = pinocchio.Force.Zero()
        self.da0_local_dx = np.zeros((6, model.state.ndx))
        self.fJf = np.zeros((6, model.state.nv))
        self.av_world_skew = np.zeros((3, 3))
        self.aw_world_skew = np.zeros((3, 3))
        self.av_skew = np.zeros((3, 3))
        self.aw_skew = np.zeros((3, 3))
        self.fv_skew = np.zeros((3, 3))
        self.fw_skew = np.zeros((3, 3))
        self.fJf_df = np.zeros((6, model.state.nv))


class Impulse3DModelDerived(crocoddyl.ImpulseModelAbstract):
    def __init__(self, state, frame, type=pinocchio.ReferenceFrame.LOCAL):
        crocoddyl.ImpulseModelAbstract.__init__(self, state, type, 3)
        self.id = frame

    def calc(self, data, x):
        pinocchio.updateFramePlacement(self.state.pinocchio, data.pinocchio, self.id)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.pinocchio,
            self.id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        if self.type == pinocchio.LOCAL:
            data.Jc[:, :] = data.fJf[:3, :]
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.Jc[:, :] = np.dot(
                data.pinocchio.oMf[self.id].rotation, data.fJf[:3, :]
            )

    def calcDiff(self, data, x):
        if tuple(int(i) for i in pinocchio.__version__.split(".")) >= (3, 0, 0):
            joint = self.state.pinocchio.frames[self.id].parentJoint
        else:
            joint = self.state.pinocchio.frames[self.id].parent
        v_partial_dq, _ = pinocchio.getJointVelocityDerivatives(
            self.state.pinocchio, data.pinocchio, joint, pinocchio.ReferenceFrame.LOCAL
        )
        data.dv0_local_dq[:, :] = np.dot(data.fXj[:3, :], v_partial_dq)
        if self.type == pinocchio.LOCAL:
            data.dv0_dq[:, :] = data.dv0_local_dq
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            oRf = data.pinocchio.oMf[self.id].rotation
            data.v0[:] = pinocchio.getFrameVelocity(
                self.state.pinocchio,
                data.pinocchio,
                self.id,
                pinocchio.LOCAL_WORLD_ALIGNED,
            ).linear
            data.v0_skew[:, :] = pinocchio.skew(data.v0)
            data.v0_world_skew[:, :] = np.dot(data.v0_skew, oRf)
            data.dv0_dq[:, :] = np.dot(oRf, data.dv0_local_dq)
            data.dv0_dq[:, :] -= np.dot(data.v0_world_skew, data.fJf[3:, :])

    def createData(self, data):
        data = Impulse3DDataDerived(self, data)
        return data

    def updateForce(self, data, force):
        assert force.shape[0] == 3
        nv = self.state.nv
        data.f.linear = force
        data.f.angular = np.zeros(3)
        if self.type == pinocchio.LOCAL:
            data.fext = data.jMf.act(data.f)
            data.dtau_dq[:, :] = np.zeros((nv, nv))
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            oRf = data.pinocchio.oMf[self.id].rotation
            data.f_local.linear = np.dot(oRf.T, force)
            data.f_local.angular = np.zeros(3)
            data.fext = data.jMf.act(data.f_local)
            data.f_skew[:, :] = pinocchio.skew(data.f_local.linear)
            data.fJf_df[:, :] = np.dot(data.f_skew, data.fJf[3:, :])
            data.dtau_dq[:, :] = -np.dot(data.fJf[:3, :].T, data.fJf_df)


class Impulse3DDataDerived(crocoddyl.ImpulseDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ImpulseDataAbstract.__init__(self, model, data)
        self.jMf = model.state.pinocchio.frames[model.id].placement
        self.fXj = self.jMf.inverse().action
        self.v0 = np.zeros(3)
        self.f_local = pinocchio.Force.Zero()
        self.dv0_local_dq = np.zeros((3, model.state.nv))
        self.fJf = np.zeros((6, model.state.nv))
        self.v0_skew = np.zeros((3, 3))
        self.v0_world_skew = np.zeros((3, 3))
        self.f_skew = np.zeros((3, 3))
        self.fJf_df = np.zeros((3, model.state.nv))


class Impulse6DModelDerived(crocoddyl.ImpulseModelAbstract):
    def __init__(self, state, frame, type=pinocchio.ReferenceFrame.LOCAL):
        crocoddyl.ImpulseModelAbstract.__init__(self, state, type, 6)
        self.id = frame

    def calc(self, data, x):
        pinocchio.updateFramePlacement(self.state.pinocchio, data.pinocchio, self.id)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.pinocchio,
            self.id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        if self.type == pinocchio.LOCAL:
            data.Jc[:, :] = data.fJf
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.lwaMl.rotation = data.pinocchio.oMf[self.id].rotation
            data.Jc[:, :] = np.dot(data.lwaMl.toActionMatrix(), data.fJf)

    def calcDiff(self, data, x):
        if tuple(int(i) for i in pinocchio.__version__.split(".")) >= (3, 0, 0):
            joint = self.state.pinocchio.frames[self.id].parentJoint
        else:
            joint = self.state.pinocchio.frames[self.id].parent
        v_partial_dq, _ = pinocchio.getJointVelocityDerivatives(
            self.state.pinocchio, data.pinocchio, joint, pinocchio.ReferenceFrame.LOCAL
        )
        data.dv0_local_dq[:, :] = np.dot(data.fXj, v_partial_dq)
        if self.type == pinocchio.LOCAL:
            data.dv0_dq[:, :] = data.dv0_local_dq
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            oRf = data.pinocchio.oMf[self.id].rotation
            data.v0 = pinocchio.getFrameVelocity(
                self.state.pinocchio, data.pinocchio, self.id, self.type
            )
            data.vv_skew[:, :] = pinocchio.skew(data.v0.linear)
            data.vw_skew[:, :] = pinocchio.skew(data.v0.angular)
            data.vv_world_skew[:, :] = np.dot(data.vv_skew, oRf)
            data.vw_world_skew[:, :] = np.dot(data.vw_skew, oRf)
            data.dv0_dq[:, :] = np.dot(data.lwaMl.toActionMatrix(), data.dv0_local_dq)
            data.dv0_dq[:3, :] -= np.dot(data.vv_world_skew, data.fJf[3:, :])
            data.dv0_dq[3:, :] -= np.dot(data.vw_world_skew, data.fJf[3:, :])

    def createData(self, data):
        data = Impulse6DDataDerived(self, data)
        return data

    def updateForce(self, data, force):
        assert force.shape[0] == 6
        nv = self.state.nv
        data.f = pinocchio.Force(force)
        if self.type == pinocchio.LOCAL:
            data.fext = data.jMf.act(data.f)
            data.dtau_dq[:, :] = np.zeros((nv, nv))
        if self.type == pinocchio.WORLD or self.type == pinocchio.LOCAL_WORLD_ALIGNED:
            data.f_local = data.lwaMl.actInv(data.f)
            data.fext = data.jMf.act(data.f_local)
            data.fv_skew[:, :] = pinocchio.skew(data.f_local.linear)
            data.fw_skew[:, :] = pinocchio.skew(data.f_local.angular)
            data.fJf_df[:3, :] = np.dot(data.fv_skew, data.fJf[3:, :])
            data.fJf_df[3:, :] = np.dot(data.fw_skew, data.fJf[3:, :])
            data.dtau_dq[:, :] = -np.dot(data.fJf.T, data.fJf_df)


class Impulse6DDataDerived(crocoddyl.ImpulseDataAbstract):
    def __init__(self, model, data):
        crocoddyl.ImpulseDataAbstract.__init__(self, model, data)
        self.jMf = model.state.pinocchio.frames[model.id].placement
        self.fXj = self.jMf.inverse().action
        self.lwaMl = pinocchio.SE3.Identity()
        self.v0 = pinocchio.Motion.Zero()
        self.f_local = pinocchio.Force.Zero()
        self.dv0_local_dq = np.zeros((6, model.state.nv))
        self.fJf = np.zeros((6, model.state.nv))
        self.vv_skew = np.zeros((3, 3))
        self.vw_skew = np.zeros((3, 3))
        self.vv_world_skew = np.zeros((3, 3))
        self.vw_world_skew = np.zeros((3, 3))
        self.fv_skew = np.zeros((3, 3))
        self.fw_skew = np.zeros((3, 3))
        self.fJf_df = np.zeros((6, model.state.nv))


class DDPDerived(crocoddyl.SolverAbstract):
    def __init__(self, shootingProblem):
        crocoddyl.SolverAbstract.__init__(self, shootingProblem)
        self.allocateData()  # TODO remove it?

        self.isFeasible = False
        self.alphas = [2 ** (-n) for n in range(10)]
        self.th_grad = 1e-12

        self.callbacks = None
        self.preg = 0
        self.dreg = 0
        self.reg_incFactor = 10
        self.reg_decFactor = 10
        self.reg_max = 1e9
        self.reg_min = 1e-9
        self.th_step = 0.5

    def solve(
        self, init_xs=[], init_us=[], maxiter=100, isFeasible=False, regInit=None
    ):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.preg = regInit if regInit is not None else self.reg_min
        self.dreg = regInit if regInit is not None else self.reg_min
        self.wasFeasible = False
        for i in range(maxiter):
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    recalc = False
                    self.increaseRegularization()
                    if self.preg == self.reg_max:
                        return self.xs, self.us, False
                    else:
                        continue
                break
            self.d = self.expectedImprovement()
            d1, d2 = self.d[0], self.d[1]

            for a in self.alphas:
                try:
                    self.dV = self.tryStep(a)
                except ArithmeticError:
                    continue
                self.dV_exp = a * (d1 + 0.5 * d2 * a)
                if self.dV_exp >= 0:
                    if (
                        d1 < self.th_grad
                        or not self.isFeasible
                        or self.dV > self.th_acceptStep * self.dV_exp
                    ):
                        # Accept step
                        self.wasFeasible = self.isFeasible
                        self.setCandidate(self.xs_try, self.us_try, True)
                        self.cost = self.cost_try
                        break
            if a > self.th_step:
                self.decreaseRegularization()
            if a == self.alphas[-1]:
                self.increaseRegularization()
                if self.preg == self.reg_max:
                    return self.xs, self.us, False
            self.stepLength = a
            self.iter = i
            self.stop = self.stoppingCriteria()
            if self.callbacks is not None:
                [c(self) for c in self.callbacks]

            if self.wasFeasible and self.stop < self.th_stop:
                return self.xs, self.us, True
        return self.xs, self.us, False

    def computeDirection(self, recalc=True):
        if recalc:
            self.calcDiff()
        self.backwardPass()
        return [np.nan] * (self.problem.T + 1), self.k, self.Vx

    def tryStep(self, stepLength=1):
        self.forwardPass(stepLength)
        return self.cost - self.cost_try

    def stoppingCriteria(self):
        return np.abs(self.d[0] + 0.5 * self.d[1])

    def expectedImprovement(self):
        d1 = sum([np.dot(q.T, k) for q, k in zip(self.Qu, self.k)])
        d2 = sum([-np.dot(k.T, np.dot(q, k)) for q, k in zip(self.Quu, self.k)])
        return np.array([d1, d2])

    def calcDiff(self):
        if self.iter == 0:
            self.problem.calc(self.xs, self.us)
        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.fs[0] = self.problem.runningModels[0].state.diff(
                self.xs[0], self.problem.x0
            )
            for i, (m, d, x) in enumerate(
                zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])
            ):
                self.fs[i + 1] = m.state.diff(x, d.xnext)
        return self.cost

    def backwardPass(self):
        self.Vx[-1][:] = self.problem.terminalData.Lx
        self.Vxx[-1][:, :] = self.problem.terminalData.Lxx

        if self.preg != 0:
            ndx = self.problem.terminalModel.state.ndx
            self.Vxx[-1][range(ndx), range(ndx)] += self.preg

        # Compute and store the Vx gradient at end of the interval (rollout state)
        if not self.isFeasible:
            self.Vx[-1] += np.dot(self.Vxx[-1], self.fs[-1])

        for t, (model, data) in rev_enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.Qxx[t][:, :] = data.Lxx + np.dot(
                data.Fx.T, np.dot(self.Vxx[t + 1], data.Fx)
            )
            self.Qxu[t][:, :] = data.Lxu + np.dot(
                data.Fx.T, np.dot(self.Vxx[t + 1], data.Fu)
            )
            self.Quu[t][:, :] = data.Luu + np.dot(
                data.Fu.T, np.dot(self.Vxx[t + 1], data.Fu)
            )
            self.Qx[t][:] = data.Lx + np.dot(data.Fx.T, self.Vx[t + 1])
            self.Qu[t][:] = data.Lu + np.dot(data.Fu.T, self.Vx[t + 1])

            if self.preg != 0:
                self.Quu[t][range(model.nu), range(model.nu)] += self.preg

            self.computeGains(t)

            self.Vx[t][:] = self.Qx[t] - np.dot(self.K[t].T, self.Qu[t])
            self.Vxx[t][:, :] = self.Qxx[t] - np.dot(self.Qxu[t], self.K[t])
            self.Vxx[t][:, :] = 0.5 * (
                self.Vxx[t][:, :] + self.Vxx[t][:, :].T
            )  # ensure symmetric

            if self.preg != 0:
                self.Vxx[t][range(model.state.ndx), range(model.state.ndx)] += self.preg

            # Compute and store the Vx gradient at end of the interval (rollout state)
            if not self.isFeasible:
                self.Vx[t] += np.dot(self.Vxx[t], self.fs[t])

            raiseIfNan(self.Vxx[t], ArithmeticError("backward error"))
            raiseIfNan(self.Vx[t], ArithmeticError("backward error"))

    def forwardPass(self, stepLength, warning="ignore"):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        for t, (m, d) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            utry[t] = (
                us[t]
                - self.k[t] * stepLength
                - np.dot(self.K[t], m.state.diff(xs[t], xtry[t]))
            )
            with warnings.catch_warnings():
                warnings.simplefilter(warning)
                m.calc(d, xtry[t], utry[t])
                xnext, cost = d.xnext, d.cost
            xtry[t + 1] = xnext.copy()  # not sure copy helpful here.
            ctry += cost
            raiseIfNan([ctry, cost], ArithmeticError("forward error"))
            raiseIfNan(xtry[t + 1], ArithmeticError("forward error"))
        with warnings.catch_warnings():
            warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, ArithmeticError("forward error"))
        self.cost_try = ctry
        return xtry, utry, ctry

    def computeGains(self, t):
        try:
            if self.Quu[t].shape[0] > 0:
                Lb = scl.cho_factor(self.Quu[t])
                self.K[t][:, :] = scl.cho_solve(Lb, self.Qux[t])
                self.k[t][:] = scl.cho_solve(Lb, self.Qu[t])
            else:
                pass
        except scl.LinAlgError:
            raise ArithmeticError("backward error")

    def increaseRegularization(self):
        self.preg *= self.reg_incFactor
        if self.preg > self.reg_max:
            self.preg = self.reg_max
        self.dreg = self.preg

    def decreaseRegularization(self):
        self.preg /= self.reg_decFactor
        if self.preg < self.reg_min:
            self.preg = self.reg_min
        self.dreg = self.preg

    def allocateData(self):
        models = [*self.problem.runningModels.tolist(), self.problem.terminalModel]
        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in models]
        self.Vx = [np.zeros([m.state.ndx]) for m in models]

        self.Q = [
            np.zeros([m.state.ndx + m.nu, m.state.ndx + m.nu])
            for m in self.problem.runningModels
        ]
        self.q = [np.zeros([m.state.ndx + m.nu]) for m in self.problem.runningModels]
        self.Qxx = [
            Q[: m.state.ndx, : m.state.ndx]
            for m, Q in zip(self.problem.runningModels, self.Q)
        ]
        self.Qxu = [
            Q[: m.state.ndx, m.state.ndx :]
            for m, Q in zip(self.problem.runningModels, self.Q)
        ]
        self.Qux = [Qxu.T for m, Qxu in zip(self.problem.runningModels, self.Qxu)]
        self.Quu = [
            Q[m.state.ndx :, m.state.ndx :]
            for m, Q in zip(self.problem.runningModels, self.Q)
        ]
        self.Qx = [q[: m.state.ndx] for m, q in zip(self.problem.runningModels, self.q)]
        self.Qu = [q[m.state.ndx :] for m, q in zip(self.problem.runningModels, self.q)]

        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]

        self.xs_try = [self.problem.x0] + [np.nan * self.problem.x0] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)] + [
            np.zeros(m.state.ndx) for m in self.problem.runningModels
        ]


class FDDPDerived(DDPDerived):
    def __init__(self, shootingProblem):
        DDPDerived.__init__(self, shootingProblem)

        self.th_acceptNegStep = 2.0
        self.dg = 0.0
        self.dq = 0.0
        self.dv = 0.0

    def solve(
        self, init_xs=[], init_us=[], maxiter=100, isFeasible=False, regInit=None
    ):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.preg = regInit if regInit is not None else self.reg_min
        self.dreg = regInit if regInit is not None else self.reg_min
        self.wasFeasible = False
        for i in range(maxiter):
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    recalc = False
                    self.increaseRegularization()
                    if self.preg == self.reg_max:
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
                self.d = self.expectedImprovement()
                d1, d2 = self.d[0], self.d[1]

                self.dV_exp = a * (d1 + 0.5 * d2 * a)
                if self.dV_exp >= 0.0:  # descend direction
                    if d1 < self.th_grad or self.dV > self.th_acceptStep * self.dV_exp:
                        self.wasFeasible = self.isFeasible
                        self.setCandidate(
                            self.xs_try, self.us_try, (self.wasFeasible or a == 1)
                        )
                        self.cost = self.cost_try
                        break
                else:
                    # reducing the gaps by allowing a small increment in the cost value
                    if self.dV > self.th_acceptNegStep * self.dV_exp:
                        self.wasFeasible = self.isFeasible
                        self.setCandidate(
                            self.xs_try, self.us_try, (self.wasFeasible or a == 1)
                        )
                        self.cost = self.cost_try
                        break
            if a > self.th_step:
                self.decreaseRegularization()
            if a == self.alphas[-1]:
                self.increaseRegularization()
                if self.preg == self.reg_max:
                    return self.xs, self.us, False
            self.stepLength = a
            self.iter = i
            self.stop = self.stoppingCriteria()
            if self.callbacks is not None:
                [c(self) for c in self.callbacks]

            if self.wasFeasible and self.stop < self.th_stop:
                return self.xs, self.us, True
        return self.xs, self.us, False

    def computeDirection(self, recalc=True):
        if recalc:
            self.calcDiff()
        self.backwardPass()
        return [np.nan] * (self.problem.T + 1), self.k, self.Vx

    def tryStep(self, stepLength=1):
        self.forwardPass(stepLength)
        return self.cost - self.cost_try

    def updateExpectedImprovement(self):
        self.dg = 0.0
        self.dq = 0.0
        if not self.isFeasible:
            self.dg -= np.dot(self.Vx[-1].T, self.fs[-1])
            self.dq += np.dot(self.fs[-1].T, np.dot(self.Vxx[-1], self.fs[-1]))
        for t in range(self.problem.T):
            self.dg += np.dot(self.Qu[t].T, self.k[t])
            self.dq -= np.dot(self.k[t].T, np.dot(self.Quu[t], self.k[t]))
            if not self.isFeasible:
                self.dg -= np.dot(self.Vx[t].T, self.fs[t])
                self.dq += np.dot(self.fs[t].T, np.dot(self.Vxx[t], self.fs[t]))

    def expectedImprovement(self):
        self.dv = 0.0
        if not self.isFeasible:
            dx = self.problem.runningModels[-1].state.diff(self.xs_try[-1], self.xs[-1])
            self.dv -= np.dot(self.fs[-1].T, np.dot(self.Vxx[-1], dx))
            for t in range(self.problem.T):
                dx = self.problem.runningModels[t].state.diff(
                    self.xs_try[t], self.xs[t]
                )
                self.dv -= np.dot(self.fs[t].T, np.dot(self.Vxx[t], dx))
        d1 = self.dg + self.dv
        d2 = self.dq - 2 * self.dv
        return np.array([d1, d2])

    def calcDiff(self):
        self.cost = self.problem.calc(self.xs, self.us)
        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.fs[0] = self.problem.runningModels[0].state.diff(
                self.xs[0], self.problem.x0
            )
            for i, (m, d, x) in enumerate(
                zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])
            ):
                self.fs[i + 1] = m.state.diff(x, d.xnext)
        elif not self.wasFeasible:
            self.fs[:] = [np.zeros_like(f) for f in self.fs]
        return self.cost

    def forwardPass(self, stepLength, warning="ignore"):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        ctry = 0
        xnext = self.problem.x0
        for t, (m, d) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            if self.isFeasible or stepLength == 1:
                xtry[t] = xnext.copy()
            else:
                xtry[t] = m.state.integrate(xnext, self.fs[t] * (stepLength - 1))
            utry[t] = (
                us[t]
                - self.k[t] * stepLength
                - np.dot(self.K[t], m.state.diff(xs[t], xtry[t]))
            )
            with warnings.catch_warnings():
                warnings.simplefilter(warning)
                m.calc(d, xtry[t], utry[t])
                xnext, cost = d.xnext, d.cost
            ctry += cost
            raiseIfNan([ctry, cost], ArithmeticError("forward error"))
            raiseIfNan(xnext, ArithmeticError("forward error"))
        if self.isFeasible or stepLength == 1:
            xtry[-1] = xnext.copy()
        else:
            xtry[-1] = self.problem.terminalModel.state.integrate(
                xnext, self.fs[-1] * (stepLength - 1)
            )
        with warnings.catch_warnings():
            warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, ArithmeticError("forward error"))
        self.cost_try = ctry
        return xtry, utry, ctry
