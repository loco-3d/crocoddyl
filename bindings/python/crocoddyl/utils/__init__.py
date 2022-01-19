import crocoddyl
import pinocchio
import numpy as np
import scipy.linalg as scl


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
        return np.matrix(np.random.rand(self.nx)).T

    def diff(self, x0, x1):
        return x1 - x0

    def integrate(self, x, dx):
        return x + dx

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [self.Jdiff(x1, x2, crocoddyl.Jcomponent.first), self.Jdiff(x1, x2, crocoddyl.Jcomponent.second)]

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
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second)
            ]
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

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [self.Jdiff(x1, x2, crocoddyl.Jcomponent.first), self.Jdiff(x1, x2, crocoddyl.Jcomponent.second)]

        if firstsecond == crocoddyl.Jcomponent.first:
            dx = self.diff(x2, x1)
            q = x2[:self.model.nq]
            dq = dx[:self.model.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            dx = self.diff(x1, x2)
            q = x1[:self.nq]
            dq = dx[:self.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second)
            ]

        q = x[:self.nq]
        dq = dx[:self.nv]
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
        data.u = 0.5 * (self.u_lb + np.power(a + np.power(s - self.u_lb, 2), 0.5)) + 0.5 * (
            self.u_ub - np.power(a + np.power(s - self.u_ub, 2), 0.5))

    def calcDiff(self, data, s):
        a = np.power(self.smooth * (self.u_ub - self.u_lb), 2)
        du_ds = 0.5 * (np.multiply(np.power(a + np.power((s - self.u_lb), 2), -0.5),
                                   (s - self.u_lb)) - np.multiply(np.power(a + np.power((s - self.u_ub), 2), -0.5),
                                                                  (s - self.u_ub)))
        np.fill_diagonal(data.du_ds, du_ds)


class FreeFloatingActuationDerived(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        assert (state.pinocchio.joints[1].shortname() == 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv - 6)

    def calc(self, data, x, u):
        data.tau[:] = np.hstack([np.zeros(6), u])

    def calcDiff(self, data, x, u):
        data.dtau_du[:, :] = np.vstack([np.zeros((6, self.nu)), np.eye(self.nu)])


class FullActuationDerived(crocoddyl.ActuationModelAbstract):

    def __init__(self, state):
        assert (state.pinocchio.joints[1].shortname() != 'JointModelFreeFlyer')
        crocoddyl.ActuationModelAbstract.__init__(self, state, state.nv)

    def calc(self, data, x, u):
        data.tau[:] = u

    def calcDiff(self, data, x, u):
        data.dtau_du[:, :] = pinocchio.utils.eye(self.nu)


class UnicycleModelDerived(crocoddyl.ActionModelAbstract):

    def __init__(self):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.dt = .1
        self.costWeights = [10., 1.]

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        v, w = u
        px, py, theta = x
        c, s, dt = np.cos(theta), np.sin(theta), self.dt
        # Rollout the dynamics
        data.xnext[0] = px + c * v * dt
        data.xnext[1] = py + s * v * dt
        data.xnext[2] = theta + w * dt
        # Compute the cost value
        data.r[:3] = self.costWeights[0] * x
        data.r[3:] = self.costWeights[1] * u
        data.cost = .5 * sum(data.r**2)

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        v = u[0]
        theta = x[2]
        # Cost derivatives
        data.Lx[:] = x * ([self.costWeights[0]**2] * self.state.nx)
        data.Lu[:] = u * ([self.costWeights[1]**2] * self.nu)
        # Dynamic derivatives
        c, s, dt = np.cos(theta), np.sin(theta), self.dt
        data.Fx[0, 2] = -s * v * dt
        data.Fx[1, 2] = c * v * dt
        data.Fu[0, 0] = c * dt
        data.Fu[1, 0] = s * dt
        data.Fu[2, 1] = dt

    def createData(self):
        data = UnicycleDataDerived(self)
        return data


class UnicycleDataDerived(crocoddyl.ActionDataAbstract):

    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        nx, nu = model.state.nx, model.nu
        self.Lxx[range(nx), range(nx)] = [model.costWeights[0]**2] * nx
        self.Luu[range(nu), range(nu)] = [model.costWeights[1]**2] * nu
        self.Fx[0, 0] = 1
        self.Fx[1, 1] = 1
        self.Fx[2, 2] = 1


class LQRModelDerived(crocoddyl.ActionModelAbstract):

    def __init__(self, nx, nu, driftFree=True):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(nx), nu)

        self.Fx = np.eye(self.state.nx)
        self.Fu = np.eye(self.state.nx)[:, :self.nu]
        self.f0 = np.zeros(self.state.nx)
        self.Lxx = np.eye(self.state.nx)
        self.Lxu = np.eye(self.state.nx)[:, :self.nu]
        self.Luu = np.eye(self.nu)
        self.lx = np.ones(self.state.nx)
        self.lu = np.ones(self.nu)

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        data.xnext[:] = np.dot(self.Fx, x) + np.dot(self.Fu, u) + self.f0
        data.cost = 0.5 * np.dot(x.T, np.dot(self.Lxx, x))
        data.cost += 0.5 * np.dot(u.T, np.dot(self.Luu, u))
        data.cost += np.dot(x.T, np.dot(self.Lxu, u))
        data.cost += np.dot(self.lx.T, x) + np.dot(self.lu.T, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        data.Lx[:] = self.lx + np.dot(self.Lxx, x) + np.dot(self.Lxu, u)
        data.Lu[:] = self.lu + np.dot(self.Lxu.T, x) + np.dot(self.Luu, u)

    def createData(self):
        data = LQRDataDerived(self)
        return data


class LQRDataDerived(crocoddyl.ActionDataAbstract):

    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        self.Fx[:, :] = model.Fx
        self.Fu[:, :] = model.Fu
        self.Lxx[:, :] = model.Lxx
        self.Luu[:, :] = model.Luu
        self.Lxu[:, :] = model.Lxu


class DifferentialLQRModelDerived(crocoddyl.DifferentialActionModelAbstract):

    def __init__(self, nq, nu, driftFree=True):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, crocoddyl.StateVector(2 * nq), nu)

        self.Fq = np.eye(self.state.nq)
        self.Fv = np.eye(self.state.nv)
        self.Fu = np.eye(self.state.nq)[:, :self.nu]
        self.f0 = np.zeros(self.state.nq)
        self.Lxx = np.eye(self.state.nx)
        self.Lxu = np.eye(self.state.nx)[:, :self.nu]
        self.Luu = np.eye(self.nu)
        self.lx = np.ones(self.state.nx)
        self.lu = np.ones(self.nu)

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[:self.state.nq], x[self.state.nq:]
        data.xout[:] = np.dot(self.Fq, q) + np.dot(self.Fv, v) + np.dot(self.Fu, u) + self.f0
        data.cost = 0.5 * np.dot(x.T, np.dot(self.Lxx, x))
        data.cost += 0.5 * np.dot(u.T, np.dot(self.Luu, u))
        data.cost += np.dot(x.T, np.dot(self.Lxu, u))
        data.cost += np.dot(self.lx.T, x) + np.dot(self.lu.T, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = self.unone
        data.Lx[:] = self.lx + np.dot(self.Lxx, x) + np.dot(self.Lxu, u)
        data.Lu[:] = self.lu + np.dot(self.Lxu.T, x) + np.dot(self.Luu, u)

    def createData(self):
        data = DifferentialLQRDataDerived(self)
        return data


class DifferentialLQRDataDerived(crocoddyl.DifferentialActionDataAbstract):

    def __init__(self, model):
        crocoddyl.DifferentialActionDataAbstract.__init__(self, model)
        self.Lxx[:, :] = model.Lxx
        self.Luu[:, :] = model.Luu
        self.Lxu[:, :] = model.Lxu
        self.Fx[:, :] = np.hstack([model.Fq, model.Fv])
        self.Fu[:, :] = model.Fu


class DifferentialFreeFwdDynamicsModelDerived(crocoddyl.DifferentialActionModelAbstract):

    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu, costModel.nr)
        self.actuation = actuationModel
        self.costs = costModel
        self.enable_force = True
        self.armature = np.matrix(np.zeros(0))

    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        q, v = x[:self.state.nq], x[-self.state.nv:]
        self.actuation.calc(data.actuation, x, u)
        tau = data.actuation.tau
        # Computing the dynamics using ABA or manually for armature case
        if self.enable_force:
            data.xout[:] = pinocchio.aba(self.state.pinocchio, data.pinocchio, q, v, tau)
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
            u = self.unone
        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[-nv:]
        # Computing the actuation derivatives
        self.actuation.calcDiff(data.actuation, x, u)
        tau = data.actuation.tau
        # Computing the dynamics derivatives
        if self.enable_force:
            pinocchio.computeABADerivatives(self.state.pinocchio, data.pinocchio, q, v, tau)
            ddq_dq = data.pinocchio.ddq_dq
            ddq_dv = data.pinocchio.ddq_dv
            data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv]) + np.dot(data.pinocchio.Minv, data.actuation.dtau_dx)
            data.Fu[:, :] = np.dot(data.pinocchio.Minv, data.actuation.dtau_du)
        else:
            pinocchio.computeRNEADerivatives(self.state.pinocchio, data.pinocchio, q, v, data.xout)
            ddq_dq = np.dot(data.Minv, (data.actuation.dtau_dx[:, :nv] - data.pinocchio.dtau_dq))
            ddq_dv = np.dot(data.Minv, (data.actuation.dtau_dx[:, nv:] - data.pinocchio.dtau_dv))
            data.Fx[:, :] = np.hstack([ddq_dq, ddq_dv])
            data.Fu[:, :] = np.dot(data.Minv, data.actuation.dtau_du)
        # Computing the cost derivatives
        self.costs.calcDiff(data.costs, x, u)

    def createData(self):
        data = DifferentialFreeFwdDynamicsDataDerived(self)
        return data

    def set_armature(self, armature):
        if armature.size is not self.state.nv:
            print('The armature dimension is wrong, we cannot set it.')
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

    def calcDiff(self, data, x, u=None):
        nv, dt = self.state.nv, self.timeStep
        self.differential.calcDiff(data.differential, x, u)
        dxnext_dx, dxnext_ddx = self.state.Jintegrate(x, data.dx)
        da_dx, da_du = data.differential.Fx, data.differential.Fu
        ddx_dx = np.vstack([da_dx * dt, da_dx])
        ddx_dx[range(nv), range(nv, 2 * nv)] += 1
        data.Fx[:, :] = dxnext_dx + dt * np.dot(dxnext_ddx, ddx_dx)
        ddx_du = np.vstack([da_du * dt, da_du])
        data.Fu[:, :] = dt * np.dot(dxnext_ddx, ddx_du)
        data.Lx[:] = data.differential.Lx
        data.Lu[:] = data.differential.Lu
        data.Lxx[:, :] = data.differential.Lxx
        data.Lxu[:, :] = data.differential.Lxu
        data.Luu[:, :] = data.differential.Luu

    def createData(self):
        data = IntegratedActionDataEulerDerived(self)
        return data


class IntegratedActionDataEulerDerived(crocoddyl.ActionDataAbstract):

    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        self.differential = model.differential.createData(self)


class IntegratedActionModelRK4Derived(crocoddyl.ActionModelAbstract):

    def __init__(self, diffModel, timeStep=1e-3, withCostResiduals=True):
        crocoddyl.ActionModelAbstract.__init__(self, diffModel.state, diffModel.nu, diffModel.nr)
        self.differential = diffModel
        self.timeStep = timeStep
        self.rk4_inc = [0.5, 0.5, 1.]
        self.nx = self.differential.state.nx
        self.ndx = self.differential.state.ndx
        self.nq = self.differential.state.nq
        self.nv = self.differential.state.nv
        self.enable_integration = (self.timeStep > 0.)

    def createData(self):
        return IntegratedActionDataRK4Derived(self)

    def calc(self, data, x, u=None):
        nq, dt = self.nq, self.timeStep

        data.y[0] = x
        for i in range(3):
            self.differential.calc(data.differential[i], data.y[i], u)
            data.acc[i] = data.differential[i].xout
            data.int[i] = data.differential[i].cost
            data.ki[i] = np.concatenate([data.y[i][nq:], data.acc[i]])
            data.y[i + 1] = self.differential.state.integrate(x, data.ki[i] * self.rk4_inc[i] * dt)

        self.differential.calc(data.differential[3], data.y[3], u)
        data.acc[3] = data.differential[3].xout
        data.int[3] = data.differential[3].cost
        data.ki[3] = np.concatenate([data.y[3][nq:], data.acc[3]])
        if (self.enable_integration):
            data.dx = (data.ki[0] + 2. * data.ki[1] + 2. * data.ki[2] + data.ki[3]) * dt / 6
            data.xnext = self.differential.state.integrate(x, data.dx)
            data.cost = (data.int[0] + 2 * data.int[1] + 2 * data.int[2] + data.int[3]) * dt / 6
        else:
            data.dx = np.zeros([self.ndx])
            data.xnext = x
            data.cost = data.differential[0].cost

        return data.xnext, data.cost

    def calcDiff(self, data, x, u):
        ndx, nu, nv, dt = self.ndx, self.nu, self.nv, self.timeStep
        for i in range(4):
            self.differential.calcDiff(data.differential[i], data.y[i], u)
            data.dki_dy[i] = np.bmat([[np.zeros([nv, nv]), np.identity(nv)], [data.differential[i].Fx]])

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

        if (self.enable_integration):
            for i in range(1, 4):
                c = self.rk4_inc[i - 1] * dt
                dyi_dx, dyi_ddx = self.state.Jintegrate(x, c * data.ki[i - 1])

                # ---------Finding the derivative wrt u--------------
                data.dy_du[i] = c * np.dot(dyi_ddx, data.dki_du[i - 1])
                data.dki_du[i] = np.vstack([
                    c * data.dki_du[i - 1][nv:, :],
                    data.differential[i].Fu + np.dot(data.differential[i].Fx, data.dy_du[i])
                ])

                data.dli_du[i] = data.differential[i].Lu + np.dot(data.differential[i].Lx, data.dy_du[i])

                data.Luu_partialx[i] = np.dot(data.differential[i].Lxu.T, data.dy_du[i])
                data.ddli_ddu[i] = data.differential[i].Luu + data.Luu_partialx[i].T + data.Luu_partialx[i] + np.dot(
                    data.dy_du[i].T, np.dot(data.differential[i].Lxx, data.dy_du[i]))

                # ---------Finding the derivative wrt x--------------
                data.dy_dx[i] = dyi_dx + c * np.dot(dyi_ddx, data.dki_dx[i - 1])
                data.dki_dx[i] = np.dot(data.dki_dy[i], data.dy_dx[i])

                data.dli_dx[i] = np.dot(data.differential[i].Lx, data.dy_dx[i])
                data.ddli_ddx[i] = np.dot(data.dy_dx[i].T, np.dot(data.differential[i].Lxx, data.dy_dx[i]))
                data.ddli_dxdu[i] = np.dot(data.dy_dx[i].T, data.differential[i].Lxu) + np.dot(
                    data.dy_dx[i].T, np.dot(data.differential[i].Lxx, data.dy_du[i]))

            dxnext_dx, dxnext_ddx = self.state.Jintegrate(x, data.dx)
            ddx_dx = (data.dki_dx[0] + 2. * data.dki_dx[1] + 2. * data.dki_dx[2] + data.dki_dx[3]) * dt / 6
            data.ddx_du = (data.dki_du[0] + 2. * data.dki_du[1] + 2. * data.dki_du[2] + data.dki_du[3]) * dt / 6
            data.Fx[:] = dxnext_dx + np.dot(dxnext_ddx, ddx_dx)
            data.Fu[:] = np.dot(dxnext_ddx, data.ddx_du)

            data.Lx[:] = (data.dli_dx[0] + 2. * data.dli_dx[1] + 2. * data.dli_dx[2] + data.dli_dx[3]) * dt / 6
            data.Lu[:] = (data.dli_du[0] + 2. * data.dli_du[1] + 2. * data.dli_du[2] + data.dli_du[3]) * dt / 6

            data.Lxx[:] = (data.ddli_ddx[0] + 2. * data.ddli_ddx[1] + 2. * data.ddli_ddx[2] +
                           data.ddli_ddx[3]) * dt / 6
            data.Luu[:] = (data.ddli_ddu[0] + 2. * data.ddli_ddu[1] + 2. * data.ddli_ddu[2] +
                           data.ddli_ddu[3]) * dt / 6
            data.Lxu[:] = (data.ddli_dxdu[0] + 2. * data.ddli_dxdu[1] + 2. * data.ddli_dxdu[2] +
                           data.ddli_dxdu[3]) * dt / 6
            data.Lux = data.Lxu.T
        else:
            data.Fx, _ = self.state.Jintegrate(x, data.dx)
            data.Fu = np.zeros([self.ndx, self.nu])
            data.Lu = data.differential[0].Lx
            data.Lu = data.differential[0].Lu
            data.Lxx = data.differential[0].Lxx
            data.Luu = data.differential[0].Luu
            data.Lxu = data.differential[0].Lxu


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
        self.dx = [np.zeros([ndx])] * 4
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
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(state.ndx)
        self.xref = xref if xref is not None else state.zero()
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u):
        data.residual.r[:] = self.state.diff(self.xref, x)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        data.residual.Rx[:] = self.state.Jdiff(self.xref, x, crocoddyl.Jcomponent.second)[0]
        self.activation.calcDiff(data.activation, data.residual.r)
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)
        data.Lxx[:, :] = np.dot(data.residual.Rx.T, np.dot(data.activation.Arr, data.residual.Rx))


class ControlCostModelDerived(crocoddyl.CostModelAbstract):

    def __init__(self, state, activation=None, uref=None, nu=None):
        nu = nu if nu is not None else state.nv
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(nu)
        self.uref = uref if uref is not None else pinocchio.utils.zero(nu)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u):
        data.residual.r[:] = u - self.uref
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        self.activation.calcDiff(data.activation, data.residual.r)
        data.Lu[:] = data.activation.Ar
        data.Luu[:, :] = data.activation.Arr


class CoMPositionCostModelDerived(crocoddyl.CostModelAbstract):

    def __init__(self, state, activation=None, cref=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.cref = cref

    def calc(self, data, x, u):
        data.residual.r[:] = data.shared.pinocchio.com[0] - self.cref
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [data.shared.pinocchio.Jcom,
             pinocchio.utils.zero((self.activation.nr, self.state.nv))])
        data.Lx[:] = np.hstack([np.dot(data.shared.pinocchio.Jcom.T, data.activation.Ar), np.zeros(self.state.nv)])
        data.Lxx[:, :] = np.vstack([
            np.hstack([
                np.dot(data.shared.pinocchio.Jcom.T, np.dot(data.activation.Arr, data.shared.pinocchio.Jcom)),
                np.zeros((self.state.nv, self.state.nv))
            ]),
            np.zeros((self.state.nv, self.state.ndx))
        ])


class FramePlacementCostModelDerived(crocoddyl.CostModelAbstract):

    def __init__(self, state, activation=None, frame_id=None, placement=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._placement = placement

    def calc(self, data, x, u):
        data.rMf = self._placement.inverse() * data.shared.pinocchio.oMf[self._frame_id]
        data.residual.r[:] = pinocchio.log(data.rMf).vector
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf[:, :] = pinocchio.Jlog6(data.rMf)
        data.fJf[:, :] = pinocchio.getFrameJacobian(self.state.pinocchio, data.shared.pinocchio, self._frame_id,
                                                    pinocchio.ReferenceFrame.LOCAL)
        data.J[:, :] = np.dot(data.rJf, data.fJf)
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack([data.J, np.zeros((self.activation.nr, self.state.nv))])
        data.Lx[:] = np.hstack([np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)])
        data.Lxx[:, :] = np.vstack([
            np.hstack(
                [np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                 np.zeros((self.state.nv, self.state.nv))]),
            np.zeros((self.state.nv, self.state.ndx))
        ])

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

    def __init__(self, state, activation=None, frame_id=None, translation=None, nu=None):
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._translation = translation

    def calc(self, data, x, u):
        data.residual.r[:] = data.shared.pinocchio.oMf[self._frame_id].translation - self._translation
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.R[:, :] = data.shared.pinocchio.oMf[self._frame_id].rotation
        data.J[:, :] = np.dot(
            data.R,
            pinocchio.getFrameJacobian(self.state.pinocchio, data.shared.pinocchio, self._frame_id,
                                       pinocchio.ReferenceFrame.LOCAL)[:3, :])
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack([data.J, np.zeros((self.activation.nr, self.state.nv))])
        data.Lx[:] = np.hstack([np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)])
        data.Lxx[:, :] = np.vstack([
            np.hstack(
                [np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                 np.zeros((self.state.nv, self.state.nv))]),
            np.zeros((self.state.nv, self.state.ndx))
        ])

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
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._rotation = rotation

    def calc(self, data, x, u):
        data.rRf[:, :] = np.dot(self._rotation.T, data.shared.pinocchio.oMf[self._frame_id].rotation)
        data.residual.r[:] = pinocchio.log3(data.rRf)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf[:, :] = pinocchio.Jlog3(data.rRf)
        data.fJf[:, :] = pinocchio.getFrameJacobian(self.state.pinocchio, data.shared.pinocchio, self._frame_id,
                                                    pinocchio.ReferenceFrame.LOCAL)[3:, :]
        data.J[:, :] = np.dot(data.rJf, data.fJf)
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack([data.J, np.zeros((self.activation.nr, self.state.nv))])
        data.Lx[:] = np.hstack([np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)])
        data.Lxx[:, :] = np.vstack([
            np.hstack(
                [np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                 np.zeros((self.state.nv, self.state.nv))]),
            np.zeros((self.state.nv, self.state.ndx))
        ])

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
        activation = activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._velocity = velocity

    def calc(self, data, x, u):
        data.residual.r[:] = (
            pinocchio.getFrameVelocity(self.state.pinocchio, data.shared.pinocchio, self._frame_id, pinocchio.LOCAL) -
            self._velocity).vector
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u):
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(self.state.pinocchio, data.shared.pinocchio,
                                                                           data.joint, pinocchio.ReferenceFrame.LOCAL)

        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack([np.dot(data.fXj, v_partial_dq), np.dot(data.fXj, v_partial_dv)])
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)
        data.Lxx[:, :] = np.dot(data.residual.Rx.T, np.dot(data.activation.Arr, data.residual.Rx))

    def createData(self, collector):
        data = FrameVelocityCostDataDerived(self, collector)
        return data


class FrameVelocityCostDataDerived(crocoddyl.CostDataAbstract):

    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.fXj = model.state.pinocchio.frames[model._frame_id].placement.inverse().action
        self.joint = model.state.pinocchio.frames[model._frame_id].parent


class Contact3DModelDerived(crocoddyl.ContactModelAbstract):

    def __init__(self, state, xref, gains=[0., 0.]):
        crocoddyl.ContactModelAbstract.__init__(self, state, 3)
        self.xref = xref
        self.gains = gains
        self.joint = state.pinocchio.frames[xref.id].parent

    def calc(self, data, x):
        assert (self.xref.translation is not None or self.gains[0] == 0.)
        v = pinocchio.getFrameVelocity(self.state.pinocchio, data.pinocchio, self.xref.id)
        data.vw[:] = v.angular
        data.vv[:] = v.linear

        fJf = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.xref.id,
                                         pinocchio.ReferenceFrame.LOCAL)
        data.Jc = fJf[:3, :]
        data.Jw[:, :] = fJf[3:, :]

        data.a0[:] = pinocchio.getFrameAcceleration(self.state.pinocchio, data.pinocchio,
                                                    self.xref.id).linear + np.cross(data.vw, data.vv)
        if self.gains[0] != 0.:
            data.a0[:] += np.asscalar(
                self.gains[0]) * (data.pinocchio.oMf[self.xref.id].translation - self.xref.translation)
        if self.gains[1] != 0.:
            data.a0[:] += np.asscalar(self.gains[1]) * data.vv

    def calcDiff(self, data, x):
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, self.joint, pinocchio.ReferenceFrame.LOCAL)

        data.vv_skew = pinocchio.skew(data.vv)
        data.vw_skew = pinocchio.skew(data.vw)
        fXjdv_dq = np.dot(data.fXj, v_partial_dq)
        da0_dq = np.dot(data.fXj, a_partial_dq)[:3, :]
        da0_dq += np.dot(data.vw_skew, fXjdv_dq[:3, :])
        da0_dq -= np.dot(data.vv_skew, fXjdv_dq[3:, :])
        da0_dv = np.dot(data.fXj, a_partial_dv)[:3, :]
        da0_dv += np.dot(data.vw_skew, data.Jc)
        da0_dv -= np.dot(data.vv_skew, data.Jw)

        if np.asscalar(self.gains[0]) != 0.:
            R = data.pinocchio.oMf[self.xref.id].rotation
            da0_dq += np.asscalar(self.gains[0]) * np.dot(
                R,
                pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.xref.id,
                                           pinocchio.ReferenceFrame.LOCAL)[:3, :])
        if np.asscalar(self.gains[1]) != 0.:
            da0_dq += np.asscalar(self.gains[1]) * np.dot(data.fXj[:3, :], v_partial_dq)
            da0_dv += np.asscalar(self.gains[1]) * np.dot(data.fXj[:3, :], a_partial_da)
        data.da0_dx[:, :] = np.hstack([da0_dq, da0_dv])

    def createData(self, data):
        data = Contact3DDataDerived(self, data)
        return data


class Contact3DDataDerived(crocoddyl.ContactDataAbstract):

    def __init__(self, model, data):
        crocoddyl.ContactDataAbstract.__init__(self, model, data)
        self.fXj = model.state.pinocchio.frames[model.xref.id].placement.inverse().action
        self.vw = np.zeros(3)
        self.vv = np.zeros(3)
        self.Jw = np.zeros((3, model.state.nv))
        self.vv_skew = np.zeros((3, 3))
        self.vw_skew = np.zeros((3, 3))


class Contact6DModelDerived(crocoddyl.ContactModelAbstract):

    def __init__(self, state, Mref, gains=[0., 0.]):
        crocoddyl.ContactModelAbstract.__init__(self, state, 6)
        self.Mref = Mref
        self.gains = gains

    def calc(self, data, x):
        assert (self.Mref.placement is not None or self.gains[0] == 0.)
        data.Jc[:, :] = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.Mref.id,
                                                   pinocchio.ReferenceFrame.LOCAL)
        data.a0[:] = pinocchio.getFrameAcceleration(self.state.pinocchio, data.pinocchio, self.Mref.id).vector
        if self.gains[0] != 0.:
            data.rMf = self.Mref.placement.inverse() * data.pinocchio.oMf[self.Mref.id]
            data.a0[:] += np.asscalar(self.gains[0]) * pinocchio.log6(data.rMf).vector
        if self.gains[1] != 0.:
            v = pinocchio.getFrameVelocity(self.state.pinocchio, data.pinocchio, self.Mref.id).vector
            data.a0[:] += np.asscalar(self.gains[1]) * v

    def calcDiff(self, data, x):
        v_partial_dq, a_partial_dq, a_partial_dv, a_partial_da = pinocchio.getJointAccelerationDerivatives(
            self.state.pinocchio, data.pinocchio, data.joint, pinocchio.ReferenceFrame.LOCAL)

        data.da0_dq[:, :] = np.dot(data.fXj, a_partial_dq)
        data.da0_dv[:, :] = np.dot(data.fXj, a_partial_dv)

        if np.asscalar(self.gains[0]) != 0.:
            data.da0_dq += np.asscalar(self.gains[0]) * np.dot(pinocchio.Jlog6(data.rMf), data.Jc)
        if np.asscalar(self.gains[1]) != 0.:
            data.da0_dq += np.asscalar(self.gains[1]) * np.dot(data.fXj, v_partial_dq)
            data.da0_dv += np.asscalar(self.gains[1]) * np.dot(data.fXj, a_partial_da)
        data.da0_dx = np.hstack([data.da0_dq, data.da0_dv])

    def createData(self, data):
        data = Contact6DDataDerived(self, data)
        return data


class Contact6DDataDerived(crocoddyl.ContactDataAbstract):

    def __init__(self, model, data):
        crocoddyl.ContactDataAbstract.__init__(self, model, data)
        self.fXj = model.state.pinocchio.frames[model.Mref.id].placement.inverse().action
        self.da0_dq = np.zeros((6, model.state.nv))
        self.da0_dv = np.zeros((6, model.state.nv))
        self.rMf = pinocchio.SE3.Identity()
        self.joint = model.state.pinocchio.frames[model.Mref.id].parent


class Impulse3DModelDerived(crocoddyl.ImpulseModelAbstract):

    def __init__(self, state, frame):
        crocoddyl.ImpulseModelAbstract.__init__(self, state, 3)
        self.frame = frame

    def calc(self, data, x):
        data.Jc[:, :] = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.frame,
                                                   pinocchio.ReferenceFrame.LOCAL)[:3, :]

    def calcDiff(self, data, x):
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(self.state.pinocchio, data.pinocchio,
                                                                           data.joint, pinocchio.ReferenceFrame.LOCAL)
        data.dv0_dq[:, :] = np.dot(data.fXj[:3, :], v_partial_dq)

    def createData(self, data):
        data = Impulse3DDataDerived(self, data)
        return data


class Impulse3DDataDerived(crocoddyl.ImpulseDataAbstract):

    def __init__(self, model, data):
        crocoddyl.ImpulseDataAbstract.__init__(self, model, data)
        self.fXj = model.state.pinocchio.frames[model.frame].placement.inverse().action
        self.joint = model.state.pinocchio.frames[model.frame].parent


class Impulse6DModelDerived(crocoddyl.ImpulseModelAbstract):

    def __init__(self, state, frame):
        crocoddyl.ImpulseModelAbstract.__init__(self, state, 6)
        self.frame = frame

    def calc(self, data, x):
        data.Jc[:, :] = pinocchio.getFrameJacobian(self.state.pinocchio, data.pinocchio, self.frame,
                                                   pinocchio.ReferenceFrame.LOCAL)

    def calcDiff(self, data, x):
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(self.state.pinocchio, data.pinocchio,
                                                                           data.joint, pinocchio.ReferenceFrame.LOCAL)
        data.dv0_dq[:, :] = np.dot(data.fXj, v_partial_dq)

    def createData(self, data):
        data = Impulse6DDataDerived(self, data)
        return data


class Impulse6DDataDerived(crocoddyl.ImpulseDataAbstract):

    def __init__(self, model, data):
        crocoddyl.ImpulseDataAbstract.__init__(self, model, data)
        self.fXj = model.state.pinocchio.frames[model.frame].placement.inverse().action
        self.joint = model.state.pinocchio.frames[model.frame].parent


class DDPDerived(crocoddyl.SolverAbstract):

    def __init__(self, shootingProblem):
        crocoddyl.SolverAbstract.__init__(self, shootingProblem)
        self.allocateData()  # TODO remove it?

        self.isFeasible = False
        self.alphas = [2**(-n) for n in range(10)]
        self.th_grad = 1e-12

        self.callbacks = None
        self.x_reg = 0
        self.u_reg = 0
        self.reg_incFactor = 10
        self.reg_decFactor = 10
        self.reg_max = 1e9
        self.reg_min = 1e-9
        self.th_step = .5

    def solve(self, init_xs=[], init_us=[], maxiter=100, isFeasible=False, regInit=None):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.x_reg = regInit if regInit is not None else self.reg_min
        self.u_reg = regInit if regInit is not None else self.reg_min
        self.wasFeasible = False
        for i in range(maxiter):
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    recalc = False
                    self.increaseRegularization()
                    if self.x_reg == self.reg_max:
                        return self.xs, self.us, False
                    else:
                        continue
                break
            self.d = self.expectedImprovement()
            d1, d2 = np.asscalar(self.d[0]), np.asscalar(self.d[1])

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
                if self.x_reg == self.reg_max:
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
        return sum([np.dot(q.T, q) for q in self.Qu])

    def expectedImprovement(self):
        d1 = sum([np.dot(q.T, k) for q, k in zip(self.Qu, self.k)])
        d2 = sum([-np.dot(k.T, np.dot(q, k)) for q, k in zip(self.Quu, self.k)])
        return np.array([d1, d2])

    def calcDiff(self):
        if self.iter == 0:
            self.problem.calc(self.xs, self.us)
        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.fs[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.x0)
            for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
                self.fs[i + 1] = m.state.diff(x, d.xnext)
        return self.cost

    def backwardPass(self):
        self.Vx[-1][:] = self.problem.terminalData.Lx
        self.Vxx[-1][:, :] = self.problem.terminalData.Lxx

        if self.x_reg != 0:
            ndx = self.problem.terminalModel.state.ndx
            self.Vxx[-1][range(ndx), range(ndx)] += self.x_reg

        # Compute and store the Vx gradient at end of the interval (rollout state)
        if not self.isFeasible:
            self.Vx[-1] += np.dot(self.Vxx[-1], self.fs[-1])

        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.Qxx[t][:, :] = data.Lxx + np.dot(data.Fx.T, np.dot(self.Vxx[t + 1], data.Fx))
            self.Qxu[t][:, :] = data.Lxu + np.dot(data.Fx.T, np.dot(self.Vxx[t + 1], data.Fu))
            self.Quu[t][:, :] = data.Luu + np.dot(data.Fu.T, np.dot(self.Vxx[t + 1], data.Fu))
            self.Qx[t][:] = data.Lx + np.dot(data.Fx.T, self.Vx[t + 1])
            self.Qu[t][:] = data.Lu + np.dot(data.Fu.T, self.Vx[t + 1])

            if self.u_reg != 0:
                self.Quu[t][range(model.nu), range(model.nu)] += self.u_reg

            self.computeGains(t)

            self.Vx[t][:] = self.Qx[t] - np.dot(self.K[t].T, self.Qu[t])
            self.Vxx[t][:, :] = self.Qxx[t] - np.dot(self.Qxu[t], self.K[t])
            self.Vxx[t][:, :] = 0.5 * (self.Vxx[t][:, :] + self.Vxx[t][:, :].T)  # ensure symmetric

            if self.x_reg != 0:
                self.Vxx[t][range(model.state.ndx), range(model.state.ndx)] += self.x_reg

            # Compute and store the Vx gradient at end of the interval (rollout state)
            if not self.isFeasible:
                self.Vx[t] += np.dot(self.Vxx[t], self.fs[t])

            raiseIfNan(self.Vxx[t], ArithmeticError('backward error'))
            raiseIfNan(self.Vx[t], ArithmeticError('backward error'))

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

    def increaseRegularization(self):
        self.x_reg *= self.reg_incFactor
        if self.x_reg > self.reg_max:
            self.x_reg = self.reg_max
        self.u_reg = self.x_reg

    def decreaseRegularization(self):
        self.x_reg /= self.reg_decFactor
        if self.x_reg < self.reg_min:
            self.x_reg = self.reg_min
        self.u_reg = self.x_reg

    def allocateData(self):
        models = self.problem.runningModels.tolist() + [self.problem.terminalModel]
        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in models]
        self.Vx = [np.zeros([m.state.ndx]) for m in models]

        self.Q = [np.zeros([m.state.ndx + m.nu, m.state.ndx + m.nu]) for m in self.problem.runningModels]
        self.q = [np.zeros([m.state.ndx + m.nu]) for m in self.problem.runningModels]
        self.Qxx = [Q[:m.state.ndx, :m.state.ndx] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qxu = [Q[:m.state.ndx, m.state.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qux = [Qxu.T for m, Qxu in zip(self.problem.runningModels, self.Qxu)]
        self.Quu = [Q[m.state.ndx:, m.state.ndx:] for m, Q in zip(self.problem.runningModels, self.Q)]
        self.Qx = [q[:m.state.ndx] for m, q in zip(self.problem.runningModels, self.q)]
        self.Qu = [q[m.state.ndx:] for m, q in zip(self.problem.runningModels, self.q)]

        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]

        self.xs_try = [self.problem.x0] + [np.nan * self.problem.x0] * self.problem.T
        self.us_try = [np.nan] * self.problem.T
        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)
                   ] + [np.zeros(m.state.ndx) for m in self.problem.runningModels]


class FDDPDerived(DDPDerived):

    def __init__(self, shootingProblem):
        DDPDerived.__init__(self, shootingProblem)

        self.th_acceptNegStep = 2.
        self.dg = 0.
        self.dq = 0.
        self.dv = 0.

    def solve(self, init_xs=[], init_us=[], maxiter=100, isFeasible=False, regInit=None):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.x_reg = regInit if regInit is not None else self.reg_min
        self.u_reg = regInit if regInit is not None else self.reg_min
        self.wasFeasible = False
        for i in range(maxiter):
            recalc = True
            while True:
                try:
                    self.computeDirection(recalc=recalc)
                except ArithmeticError:
                    recalc = False
                    self.increaseRegularization()
                    if self.x_reg == self.reg_max:
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
                if self.x_reg == self.reg_max:
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
        self.dg = 0.
        self.dq = 0.
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
        self.dv = 0.
        if not self.isFeasible:
            dx = self.problem.runningModels[-1].state.diff(self.xs_try[-1], self.xs[-1])
            self.dv -= np.dot(self.fs[-1].T, np.dot(self.Vxx[-1], dx))
            for t in range(self.problem.T):
                dx = self.problem.runningModels[t].state.diff(self.xs_try[t], self.xs[t])
                self.dv -= np.dot(self.fs[t].T, np.dot(self.Vxx[t], dx))
        d1 = self.dg + self.dv
        d2 = self.dq - 2 * self.dv
        return np.array([d1, d2])

    def calcDiff(self):
        self.cost = self.problem.calc(self.xs, self.us)
        self.cost = self.problem.calcDiff(self.xs, self.us)
        if not self.isFeasible:
            self.fs[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.x0)
            for i, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
                self.fs[i + 1] = m.state.diff(x, d.xnext)
        elif not self.wasFeasible:
            self.fs[:] = [np.zeros_like(f) for f in self.fs]
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
                xtry[t] = m.state.integrate(xnext, self.fs[t] * (stepLength - 1))
            utry[t] = us[t] - self.k[t] * stepLength - np.dot(self.K[t], m.state.diff(xs[t], xtry[t]))
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
            xtry[-1] = self.problem.terminalModel.state.integrate(xnext, self.fs[-1] * (stepLength - 1))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, ArithmeticError('forward error'))
        self.cost_try = ctry
        return xtry, utry, ctry
